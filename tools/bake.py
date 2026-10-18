"""Pack map, collision, minimap, sprites, font and game data into assets.bin + assets.h"""
import os, sys, json, zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palgame import PAL, R, dark_lut, MAP_START, MAP_COLORS, offroad
import citydraw as CD
import sprites

BUILD = sys.argv[1]
OUT = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))

img = np.load(os.path.join(BUILD, 'map.npy'))
solid = np.load(os.path.join(BUILD, 'solid.npy'))
md = json.load(open(os.path.join(BUILD, 'mapdata.json'), encoding='utf-8'))
H, W = img.shape
CS = 64
NCX, NCY = W // CS, H // CS
assert NCX * CS == W and NCY * CS == H

blob = bytearray()


def put(b):
    while len(blob) % 4:
        blob.append(0)
    off = len(blob)
    blob.extend(b)
    return off


# ---- map chunks -> окремий файл MAP.BIN поруч з EBOOT (карта завелика для вбудовування)
#      формат: "KHM1", u32 NCX, u32 NCY, u32 CS, u32 off[NCX*NCY+1] (від початку файлу), далі zlib(пікселі CS*CS + біти твердості CS*CS/8)
import struct
sbits = np.packbits(solid.astype(np.uint8), axis=1, bitorder='little')     # (H, W/8)
chunks = []
for cy in range(NCY):
    for cx in range(NCX):
        pxl = img[cy * CS:(cy + 1) * CS, cx * CS:(cx + 1) * CS].tobytes()
        sb = sbits[cy * CS:(cy + 1) * CS, cx * CS // 8:(cx + 1) * CS // 8].tobytes()
        chunks.append(zlib.compress(pxl + sb, 9))
hdr = 16 + 4 * (len(chunks) + 1)
offs = [hdr]
for c in chunks: offs.append(offs[-1] + len(c))
with open(os.path.join(BUILD, 'MAP.BIN'), 'wb') as f:
    f.write(b'KHM1' + struct.pack('<3I', NCX, NCY, CS) + struct.pack('<%dI' % len(offs), *offs))
    for c in chunks: f.write(c)
csizes = [len(c) for c in chunks]
MAPBIN_SIZE = offs[-1]

# ---- minimap (1/8): average colour -> nearest palette entry (рядками, щоб не тримати всю карту в RGB)
pal = np.array(PAL, np.float32)
MS = 8
mh, mw = H // MS, W // MS
avg = np.zeros((mh, mw, 3), np.float32)
for r0 in range(0, mh, 64):
    r1 = min(mh, r0 + 64)
    blk = pal[np.clip(img[r0 * MS:r1 * MS, :mw * MS], 0, len(PAL) - 1)]
    avg[r0:r1] = blk.reshape(r1 - r0, MS, mw, MS, 3).mean(axis=(1, 3))
playz = np.load(os.path.join(BUILD, 'play.npy'))
cand = list(range(MAP_START, MAP_START + len(MAP_COLORS)))
ASPH_IDS = [MAP_START + k for k, c in enumerate(MAP_COLORS) if tuple(c) in [tuple(a) for a in CD.ASPH]]
cp = pal[cand]
best = np.zeros((mh, mw), np.uint8)
for y in range(mh):
    d = ((avg[y][:, None, :] - cp[None]) ** 2).sum(-1)
    best[y] = np.array(cand, np.uint8)[d.argmin(-1)]
outside = playz[MS // 2::MS, MS // 2::MS][:mh, :mw] == 0
best_lit = best.copy()
best[outside] = np.array(dark_lut(), np.uint8)[best[outside]]
mini = best
MINI_OFF = put(zlib.compress(mini.tobytes(), 9))
MINI_ZLEN = len(blob) - MINI_OFF

# ---- велика карта (SELECT): вміщається в екран PSP (≤ 470×250), крок кратний мінікарті
FB = 4
while (W // (MS * FB)) > 470 or (H // (MS * FB)) > 250: FB += 1
BS_ = MS * FB
bh, bw = H // BS_, W // BS_
mm = best_lit[:bh * FB, :bw * FB]
# дороги краще видно, якщо блок з достатньою часткою асфальту малювати асфальтом (тонкі вулиці не губляться)
mmb = mm.reshape(bh, FB, bw, FB).transpose(0, 2, 1, 3).reshape(bh, bw, FB * FB)
roadish = np.isin(mmb, ASPH_IDS)
big = np.where(roadish.sum(-1) >= FB * FB // 4, ASPH_IDS[0] if ASPH_IDS else mmb[:, :, FB * FB // 2], mmb[:, :, FB * FB // 2 + FB // 2]).astype(np.uint8)
DK = np.array(dark_lut(), np.uint8)
ob = outside[FB // 2::FB, FB // 2::FB][:bh, :bw]
big[ob] = DK[DK[big[ob]]]
BIG_OFF = put(zlib.compress(big.tobytes(), 9))
BIG_ZLEN = len(blob) - BIG_OFF

# ---- sprites
fr = sprites.car_frames()
SPR_OFF = put(b''.join(f.tobytes() for f in fr))
SPR_LIT = put(b''.join(f.tobytes() for f in sprites.car_frames(lit=True)))
ICON_STO = put(sprites.icon('sto').tobytes())
ICON_FUEL = put(sprites.icon('fuel').tobytes())
ICON_SHOP = put(sprites.icon('shop').tobytes())
ICON_HOME = put(sprites.icon('home').tobytes())

# ---- пам'ятки: спрайти-позначки (індекси палітри, 0 = прозоро) + таблиця
import landmarks as LMK
LM = json.load(open(os.path.join(BUILD, 'landmarks.json'), encoding='utf-8')) if os.path.exists(os.path.join(BUILD, 'landmarks.json')) else []
palf = np.array(PAL, np.float32)
lm_rows = []
for e in LM:
    frs = LMK.sprite(e)
    h, w = frs[0].shape[:2]
    offs = []
    for fr in frs:
        rgb_ = fr[..., :3].reshape(-1, 3).astype(np.float32)
        d = ((rgb_[:, None, :] - palf[None, 1:]) ** 2).sum(-1)
        idx = (d.argmin(1) + 1).astype(np.uint8)
        idx[fr[..., 3].reshape(-1) == 0] = 0
        offs.append(put(idx.tobytes()))
    lm_rows.append((e['x'], e['y'], w, h, len(frs), offs[0], e['r'], e['label'], e['bit']))

open(os.path.join(OUT, 'assets.bin'), 'wb').write(blob)

# ---- шрифт: Rubik (OFL), пропорційний, 1 біт; розміри для sc = 1..5 (висота великих літер ≈ 8*sc)
cps = list(range(32, 127)) + list(range(0x400, 0x460)) + [0x490, 0x491, 0x2019, 0x2116]
FONT_PX = {1: 11, 2: 20, 3: 29, 4: 38, 5: 47}
FD = os.path.join(HERE, 'fonts')
font_bits = bytearray(); font_tabs = {}
for sc, px in FONT_PX.items():
    wgt = '500' if sc == 1 else '700'
    fl = ImageFont.truetype(os.path.join(FD, 'rubik-latin-%s.ttf' % wgt), px)
    fc = ImageFont.truetype(os.path.join(FD, 'rubik-cyrillic-%s.ttf' % wgt), px)
    from fontTools.ttLib import TTFont
    cml = TTFont(os.path.join(FD, 'rubik-latin-%s.ttf' % wgt)).getBestCmap(); cmc = TTFont(os.path.join(FD, 'rubik-cyrillic-%s.ttf' % wgt)).getBestCmap()
    captop = fc.getbbox('Н')[1]
    top = min(fc.getbbox('ЙЇЁй')[1], fl.getbbox('ÉÀ')[1] if False else fc.getbbox('Й')[1])
    asc = captop - top                                   # над великими літерами (Й, Ї)
    bottom = max(fc.getbbox('Щуд')[3], fl.getbbox('gjy')[3])
    ch_h = bottom - top + 1
    rows = []
    for c in cps:
        f = fl if c in cml else (fc if c in cmc else None)
        if f is None: continue
        pad = px
        im = Image.new('L', (px * 3, ch_h), 0); dd = ImageDraw.Draw(im); dd.fontmode = '1'
        dd.text((pad, -top), chr(c), font=f, fill=255)
        a_ = np.array(im) > 0
        adv = int(round(f.getlength(chr(c)))) + (1 if sc == 1 else 0)
        cols = np.nonzero(a_.any(0))[0]
        if len(cols) == 0: rows.append((c, adv, 0, 0, len(font_bits))); continue
        x0, x1 = cols.min(), cols.max() + 1
        g = a_[:, x0:x1]; w = x1 - x0; st = (w + 7) // 8
        off = len(font_bits)
        for y in range(ch_h):
            for k in range(st):
                v = 0
                for bit in range(8):
                    x = k * 8 + bit
                    if x < w and g[y, x]: v |= 128 >> bit
                font_bits.append(v)
        rows.append((c, adv, int(x0 - pad), int(w), off))
    font_tabs[sc] = (ch_h, rows, asc)
glyphs = font_tabs[1][1]


def cstr(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


pal32 = []
for i in range(256):
    r, g, b = PAL[i] if i < len(PAL) else (0, 0, 0)
    pal32.append(0xFF000000 | (b << 16) | (g << 8) | r)

with open(os.path.join(OUT, 'assets.h'), 'w', encoding='utf-8') as f:
    f.write('// generated by tools/bake.py\n')
    f.write('extern const uint8_t assets_bin[];\n')
    f.write('#define MAP_W %d\n#define MAP_H %d\n#define CS %d\n#define NCX %d\n#define NCY %d\n' % (W, H, CS, NCX, NCY))
    f.write('#define MPP %.3ff\n' % md['M'])
    f.write('#define MAPBIN_SIZE %d\n' % MAPBIN_SIZE)
    f.write('#define BIG_OFF %d\n#define BIG_ZLEN %d\n#define BIG_W %d\n#define BIG_H %d\n#define BIG_S %d\n' % (BIG_OFF, BIG_ZLEN, bw, bh, BS_))
    f.write('#define MINI_OFF %d\n#define MINI_ZLEN %d\n#define MINI_W %d\n#define MINI_H %d\n#define MINI_S %d\n' % (MINI_OFF, MINI_ZLEN, mw, mh, MS))
    f.write('#define SPR_OFF %d\n#define SPR_SZ %d\n#define SPR_N %d\n#define ICON_SHOP %d\n#define ICON_HOME %d\n#define ICON_SZ 16\n' % (SPR_OFF, sprites.CAR_SZ, sprites.NROT, ICON_SHOP, ICON_HOME))
    f.write('#define SPAWN_X %d\n#define SPAWN_Y %d\n' % tuple(md['spawn']))
    f.write('#define SPR_LIT %d\n#define ICON_STO %d\n#define ICON_FUEL %d\n' % (SPR_LIT, ICON_STO, ICON_FUEL))
    widx = json.load(open(os.path.join(BUILD, 'water_idx.json'))) if os.path.exists(os.path.join(BUILD, 'water_idx.json')) else []
    f.write('static const uint8_t WATERIDX[256] = {%s};\n' % ','.join('1' if i in widx else '0' for i in range(256)))
    sto = md.get('sto') or [dict(x=md['spawn'][0], y=md['spawn'][1], name='СТО')]
    f.write('typedef struct { int16_t x, y; const char *name; } Sto;\n')
    f.write('static const Sto STOS[] = {%s};\n#define NSTO %d\n' % (','.join('{%d,%d,%s}' % (a['x'], a['y'], cstr(a['name'])) for a in sto), len(sto)))
    fuel = md.get('fuel') or [dict(x=md['spawn'][0], y=md['spawn'][1], name='АЗС')]
    f.write('static const Sto FUELS[] = {%s};\n#define NFUEL %d\n' % (','.join('{%d,%d,%s}' % (a['x'], a['y'], cstr(a['name'])) for a in fuel), len(fuel)))
    ar = md.get('arches', [])
    f.write('typedef struct { int16_t n, x0, y0, x1, y1; int16_t p[24][2]; } Arch;\n')
    f.write('static const Arch ARCHES[] = {%s};\n#define NARCH %d\n' % (','.join('{%d,%d,%d,%d,%d,{%s}}' % (len(q), min(a for a, b in q), min(b for a, b in q), max(a for a, b in q), max(b for a, b in q), ','.join('{%d,%d}' % (a, b) for a, b in q)) for q in ar) or '{0}', len(ar)))
    # смуги для помічника: відрізки осі дороги + сітка 128 пікс
    lanes = md.get('lanes', [])
    LG = 128; gw, gh = (W + LG - 1) // LG, (H + LG - 1) // LG
    cells = [[] for _ in range(gw * gh)]
    for k, (x0, y0, x1, y1, off, ow) in enumerate(lanes):
        m = 26
        for cy in range(max(0, (min(y0, y1) - m) // LG), min(gh, (max(y0, y1) + m) // LG + 1)):
            for cx in range(max(0, (min(x0, x1) - m) // LG), min(gw, (max(x0, x1) + m) // LG + 1)):
                cells[cy * gw + cx].append(k)
    starts = [0]; idx = []
    for c in cells: idx.extend(c); starts.append(len(idx))
    assert len(lanes) < 65536
    f.write('#define LANE_G %d\n#define LANE_GW %d\n#define LANE_GH %d\n#define NLANE %d\n' % (LG, gw, gh, len(lanes)))
    f.write('static const int16_t LANES[][6] = {%s};\n' % (','.join('{%d,%d,%d,%d,%d,%d}' % tuple(l) for l in lanes) or '{0}'))
    f.write('static const uint32_t LANE_CELL[] = {%s};\n' % ','.join(map(str, starts)))
    f.write('static const uint16_t LANE_IDX[] = {%s};\n' % (','.join(map(str, idx)) or '0'))
    f.write('typedef struct { int16_t x, y; const char *name; const char *kind; } Poi;\n')
    f.write('static const Poi SHOPS[] = {\n%s};\n' % ',\n'.join('{%d,%d,%s,%s}' % (s['x'], s['y'], cstr(s['name']), cstr(s['kind'])) for s in md['shops']))
    f.write('#define NSHOPS (int)(sizeof(SHOPS)/sizeof(SHOPS[0]))\n')
    f.write('static const Poi ADDRS[] = {\n%s};\n' % ',\n'.join('{%d,%d,%s,""}' % (a['x'], a['y'], cstr(a['name'])) for a in md['addrs']))
    f.write('#define NADDRS (int)(sizeof(ADDRS)/sizeof(ADDRS[0]))\n')
    f.write('static const char *STREETS[] = {%s};\n' % ','.join(cstr(n) for n in md['names']))
    f.write('static const int16_t SEGS[][5] = {%s};\n' % ','.join('{%d,%d,%d,%d,%d}' % tuple(s) for s in md['segs']))
    f.write('#define NSEGS (int)(sizeof(SEGS)/sizeof(SEGS[0]))\n')
    f.write('typedef struct { int16_t x, y; uint8_t w, h, nfr; int32_t off; int16_t r; const char *name; uint8_t bit; } Landmark;\n')
    f.write('static const Landmark LMS[] = {\n%s};\n' % ',\n'.join('{%d,%d,%d,%d,%d,%d,%d,%s,%d}' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], cstr(r[7]), r[8]) for r in lm_rows) if lm_rows else 'static const Landmark LMS[1] = {{0}};\n')
    f.write('#define NLM %d\n' % len(lm_rows))
    f.write('typedef struct { uint16_t cp; int8_t adv, xoff; uint8_t w; uint32_t off; } Glyph;\n')
    f.write('static const uint8_t FONT_DATA[] = {%s};\n' % ','.join(map(str, font_bits)))
    for sc_, (hh, rows, _) in font_tabs.items():
        f.write('static const Glyph FG%d[] = {%s};\n' % (sc_, ','.join('{%d,%d,%d,%d,%d}' % r for r in rows)))
    f.write('typedef struct { const Glyph *g; int n, h, asc; } FontTab;\n')
    f.write('static const FontTab FONTS[6] = {{0,0,0,0},%s};\n' % ','.join('{FG%d,%d,%d,%d}' % (k, len(font_tabs[k][1]), font_tabs[k][0], font_tabs[k][2]) for k in range(1, 6)))
    f.write('static const uint32_t PALETTE[256] = {%s};\n' % ','.join('0x%08X' % v for v in pal32))
    f.write('static const uint8_t DARK[256] = {%s};\n' % ','.join(map(str, dark_lut())))
    # surface classes for driving: 1 = off-road (grass, dirt)
    off = offroad()
    f.write('static const uint8_t OFFROAD[256] = {%s};\n' % ','.join(map(str, off)))

with open(os.path.join(OUT, 'pal_ids.h'), 'w') as f:
    for name, ids in R.items():
        for k, i in enumerate(ids):
            f.write('#define C_%s_%d %d\n' % (name, k, i))

with open(os.path.join(OUT, 'assets_data.c'), 'w') as f:
    f.write('#include <stdint.h>\nconst uint8_t assets_bin[%d] __attribute__((aligned(16))) = {\n' % len(blob))
    for i in range(0, len(blob), 40):
        f.write(','.join(str(b) for b in blob[i:i + 40]) + ',\n')
    f.write('};\n')
print('MAP.BIN MB %.2f  ' % (MAPBIN_SIZE / 1e6) + 'blob MB %.2f  map chunks MB %.2f  max chunk %d  shops %d addrs %d glyphs %d' % (
    len(blob) / 1e6, sum(csizes) / 1e6, max(csizes), len(md['shops']), len(md['addrs']), len(glyphs)))
