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
    while len(blob) % 16:                     # 16 — вимога графічного чипа до адрес текстур
        blob.append(0)
    off = len(blob)
    blob.extend(b)
    return off


# ---- map chunks -> окремий файл MAP.BIN поруч з EBOOT (карта завелика для вбудовування)
#      формат KHM2: "KHM2", u32 NCX, u32 NCY, u32 CS, u32 off[NCX*NCY+1] (від початку файлу), далі zlib(
#        пікселі CS*CS u16 (PSP 5650: R біти 0-4, G 5-10, B 11-15) + біти твердості CS*CS/8 + поверхня CS*CS/2 (4 біти: 0 тверда, 1 ґрунт, 2 вода))
import struct
rgbm = np.load(os.path.join(BUILD, 'map_rgb.npy'), mmap_mode='r')
surfm = np.load(os.path.join(BUILD, 'surf.npy'), mmap_mode='r')
sbits = np.packbits(solid.astype(np.uint8), axis=1, bitorder='little')     # (H, W/8)
import cv2
SHARP = float(os.environ.get('SHARP', '0.6'))         # різкість карти (нерізка маска, сигма 0,8); 0 — вимкнено
chunks = []
for cy in range(NCY):
    if SHARP > 0:
        y0 = max(0, cy * CS - 4); y1 = min(H, (cy + 1) * CS + 4)
        bf = np.asarray(rgbm[y0:y1]).astype(np.float32)
        bf = bf * (1 + SHARP) - cv2.GaussianBlur(bf, (0, 0), 0.8) * SHARP
        band = np.clip(bf[cy * CS - y0:cy * CS - y0 + CS] + 0.5, 0, 255).astype(np.uint16)
    else:
        band = np.asarray(rgbm[cy * CS:(cy + 1) * CS]).astype(np.uint16)
    p565 = (band[..., 0] >> 3) | ((band[..., 1] >> 2) << 5) | ((band[..., 2] >> 3) << 11)
    sb_ = np.asarray(surfm[cy * CS:(cy + 1) * CS])
    s4 = (sb_[:, 0::2] & 15) | ((sb_[:, 1::2] & 15) << 4)
    for cx in range(NCX):
        pxl = p565[:, cx * CS:(cx + 1) * CS].astype('<u2').tobytes()
        sb = sbits[cy * CS:(cy + 1) * CS, cx * CS // 8:(cx + 1) * CS // 8].tobytes()
        su = s4[:, cx * CS // 2:(cx + 1) * CS // 2].astype(np.uint8).tobytes()
        chunks.append(zlib.compress(pxl + sb + su, 9))
hdr = 16 + 4 * (len(chunks) + 1)
offs = [hdr]
for c in chunks: offs.append(offs[-1] + len(c))
with open(os.path.join(BUILD, 'MAP.BIN'), 'wb') as f:
    f.write(b'KHM2' + struct.pack('<3I', NCX, NCY, CS) + struct.pack('<%dI' % len(offs), *offs))
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
CARTEX_OFF, CARTEX_LIT, CARTEX_SHD = (put(t) for t in sprites.car_textures())
# ---- дерева: стовбури (колізія) і крони (поверх машини), відсортовані за блоком карти 64x64
_tr = json.load(open(os.path.join(BUILD, 'trees.json'))) if os.path.exists(os.path.join(BUILD, 'trees.json')) else []
_pl = np.load(os.path.join(BUILD, 'play.npy'), mmap_mode='r')
_tr = [t for t in _tr if 0 <= t[0] < W and 0 <= t[1] < H and _pl[int(t[1]), int(t[0])]]
_tr.sort(key=lambda t: (t[1] // CS) * NCX + t[0] // CS)
_cells = np.zeros(NCX * NCY + 1, np.uint32)
for t in _tr: _cells[(t[1] // CS) * NCX + t[0] // CS + 1] += 1
_cells = np.cumsum(_cells).astype(np.uint32)
# густі гаї: стовбур-перешкода лише якщо поруч (12 пікс = 6 м) немає іншого — щоб між стовбурами завжди проходила машина;
# решта дерев — лише крона (від'ємний радіус)
_coll = np.zeros(len(_tr), bool); _grid = {}
for i in sorted(range(len(_tr)), key=lambda i: -_tr[i][2]):
    x, y = _tr[i][0], _tr[i][1]; gx, gy = x // 12, y // 12
    if any((x - _tr[j][0]) ** 2 + (y - _tr[j][1]) ** 2 < 144 for a in (-1, 0, 1) for b in (-1, 0, 1) for j in _grid.get((gx + a, gy + b), ())):
        continue
    _coll[i] = 1; _grid.setdefault((gx, gy), []).append(i)
print('trees', len(_tr), 'trunks', int(_coll.sum()))
TREE_OFF = put(np.array([[t[0], t[1], max(1, int(round(t[2]))) * (1 if c else -1)] for t, c in zip(_tr, _coll)] or [[0, 0, 0]], '<i2').tobytes())
TREE_CELL_OFF = put(_cells.astype('<u4').tobytes())
NTREE = len(_tr)
# ---- будинки для паралаксу дахів: центр, висота (м), контур (≤32 вершини) для процесора +
#      розширений на ~0,5 пікс контур, розбитий на трикутники, — для графічного чипа (плавний зсув зі згладженими краями)
import cv2


def earclip(P):
    """трикутники простого многокутника (індекси); при збої — віяло"""
    P = [tuple(map(float, p)) for p in P]
    n = len(P)
    if n < 3: return []
    area = sum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    def cross(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1; found = False
        for k in range(len(idx)):
            ia, ib, ic = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            A, B, C = P[ia], P[ib], P[ic]
            cr = cross(A, B, C)
            if abs(cr) < 1e-6: idx.pop(k); found = True; break        # на одній прямій
            if cr < 0: continue
            if any(cross(A, B, P[j]) >= 0 and cross(B, C, P[j]) >= 0 and cross(C, A, P[j]) >= 0 for j in idx if j not in (ia, ib, ic)):
                continue
            tris.append((ia, ib, ic)); idx.pop(k); found = True; break
        if not found: break
    if len(idx) >= 3:
        tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return tris


def expand(Q):
    x0, y0 = np.floor(Q.min(0)).astype(int) - 4
    x1, y1 = np.ceil(Q.max(0)).astype(int) + 4
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round((Q - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea).astype(np.float32)
    eps = 0.6
    while True:
        e = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(e) <= 32: break
        eps *= 1.4
    return (e + [x0, y0]).astype(np.int16) if len(e) >= 3 else None


_D = CD.data()
_arch = [np.array(q, np.float32) for q in md.get('arches', [])]
_bl = []
for b in _D['B'] + _D['SAT']:
    Q = np.array(b['q'], np.float32) * CD.K
    if len(Q) < 3: continue
    cx, cy = Q.mean(0)
    if not (0 <= cx < W and 0 <= cy < H) or not _pl[int(cy), int(cx)]: continue
    if cv2.contourArea(Q) < 12: continue
    if any(cv2.pointPolygonTest(A, (float(cx), float(cy)), False) >= 0 for A in _arch): continue     # будинки над арками — окремо
    eps = 0.7
    while True:
        q = cv2.approxPolyDP(Q, eps, True)[:, 0, :]
        if len(q) <= 32: break
        eps *= 1.4
    if len(q) < 3: continue
    e = expand(Q)
    if e is None: continue
    tr = earclip(e)
    if not tr: continue
    _bl.append((int(cx), int(cy), int(round(b.get('h') or 6)), np.round(q).astype(np.int16), e, np.array(tr, np.uint8)))
_bl.sort(key=lambda t: (t[1] // CS) * NCX + t[0] // CS)
_bc = np.zeros(NCX * NCY + 1, np.uint32)
for t in _bl: _bc[(t[1] // CS) * NCX + t[0] // CS + 1] += 1
_bc = np.cumsum(_bc).astype(np.uint32)
def _cum(lens):
    o = np.zeros(len(lens) + 1, np.uint32); o[1:] = np.cumsum(lens); return o
_po = _cum([len(t[3]) for t in _bl]); _eo = _cum([len(t[4]) for t in _bl]); _to = _cum([len(t[5]) for t in _bl])
BLD_OFF = put(np.array([[t[0], t[1], t[2], len(t[3])] for t in _bl] or [[0, 0, 0, 0]], '<i2').tobytes())
BLD_PO_OFF = put(_po.astype('<u4').tobytes())
BLD_PT_OFF = put((np.concatenate([t[3] for t in _bl]) if _bl else np.zeros((1, 2), np.int16)).astype('<i2').tobytes())
BLD_EO_OFF = put(_eo.astype('<u4').tobytes())
BLD_EP_OFF = put((np.concatenate([t[4] for t in _bl]) if _bl else np.zeros((1, 2), np.int16)).astype('<i2').tobytes())
BLD_TO_OFF = put(_to.astype('<u4').tobytes())
BLD_TR_OFF = put((np.concatenate([t[5] for t in _bl]) if _bl else np.zeros((1, 3), np.uint8)).astype(np.uint8).tobytes())
BLD_CELL_OFF = put(_bc.astype('<u4').tobytes())
NBLD = len(_bl)
print('buildings for parallax', NBLD, 'points', int(_po[-1]), 'expanded', int(_eo[-1]), 'tris', int(_to[-1]))
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
cps = list(range(32, 127)) + [0xB7] + list(range(0x400, 0x460)) + [0x490, 0x491, 0x2014, 0x2019, 0x2116]
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
    f.write('#define TREE_OFF %d\n#define TREE_CELL_OFF %d\n#define NTREE %d\n' % (TREE_OFF, TREE_CELL_OFF, NTREE))
    f.write('#define BLD_OFF %d\n#define BLD_PO_OFF %d\n#define BLD_PT_OFF %d\n#define BLD_CELL_OFF %d\n#define NBLD %d\n' % (BLD_OFF, BLD_PO_OFF, BLD_PT_OFF, BLD_CELL_OFF, NBLD))
    f.write('#define BLD_EO_OFF %d\n#define BLD_EP_OFF %d\n#define BLD_TO_OFF %d\n#define BLD_TR_OFF %d\n' % (BLD_EO_OFF, BLD_EP_OFF, BLD_TO_OFF, BLD_TR_OFF))
    f.write('#define CARTEX_OFF %d\n#define CARTEX_LIT %d\n#define CARTEX_SHD %d\n#define CARTEX_W %d\n#define CARTEX_H %d\n#define CARTEX_LEN %d\n'
            % (CARTEX_OFF, CARTEX_LIT, CARTEX_SHD, sprites.CARTEX_W, sprites.CARTEX_H, sprites.CARTEX_LEN))
    widx = json.load(open(os.path.join(BUILD, 'water_idx.json'))) if os.path.exists(os.path.join(BUILD, 'water_idx.json')) else []
    f.write('static const uint8_t WATERIDX[256] = {%s};\n' % ','.join('1' if i in widx else '0' for i in range(256)))
    sto = md.get('sto') or [dict(x=md['spawn'][0], y=md['spawn'][1], name='СТО')]
    f.write('typedef struct { int16_t x, y; const char *name; } Sto;\n')
    f.write('static const Sto STOS[] = {%s};\n#define NSTO %d\n' % (','.join('{%d,%d,%s}' % (a['x'], a['y'], cstr(a['name'])) for a in sto), len(sto)))
    fuel = md.get('fuel') or [dict(x=md['spawn'][0], y=md['spawn'][1], name='АЗС')]
    f.write('static const Sto FUELS[] = {%s};\n#define NFUEL %d\n' % (','.join('{%d,%d,%s}' % (a['x'], a['y'], cstr(a['name'])) for a in fuel), len(fuel)))
    ar = md.get('arches', [])
    f.write('typedef struct { int16_t n, x0, y0, x1, y1; int16_t p[64][2]; } Arch;\n')
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
