"""Courier-on-scooter sprite (32 rotations) + map icons, supersampled and quantised to the palette."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palgame import PAL, R

SS = 4
SZ = 40          # sprite canvas (1x)
K = 1.22         # sprite scale
NROT = 32


def col(i):
    return PAL[i] + (255,)


def draw_scooter(lean=0.0):
    """scooter pointing east (+x), centred, drawn at SS scale"""
    S = SZ * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2

    def P(x, y): return (c + x * SS * K, c + y * SS * K)

    def rect(x0, y0, x1, y1, ci, r=0):
        d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=r * SS * K, fill=col(ci))

    def ell(x, y, rx, ry, ci):
        d.ellipse([P(x - rx, y - ry), P(x + rx, y + ry)], fill=col(ci))

    # wheels
    rect(-12, -1.6, -6.5, 1.6, R['tire'][0], 1.2)
    rect(6.5, -1.5, 12, 1.5, R['tire'][0], 1.2)
    # body
    d.polygon([P(-11, -3.2), P(4, -3.6), P(9, -2.0), P(10.5, 0), P(9, 2.0), P(4, 3.6), P(-11, 3.2)], fill=col(R['red'][1]))
    d.polygon([P(-10, -2.4), P(4, -2.8), P(8.5, -1.2), P(9.5, 0), P(8.5, 1.2), P(4, 2.8), P(-10, 2.4)], fill=col(R['red'][2]))
    d.polygon([P(-6, -1.6), P(4, -2.0), P(7.5, -0.5), P(7.5, 0.3), P(-6, 0.6)], fill=col(R['red'][3]))
    ell(9.6, 0, 1.0, 1.4, R['yellow'][3])            # headlight
    # handlebar
    rect(5.5, -6.2, 7.0, 6.2, R['steel'][1], 0.6)
    rect(5.2, -7.0, 7.4, -5.2, R['tire'][1], 0.5)
    rect(5.2, 5.2, 7.4, 7.0, R['tire'][1], 0.5)
    ell(7.2, -6.6, 0.9, 0.8, R['steel'][2]); ell(7.2, 6.6, 0.9, 0.8, R['steel'][2])   # mirrors
    # delivery box
    rect(-14.5, -4.8, -6.0, 4.8, R['box'][0], 1.0)
    rect(-14.0, -4.3, -6.5, 4.3, R['box'][2], 0.8)
    rect(-13.2, -3.5, -7.3, 3.5, R['box'][3], 0.6)
    rect(-10.6, -3.5, -9.6, 3.5, R['box'][1], 0)
    # rider: legs/seat
    rect(-5.6, -3.6, 0.5, 3.6, R['tire'][1], 1.5)
    # torso / shoulders (jacket)
    ell(-1.0, 0, 3.4, 5.6, R['blue'][0])
    ell(-0.6, 0, 2.8, 5.0, R['blue'][2])
    ell(-0.2, -0.8 + lean, 1.8, 3.2, R['blue'][3])
    # arms
    for sg in (-1, 1):
        d.line([P(0, sg * 4.6), P(4.6, sg * 5.6)], fill=col(R['blue'][1]), width=int(2.4 * SS * K))
        ell(5.6, sg * 5.6, 1.2, 1.2, R['skin'][2])
    # helmet
    ell(0.6, 0, 3.4, 3.4, R['tire'][0])
    ell(0.6, 0, 3.0, 3.0, R['white'][2])
    d.chord([P(-2.4, -3.0), P(3.6, 3.0)], -70, 70, fill=col(R['blue'][1]))
    ell(-0.6, -1.4, 1.0, 0.8, R['white'][2])
    d.line([P(-2.0, 0), P(0.6, 0)], fill=col(R['red'][2]), width=int(1.0 * SS * K))
    return im


def quantise(im, allowed):
    a = np.asarray(im, np.float32)
    al = a[..., 3] / 255.0
    pre = a[..., :3] * al[..., None]
    S = im.size[0] // SS
    pre = pre.reshape(S, SS, S, SS, 3).mean(axis=(1, 3))
    al = al.reshape(S, SS, S, SS).mean(axis=(1, 3))
    rgb = pre / np.maximum(al[..., None], 1e-4)
    pal = np.array([PAL[i] for i in allowed], np.float32)
    dd = ((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    out = np.where(al > 0.45, np.array(allowed, np.uint8)[dd.argmin(-1)], 0).astype(np.uint8)
    m = out > 0
    ol = ndi.binary_dilation(m) & ~m
    out[ol] = R['out'][0]
    return out


def scooter_frames():
    base = draw_scooter()
    allowed = sum((R[k] for k in ('red', 'box', 'blue', 'tire', 'white', 'skin', 'steel', 'yellow', 'ink')), [])
    frames = []
    for i in range(NROT):
        ang = i * 360.0 / NROT
        rot = base.rotate(-ang, resample=Image.BICUBIC, center=(SZ * SS / 2, SZ * SS / 2))
        frames.append(quantise(rot, allowed))
    return frames


CAR_SZ = 28


def draw_golf(lit=False):
    """VW Golf V хетчбек, сріблястий, вид згори, ніс на схід. ~21 x 9,6 пікс (гра 0,5 м/пікс, машина збільшена).
    v2 (18.10.2026): закруглений ніс, капот з ребром, решітка з емблемою, лобове з відблиском, світлий дах,
    коротка корма зі спойлером, ліхтарі, що заходять на боки; варіант lit — стоп-сигнали."""
    S = CAR_SZ * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = S / 2
    def P(x, y): return (c + x * SS, c + y * SS)
    def rr(x0, y0, x1, y1, ci, r=0): d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=r * SS, fill=col(ci))
    def poly(pts, ci): d.polygon([P(x, y) for x, y in pts], fill=col(ci))
    sv, gl, ti, ink = R['silver'], R['glass'], R['tire'], R['ink']
    # колеса трохи виступають з-під кузова
    for x0 in (-7.6, 3.6):
        rr(x0, -5.0, x0 + 3.8, 5.0, ti[0], 0.9)
    # кузов: обвідка, основний тон, світлий верх
    poly([(-10.3, -3.6), (-9.6, -4.6), (6.0, -4.8), (8.8, -4.2), (10.4, -2.6), (10.6, 0), (10.4, 2.6), (8.8, 4.2), (6.0, 4.8), (-9.6, 4.6), (-10.3, 3.6)], ink[1])
    poly([(-9.9, -3.4), (-9.3, -4.2), (6.0, -4.4), (8.5, -3.9), (10.0, -2.4), (10.2, 0), (10.0, 2.4), (8.5, 3.9), (6.0, 4.4), (-9.3, 4.2), (-9.9, 3.4)], sv[1])
    poly([(-9.4, -3.0), (6.0, -3.8), (8.3, -3.3), (9.6, -2.0), (9.7, 0), (9.6, 2.0), (8.3, 3.3), (6.0, 3.8), (-9.4, 3.0)], sv[2])
    # капот з ребром посередині
    poly([(4.2, -3.6), (8.0, -3.1), (9.4, -1.8), (9.5, 0), (9.4, 1.8), (8.0, 3.1), (4.2, 3.6)], sv[3])
    rr(4.6, -0.25, 9.2, 0.25, sv[2], 0)
    # решітка й емблема
    rr(9.6, -1.6, 10.2, 1.6, ink[2], 0.2); rr(9.7, -0.45, 10.2, 0.45, sv[4], 0.4)
    # фари
    poly([(8.6, -3.9), (10.3, -2.5), (10.1, -1.4), (8.9, -2.0)], R['yellow'][3]); poly([(8.6, 3.9), (10.3, 2.5), (10.1, 1.4), (8.9, 2.0)], R['yellow'][3])
    # лобове скло з відблиском
    poly([(1.4, -3.3), (4.3, -3.7), (4.3, 3.7), (1.4, 3.3)], gl[0])
    poly([(2.9, -2.6), (3.8, -2.8), (2.3, 1.2), (1.9, 0.6)], gl[2])
    # дах
    rr(-5.8, -3.3, 1.6, 3.3, sv[3], 0.9)
    rr(-5.2, -2.6, 1.0, 2.6, sv[4], 0.7)
    rr(-5.0, -2.9, 0.8, -2.5, sv[2], 0); rr(-5.0, 2.5, 0.8, 2.9, sv[2], 0)               # рейлінги-«лінії»
    # заднє скло, спойлер
    poly([(-8.8, -2.9), (-6.1, -3.2), (-6.1, 3.2), (-8.8, 2.9)], gl[1])
    rr(-6.4, -3.0, -5.9, 3.0, sv[1], 0)
    # бокові вікна з центральною стійкою
    for sg in (-1, 1):
        poly([(-5.6, sg * 3.55), (1.3, sg * 3.55), (0.9, sg * 4.15), (-5.3, sg * 4.15)], gl[1])
        rr(-2.2, sg * 3.55 - (0.3 if sg > 0 else 0.3), -1.6, sg * 4.15 + 0.0, sv[1], 0) if False else None
        poly([(-2.1, sg * 3.5), (-1.5, sg * 3.5), (-1.5, sg * 4.2), (-2.1, sg * 4.2)], sv[1])
        rr(1.8, sg * 4.45 - 0.5, 2.9, sg * 4.45 + 0.5, ink[2], 0.3)                         # дзеркала
    # ліхтарі (заходять на боки)
    tl = R['red'][3] if lit else R['red'][1]
    for sg in (-1, 1):
        poly([(-10.2, sg * 3.5), (-9.4, sg * 4.3), (-8.4, sg * 4.3), (-9.9, sg * 2.2)], tl)
    if lit:
        rr(-10.7, -3.9, -10.0, -2.0, R['red'][3], 0.2); rr(-10.7, 2.0, -10.0, 3.9, R['red'][3], 0.2)
    return im


CARTEX_W, CARTEX_H, CARTEX_LEN = 32, 16, 26     # текстура для графічного чипа: машина ~26 текселів завдовжки


def _bleed(a):
    """Колір прозорих текселів = колір найближчих непрозорих (щоб білінійне згладжування не давало темної облямівки)."""
    rgb = a[..., :3].astype(np.float32); al = a[..., 3] > 8
    out = rgb.copy(); filled = al.copy()
    for _ in range(4):
        acc = np.zeros_like(out); n = np.zeros(al.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                m = np.roll(np.roll(filled, dy, 0), dx, 1); c = np.roll(np.roll(out, dy, 0), dx, 1)
                acc += c * m[..., None]; n += m
        new = ~filled & (n > 0)
        out[new] = acc[new] / n[new][:, None]; filled |= new
    a = a.copy(); a[..., :3] = np.clip(out, 0, 255).astype(np.uint8)
    return a


def car_textures():
    """RGBA 8888 (байти R,G,B,A) для GU: звичайна, зі стоп-сигналами, м'яка тінь. Ніс на схід, центр машини — центр текстури."""
    from PIL import ImageFilter
    k = CARTEX_LEN / 21.3                      # пікс текстури на одиницю креслення draw_golf
    S = CAR_SZ * SS
    cw, ch = CARTEX_W / k * SS, CARTEX_H / k * SS
    box = (S / 2 - cw / 2, S / 2 - ch / 2, S / 2 + cw / 2, S / 2 + ch / 2)
    out = []
    for lit in (False, True):
        im = draw_golf(lit).transform((CARTEX_W * 4, CARTEX_H * 4), Image.EXTENT, box, resample=Image.BICUBIC)
        a = np.asarray(im.resize((CARTEX_W, CARTEX_H), Image.LANCZOS)).copy()
        out.append(_bleed(a).tobytes())
    sil = draw_golf(False).transform((CARTEX_W * 4, CARTEX_H * 4), Image.EXTENT, box, resample=Image.BICUBIC).split()[3]
    sil = sil.resize((CARTEX_W, CARTEX_H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.9))
    al = (np.asarray(sil, np.float32) * 0.42).astype(np.uint8)
    shd = np.zeros((CARTEX_H, CARTEX_W, 4), np.uint8); shd[..., 3] = al
    out.append(shd.tobytes())
    return out


TRAFFIC = [('sedan', (232, 232, 228)), ('sedan', (36, 38, 44)), ('hatch', (176, 40, 40)), ('sedan', (52, 86, 156)),
           ('hatch', (112, 116, 124)), ('van', (238, 238, 234)), ('taxi', (240, 196, 40)), ('hatch', (60, 120, 80))]


def draw_traffic(kind, body, lit=False):
    """Машина трафіку згори (ніс на схід), ті самі одиниці креслення, що й draw_golf: седан, хетчбек, бус, таксі."""
    S = CAR_SZ * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = S / 2
    def P(x, y): return (c + x * SS, c + y * SS)
    def rr(x0, y0, x1, y1, col_, r=0): d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=r * SS, fill=tuple(int(v) for v in col_))
    def poly(pts, col_): d.polygon([P(x, y) for x, y in pts], fill=tuple(int(v) for v in col_))
    b = np.array(body, np.float32)
    dark, lite = b * 0.62, np.minimum(255, b * 1.12 + 12)
    glass, glass2 = (40, 52, 64), (96, 120, 140)
    L = {'sedan': 10.4, 'hatch': 9.6, 'van': 11.0, 'taxi': 10.4}[kind]
    Wd = {'sedan': 4.5, 'hatch': 4.4, 'van': 4.8, 'taxi': 4.5}[kind]
    for x0 in (-L * 0.62, L * 0.38):
        rr(x0, -Wd - 0.3, x0 + 3.4, Wd + 0.3, (24, 24, 26), 0.9)
    rr(-L - 0.3, -Wd - 0.3, L + 0.3, Wd + 0.3, (20, 20, 24), 2.2)
    rr(-L, -Wd, L, Wd, b, 2.0)
    if kind == 'van':
        rr(-L + 0.6, -Wd + 0.6, L - 3.6, Wd - 0.6, lite, 0.8)                                  # дах буса
        poly([(L - 3.4, -Wd + 0.5), (L - 1.2, -Wd + 1.0), (L - 1.2, Wd - 1.0), (L - 3.4, Wd - 0.5)], glass)
    else:
        nose = L * (0.42 if kind != 'hatch' else 0.48)
        poly([(nose - 3.0, -Wd + 0.8), (nose, -Wd + 1.1), (nose, Wd - 1.1), (nose - 3.0, Wd - 0.8)], glass)     # лобове
        poly([(nose - 2.4, -Wd + 1.6), (nose - 1.6, -Wd + 1.7), (nose - 2.6, 0.8), (nose - 2.9, 0.4)], glass2)
        tail = -L * (0.62 if kind == 'sedan' or kind == 'taxi' else 0.78)
        rr(tail + 2.4, -Wd + 0.9, nose - 3.0, Wd - 0.9, lite, 0.8)                          # дах
        poly([(tail, -Wd + 1.0), (tail + 2.4, -Wd + 0.8), (tail + 2.4, Wd - 0.8), (tail, Wd - 1.0)], glass)   # заднє скло
        for sg in (-1, 1):
            poly([(tail + 2.4, sg * (Wd - 0.75)), (nose - 3.0, sg * (Wd - 0.75)), (nose - 3.2, sg * (Wd - 0.2)), (tail + 2.6, sg * (Wd - 0.2))], glass)
        rr(nose + 0.4, -Wd + 1.0, L - 0.4, Wd - 1.0, b * 1.04, 1.0)                          # капот
    if kind == 'taxi':
        rr(-1.6, -1.2, 0.6, 1.2, (250, 240, 200), 0.3); rr(-1.4, -0.9, 0.4, 0.9, (40, 40, 40), 0.2)
    for sg in (-1, 1):
        rr(L - 1.1, sg * (Wd - 1.4) - 0.6, L - 0.1, sg * (Wd - 1.4) + 0.6, (250, 240, 190), 0.3)   # фари
        rr(-L + 0.1, sg * (Wd - 1.3) - 0.6, -L + 1.0, sg * (Wd - 1.3) + 0.6, (255, 60, 50) if lit else (150, 30, 30), 0.3)
        rr(L * 0.18, sg * (Wd + 0.1) - 0.4, L * 0.18 + 1.0, sg * (Wd + 0.1) + 0.4, dark, 0.3)    # дзеркала
    return im


def traffic_textures():
    """RGBA для кожної моделі трафіку: звичайна + зі стоп-сигналами (той самий розмір текстури, що й у Golf)."""
    k = CARTEX_LEN / 21.3
    S = CAR_SZ * SS
    cw, ch = CARTEX_W / k * SS, CARTEX_H / k * SS
    box = (S / 2 - cw / 2, S / 2 - ch / 2, S / 2 + cw / 2, S / 2 + ch / 2)
    out = []
    for kind, col_ in TRAFFIC:
        for lit in (False, True):
            im = draw_traffic(kind, col_, lit).transform((CARTEX_W * 4, CARTEX_H * 4), Image.EXTENT, box, resample=Image.BICUBIC)
            out.append(_bleed(np.asarray(im.resize((CARTEX_W, CARTEX_H), Image.LANCZOS)).copy()).tobytes())
    return out


def car_frames(lit=False):
    base = draw_golf(lit)
    allowed = sum((R[k] for k in ('silver', 'glass', 'tire', 'ink', 'yellow', 'red', 'white')), [])
    S = CAR_SZ * SS
    return [quantise(base.rotate(-i * 360.0 / NROT, resample=Image.BICUBIC, center=(S / 2, S / 2)), allowed) for i in range(NROT)]


def icon(kind):
    """14x14 map icon: 'shop' (bag) or 'home' (house)"""
    S = 16 * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    def P(x, y): return (x * SS, y * SS)
    if kind == 'fuel':
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['ink'][0]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['green'][1]))
        d.rounded_rectangle([P(4.5, 4), P(9.5, 12.5)], radius=SS, fill=col(R['white'][2]))
        d.rectangle([P(5.5, 5), P(8.5, 7.5)], fill=col(R['green'][1]))
        d.line([P(9.5, 6), P(11.5, 7.5), P(11.5, 11)], fill=col(R['white'][2]), width=int(1.2 * SS))
    elif kind == 'sto':
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['ink'][0]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['red'][2]))
        d.line([P(5, 11), P(10.5, 5.5)], fill=col(R['white'][2]), width=int(2.2 * SS))
        d.ellipse([P(9, 3), P(13, 7)], fill=col(R['white'][2])); d.ellipse([P(10.4, 4.4), P(11.6, 5.6)], fill=col(R['red'][2]))
        d.ellipse([P(3.5, 9.5), P(6.5, 12.5)], fill=col(R['white'][2]))
    elif kind == 'shop':
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['green'][1]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['green'][2]))
        d.rounded_rectangle([P(4.5, 6.5), P(11.5, 12.5)], radius=SS, fill=col(R['white'][2]))
        d.arc([P(6, 3.5), P(10, 9)], 180, 360, fill=col(R['white'][2]), width=int(1.3 * SS))
    else:
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['yellow'][1]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['yellow'][2]))
        d.polygon([P(8, 3.5), P(12.8, 7.8), P(3.2, 7.8)], fill=col(R['red'][1]))
        d.rectangle([P(4.6, 7.6), P(11.4, 12.6)], fill=col(R['white'][2]))
        d.rectangle([P(7.0, 9.4), P(9.0, 12.6)], fill=col(R['box'][1]))
    allowed = sum((R[k] for k in ('green', 'yellow', 'white', 'red', 'box', 'ink')), [])
    return quantise(im, allowed)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'car':
    fr = car_frames()
    pal = np.array(PAL + [(255, 0, 255)] * (256 - len(PAL)), np.uint8); pal[0] = (255, 0, 255)
    sheet = np.hstack([fr[k * 4] for k in range(8)])
    Image.fromarray(pal[sheet]).resize((sheet.shape[1] * 8, sheet.shape[0] * 8), Image.NEAREST).save(sys.argv[2]); sys.exit()

if __name__ == '__main__':
    fr = scooter_frames()
    sheet = np.zeros((SZ, SZ * 8), np.uint8)
    for k in range(8):
        sheet[:, k * SZ:(k + 1) * SZ] = fr[k * 4]
    pal = np.array(PAL + [(255, 0, 255)] * (256 - len(PAL)), np.uint8)
    pal[0] = (255, 0, 255)
    Image.fromarray(pal[sheet]).resize((SZ * 8 * 6, SZ * 6), Image.NEAREST).save(sys.argv[1])
    ic = np.hstack([icon('shop'), icon('home')])
    Image.fromarray(pal[ic]).resize((ic.shape[1] * 8, ic.shape[0] * 8), Image.NEAREST).save(sys.argv[2])
