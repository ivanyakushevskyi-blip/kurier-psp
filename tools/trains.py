"""Поїзди згори (07.11.2026): текстури вагонів 256 x 32 (тексель 0,125 м: до 32 x 4 м), ніс на схід.
Вид строго згори — видно дахи: обладнання, вентилятори, пантографи, кабіни з лобовим склом; боки — смужкою кольору.
TRAIN_TEX: [(ім'я, довжина м)], textures() -> [RGBA bytes]; складів типи — у COMPOS."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TW, TH, TM_ = 256, 32, 0.125
R = 6                                                   # надвибірка
WID = 3.3                                               # ширина вагона, м


def _canvas():
    im = Image.new('RGBA', (TW * R, TH * R), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    k = R / TM_; cx, cy = im.size[0] / 2, im.size[1] / 2
    def P(x, y): return (cx + x * k, cy + y * k)
    def rr(x0, y0, x1, y1, c, r=0.15): d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=max(0, r * k), fill=tuple(int(v) for v in c))
    def ell(x0, y0, x1, y1, c): d.ellipse([P(x0, y0), P(x1, y1)], fill=tuple(int(v) for v in c))
    def ln(x0, y0, x1, y1, c, w=0.08): d.line([P(x0, y0), P(x1, y1)], fill=tuple(int(v) for v in c), width=max(1, int(w * k)))
    return im, d, P, rr, ell, ln


def body(L, side, roof, rr, w=WID):
    L2, W2 = L / 2, w / 2
    rr(-L2 - 0.05, -W2 - 0.05, L2 + 0.05, W2 + 0.05, (24, 24, 26), 0.3)
    rr(-L2, -W2, L2, W2, side, 0.25)                                    # бік (видно смужкою)
    rr(-L2 + 0.1, -W2 + 0.22, L2 - 0.1, W2 - 0.22, roof, 0.4)           # дах


def emu(head):
    """ЕР9: зелений бік, сірий дах; головний вагон з кабіною (жовтий лоб), моторний — з пантографом"""
    L = 20.0; im, d, P, rr, ell, ln = _canvas()
    side, roof = (46, 104, 70), (150, 154, 150)
    body(L, side, roof, rr)
    for x in np.arange(-L / 2 + 1.2, L / 2 - 1.0, 1.25):                # ребра даху
        rr(x, -1.2, x + 0.12, 1.2, (138, 142, 138), 0.03)
    for x in (-6.5, 0.0, 6.5): rr(x - 0.45, -0.45, x + 0.45, 0.45, (120, 124, 122), 0.2); ell(x - 0.3, -0.3, x + 0.3, 0.3, (90, 94, 92))   # вентилятори
    if head:
        rr(L / 2 - 1.1, -1.65, L / 2 - 0.02, 1.65, (240, 196, 40), 0.5)      # жовтий лоб
        rr(L / 2 - 0.75, -1.3, L / 2 - 0.25, 1.3, (40, 52, 64), 0.15)        # лобове скло
        for sg in (-1, 1): ell(L / 2 - 0.25, sg * 1.1 - 0.15, L / 2 + 0.05, sg * 1.1 + 0.15, (250, 240, 190))
    else:
        rr(-2.6, -0.9, 2.6, 0.9, (110, 112, 110), 0.1)                       # пантограф (рама + полоз)
        ln(-2.2, -0.75, 2.2, 0.75, (40, 40, 40), 0.1); ln(-2.2, 0.75, 2.2, -0.75, (40, 40, 40), 0.1)
        rr(-0.12, -1.3, 0.12, 1.3, (30, 30, 30), 0.03)
    return im, L


def loco_pass():
    """пасажирський електровоз ЧС: червоно-сірий, дві кабіни, пантографи, обладнання на даху"""
    L = 17.5; im, d, P, rr, ell, ln = _canvas()
    body(L, (176, 40, 36), (120, 124, 128), rr)
    for sg in (-1, 1):
        rr(sg * (L / 2) - (0.95 if sg > 0 else -0.05), -1.6, sg * (L / 2) + (0.05 if sg > 0 else 0.95), 1.6, (196, 50, 44), 0.5)
        rr(sg * (L / 2 - 0.6) - 0.25, -1.3, sg * (L / 2 - 0.6) + 0.25, 1.3, (40, 52, 64), 0.12)
    for x in (-4.2, 4.2):
        rr(x - 1.8, -0.8, x + 1.8, 0.8, (96, 98, 100), 0.1); ln(x - 1.5, -0.7, x + 1.5, 0.7, (34, 34, 34), 0.09); ln(x - 1.5, 0.7, x + 1.5, -0.7, (34, 34, 34), 0.09)
    rr(-2.0, -1.1, 2.0, 1.1, (150, 152, 156), 0.15)
    for x in np.arange(-1.7, 1.8, 0.45): rr(x, -1.0, x + 0.2, 1.0, (120, 122, 126), 0.02)
    ell(L / 2 - 0.15, -0.15, L / 2 + 0.1, 0.15, (250, 240, 190))
    return im, L


def coach():
    """пасажирський вагон УЗ: темно-синій бік, сірий дах з рядом дефлекторів, гармошки на торцях"""
    L = 24.5; im, d, P, rr, ell, ln = _canvas()
    body(L, (34, 52, 104), (132, 136, 140), rr)
    for x in np.arange(-L / 2 + 1.8, L / 2 - 1.5, 2.1): ell(x - 0.28, -0.28, x + 0.28, 0.28, (100, 104, 108)); ell(x - 0.16, -0.16, x + 0.16, 0.16, (70, 72, 76))
    for sg in (-1, 1): rr(sg * L / 2 - 0.35, -1.0, sg * L / 2 + 0.35, 1.0, (40, 40, 42), 0.1)
    for sg in (-1, 1): rr(-L / 2 + 0.4, sg * 1.42 - 0.06, L / 2 - 0.4, sg * 1.42 + 0.06, (190, 196, 204), 0)     # світла смуга
    return im, L


def loco_freight():
    """вантажний ВЛ (секція): синьо-сірий, одна кабіна (друга секція — розвернута)"""
    L = 16.5; im, d, P, rr, ell, ln = _canvas()
    body(L, (44, 86, 140), (118, 122, 128), rr)
    rr(L / 2 - 1.0, -1.6, L / 2 + 0.03, 1.6, (52, 100, 160), 0.5); rr(L / 2 - 0.7, -1.3, L / 2 - 0.2, 1.3, (40, 52, 64), 0.12)
    rr(-L / 2 + 1.0, -0.9, -L / 2 + 4.6, 0.9, (96, 98, 100), 0.1); ln(-L / 2 + 1.3, -0.8, -L / 2 + 4.3, 0.8, (34, 34, 34), 0.09); ln(-L / 2 + 1.3, 0.8, -L / 2 + 4.3, -0.8, (34, 34, 34), 0.09)
    for x in np.arange(-2.5, 5.5, 0.5): rr(x, -1.1, x + 0.22, 1.1, (104, 108, 112), 0.02)
    ell(L / 2 - 0.15, -0.15, L / 2 + 0.1, 0.15, (250, 240, 190))
    return im, L


def boxcar(col=(120, 62, 40)):
    L = 14.7; im, d, P, rr, ell, ln = _canvas()
    body(L, col, tuple(int(c * 0.85) for c in col), rr, 3.2)
    for x in np.arange(-L / 2 + 0.9, L / 2 - 0.5, 0.95): rr(x, -1.45, x + 0.14, 1.45, tuple(int(c * 0.7) for c in col), 0.02)
    rr(-0.5, -1.0, 0.5, 1.0, tuple(int(c * 0.75) for c in col), 0.1)
    return im, L


def tank():
    L = 12.0; im, d, P, rr, ell, ln = _canvas()
    rr(-L / 2 - 0.05, -1.55, L / 2 + 0.05, 1.55, (20, 20, 22), 0.2)
    for k in range(10):                                                 # циліндр: світло зверху-ліворуч
        t = k / 9; w = 1.45 * (1 - t * 0.98)
        c = int(36 + 60 * (1 - abs(t - 0.25) * 1.3) ** 2)
        rr(-L / 2 + 0.3 * t, -w - (0.25 * t), L / 2 - 0.3 * t, w - (0.25 * t), (c, c, c + 4), 0.6)
    ell(-0.6, -0.6, 0.6, 0.6, (60, 60, 64)); ell(-0.35, -0.35, 0.35, 0.35, (90, 90, 94))
    rr(-L / 2 + 0.6, -0.08, L / 2 - 0.6, 0.08, (70, 70, 74), 0)
    return im, L


def hopper():
    L = 14.7; im, d, P, rr, ell, ln = _canvas()
    body(L, (110, 112, 108), (130, 132, 126), rr, 3.2)
    for x in np.arange(-L / 2 + 1.0, L / 2 - 0.8, 2.6): rr(x, -1.2, x + 2.0, 1.2, (150, 150, 144), 0.15); rr(x + 0.3, -0.25, x + 1.7, 0.25, (100, 100, 96), 0.05)
    return im, L


def gondola():
    L = 13.9; im, d, P, rr, ell, ln = _canvas()
    body(L, (80, 72, 64), (36, 34, 32), rr, 3.2)
    rs = np.random.RandomState(4)
    for _ in range(260):                                                # вугілля
        x, y, r = rs.uniform(-L / 2 + 0.4, L / 2 - 0.4), rs.uniform(-1.25, 1.25), rs.uniform(0.08, 0.22)
        c = int(rs.uniform(20, 62)); ell(x - r, y - r, x + r, y + r, (c, c, c + 3))
    return im, L


def flat_cont():
    L = 18.4; im, d, P, rr, ell, ln = _canvas()
    body(L, (60, 62, 66), (70, 72, 76), rr, 2.9)
    for x0, col in ((-L / 2 + 0.3, (180, 60, 40)), (0.15, (40, 110, 150))):
        rr(x0, -1.25, x0 + 8.8, 1.25, col, 0.05)
        for x in np.arange(x0 + 0.3, x0 + 8.6, 0.32): rr(x, -1.2, x + 0.1, 1.2, tuple(int(c * 0.8) for c in col), 0)
    return im, L


TYPES = [('emu_head', lambda: emu(True)), ('emu_mid', lambda: emu(False)), ('loco_pass', loco_pass), ('coach', coach),
         ('loco_frt', loco_freight), ('boxcar', boxcar), ('tank', tank), ('hopper', hopper), ('gondola', gondola), ('flat', flat_cont),
         ('boxcar2', lambda: boxcar((92, 96, 60)))]
NAMES = [t[0] for t in TYPES]


def textures():
    out, lens = [], []
    for nm, fn in TYPES:
        im, L = fn()
        a = np.asarray(im.resize((TW, TH), Image.LANCZOS)).copy()
        out.append(a.tobytes()); lens.append(L)
    return out, lens


if __name__ == '__main__':
    import sys
    ims = [fn()[0].resize((TW * 2, TH * 2), Image.LANCZOS) for _, fn in TYPES]
    W = Image.new('RGBA', (TW * 2, TH * 2 * len(ims)), (70, 70, 74, 255))
    for i, im in enumerate(ims): W.alpha_composite(im, (0, i * TH * 2))
    W.save(sys.argv[1])
