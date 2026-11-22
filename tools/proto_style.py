"""Прототип стилю (22.11.2026): «місто — головний герой». Ті самі дані й палітра, інший вигляд.

  A — згори, піксель-арт: пласкі кольори без шуму, контури, плитка, дахи з 3 тонами й обведенням, дерева-кулі з контуром;
  B — 3/4 з фасадами: кожен будинок «піднято» на висоту (стіни з вікнами за поверхами, вітрини й вивіски закладів),
      дах зсунуто вгору; порядок малювання — за нижнім краєм.
Запуск: python3 tools/proto_style.py X Y [назва]  (центр, пікселі карти 1×) -> build/proto2/<назва>_{A,B}.png (480×272, як екран PSP)
"""
import os, sys, inspect, json, random
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import citydraw as CD

cx, cy = int(sys.argv[1]), int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else 'p'
SW, SH = 480, 272
PADT = 140                                                             # зверху більше: високі будинки знизу кадру «ростуть» угору
W, H, M = SW, SH + PADT, 96
x0, y0 = cx - SW // 2, cy - SH // 2 - PADT
OUT = 'build/proto2'; os.makedirs(OUT, exist_ok=True)
MPP = CD.MPP

# ---------- рендер тим самим кодом карти з перехопленням землі, будинків і класів
src = inspect.getsource(CD.render)
old = "        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)\n        cls = clsb.copy()"
assert old in src
src = src.replace(old, "        _CAP['gnd'] = cv2.resize(img_ground, (W, H), interpolation=cv2.INTER_AREA); _CAP['blds'] = blds\n" + old)
src = src.replace("def render(", "def render_cap(", 1)
CD.__dict__['_CAP'] = {}
exec(compile(src, CD.__file__, 'exec'), CD.__dict__)
tl, so, pr = [], [], []
img, cls = CD.render_cap(x0 - M, y0 - M, W + 2 * M, H + 2 * M, tree_out=tl, ss=4, solid_out=so, props_out=pr)
C = CD._CAP
C['blds'] = [(i, b, [(p[0] - M, p[1] - M) for p in pts], ptf) for i, b, pts, ptf in C['blds']]   # у координати обрізаного кадру
sl = (slice(M, M + H), slice(M, M + W))
img, cls, gnd = img[sl].copy(), cls[sl].copy(), C['gnd'][sl].copy()
cv2.imwrite(os.path.join(OUT, 'raw.png'), cv2.cvtColor(img[PADT:PADT + SH], cv2.COLOR_RGB2BGR)); cv2.imwrite(os.path.join(OUT, 'rawg.png'), cv2.cvtColor(gnd[PADT:PADT + SH], cv2.COLOR_RGB2BGR))
ox, oy = x0 - M, y0 - M                                                # світ -> локальні: (x - x0, y - y0)
md = json.load(open('build/mapdata.json', encoding='utf-8'))
shops = [(s['x'] - x0, s['y'] - y0, s['name']) for s in md['shops'] if -40 < s['x'] - x0 < W + 40 and -40 < s['y'] - y0 < H + 40]
G = dict(zip('YARD GRASS WATER SIDE ROAD PATH PED RAIL DIRT PITCH TRACK PARK PAVED WOOD SAND'.split(), range(1, 16)))

def dark(c, k): return tuple(int(max(0, min(255, v * k))) for v in c)

# ---------- земля: пласкі кольори, контури бордюрів, плитка
def stylize_ground(g):
    f = cv2.medianBlur(g, 5)
    f = cv2.bilateralFilter(f, 7, 30, 7)
    out = f.astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]; gx, gy = xx + x0, yy + y0
    paved = np.isin(cls, [G['SIDE'], G['PED'], G['PAVED'], G['PATH']])
    row = gy // 6; seam = ((gy % 6) == 0) | (((gx + 3 * (row % 2)) % 6) == 0)
    out[paved & seam] *= 0.9
    grass = np.isin(cls, [G['GRASS'], G['PARK'], G['YARD'], G['PITCH']])
    # трава й двори: два-три пласкі тони замість плям зі знімка (відтінок району лишається — середній колір околиці)
    if grass.any():
        avg = cv2.blur(np.where(grass[..., None], out, 0), (61, 61)) / np.maximum(cv2.blur(grass.astype(np.float32), (61, 61))[..., None], 1e-3)
        blob = cv2.GaussianBlur(((np.sin(gx * 0.11) + np.sin(gy * 0.13 + gx * 0.05) + np.sin((gx + gy) * 0.07)) / 3).astype(np.float32), (0, 0), 2)
        k = np.where(blob > 0.35, 1.08, np.where(blob < -0.35, 0.92, 1.0))[..., None]
        out = np.where(grass[..., None], avg * k, out)
    tuft = ((np.sin(gx * 12.9898 + gy * 78.233) * 43758.5453) % 1.0) > 0.93
    out[grass & tuft] *= 0.86
    road = cls == G['ROAD']
    # контур між поверхнями (бордюр, край газону) — темна лінія 1 пікс
    edge = np.zeros((H, W), bool)
    for a, b in ((road, paved), (grass, paved), (road, grass)):
        da = cv2.dilate(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge |= da & b
    out[edge] *= 0.72
    # розмітка лишається з оригіналу (контрастні пікселі на дорозі)
    lum = g.astype(np.float32).mean(2); loc = cv2.blur(lum, (7, 7))
    mk = road & ((lum > loc + 30) | ((g[..., 0] > 150) & (g[..., 2] < 110)))
    out[mk] = g[mk]
    return np.clip(out, 0, 255).astype(np.uint8)

def roof_colors(i, b, pts):
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
    m = m.astype(bool)
    if m.sum() < 4: return None, m
    return np.median(img[m].reshape(-1, 3), 0), m

def draw_roof_flat(canvas, b, pts, base, dy=0):
    """дах: 3 тони з оригінального світлотіньового малюнка + обведення"""
    P = np.round(np.array(pts) - [0, dy]).astype(np.int32)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [P], 1); m = m.astype(bool)
    if not m.any(): return
    Pn = np.round(np.array(pts)).astype(np.int32)
    m0 = np.zeros((H, W), np.uint8); cv2.fillPoly(m0, [Pn], 1); m0 = m0.astype(bool)
    lum = img.astype(np.float32).mean(2)
    l0 = lum[m0]; lo, hi = np.percentile(l0, 30), np.percentile(l0, 70)
    tone = np.zeros((H, W), np.float32)
    sh = np.roll(lum, -int(dy), axis=0) if dy else lum
    t = np.where(sh < lo, 0.84, np.where(sh > hi, 1.1, 1.0))
    col = np.array(base, np.float32)
    canvas[m] = np.clip(col * t[m][:, None], 0, 255).astype(np.uint8)
    ed = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    canvas[ed] = dark(base, 0.5)

def draw_tree(canvas, x, y, r, kind, lift=0):
    r = max(4, int(r * 1.05)); y = int(y - lift)
    rs = random.Random(int(x * 31 + y))
    base = [(70, 124, 52), (58, 108, 60), (88, 130, 48), (50, 96, 70)][rs.randrange(4)]
    if kind and 'autumn' in str(kind): base = (196, 120, 46)
    yy, xx = np.ogrid[-r - 2:r + 3, -r - 2:r + 3]
    d = np.sqrt(xx ** 2 + (yy * 1.08) ** 2)
    x0_, y0_ = int(x) - r - 2, int(y) - r - 2
    # тінь
    sy0, sx0 = y0_ + 3, x0_ + 3
    for (yy_, xx_), v in np.ndenumerate(d <= r):
        if not v: continue
        Y, X = sy0 + yy_, sx0 + xx_
        if 0 <= Y < H and 0 <= X < W: canvas[Y, X] = (canvas[Y, X] * 0.72).astype(np.uint8)
    lit = (-(xx) - (yy)) / (r * 1.4)
    for (yy_, xx_), dv in np.ndenumerate(d):
        if dv > r + 0.5: continue
        Y, X = y0_ + yy_, x0_ + xx_
        if not (0 <= Y < H and 0 <= X < W): continue
        if dv > r - 0.8: canvas[Y, X] = dark(base, 0.45); continue
        L = lit[yy_, xx_] + (rs.random() - 0.5) * 0.25
        canvas[Y, X] = dark(base, 1.22 if L > 0.35 else (0.8 if L < -0.25 else 1.0))

# ---------- A: згори
A = stylize_ground(gnd)
for i, b, pts, ptf in C['blds']:
    base, m = roof_colors(i, b, pts)
    if base is None: continue
    sh = np.round(np.array(pts) + [3, 4]).astype(np.int32)              # тінь від даху
    ms = np.zeros((H, W), np.uint8); cv2.fillPoly(ms, [sh], 1)
    A[ms.astype(bool)] = (A[ms.astype(bool)] * 0.72).astype(np.uint8)
for i, b, pts, ptf in sorted(C['blds'], key=lambda z: z[1]['h']):
    base, m = roof_colors(i, b, pts)
    if base is None: continue
    draw_roof_flat(A, b, pts, base)
for x, y, r, k in sorted(tl, key=lambda t: t[1]):
    draw_tree(A, x - x0, y - y0, r, k)

# ---------- B: 3/4 з фасадами
# Стіна малюється як текстура у «власних» координатах (u — уздовж стіни, v — висота), а на екран переноситься
# стовпчиками: кожен стовпчик зсувається вертикально за нахилом стіни. Тому вікна — паралелограми з вертикальними
# боками, рівно в площині стіни (як у піксель-арті 3/4). Стіни, що дивляться вбік (майже вертикальні на екрані), — вузька
# смужка без вікон, бо в цьому виді їх майже не видно.
KV = 0.62
FLOOR = int(round(3.0 / MPP * KV))                                     # 7 пікс на поверх
def jit(c, rs, a=8): return tuple(int(np.clip(v + rs.randint(-a, a), 0, 255)) for v in c)

def btype(b, floors, area_m2):
    bt = b.get('t', {}).get('building', 'yes')
    if bt in ('garage', 'garages', 'shed', 'kiosk', 'service') or area_m2 < 25: return 'shed'
    if bt in ('house', 'detached', 'semidetached_house', 'terrace') or (floors <= 2 and area_m2 < 220): return 'house'
    if floors >= 9: return 'panel'
    if floors >= 4: return 'brick'
    return 'old'

PAL = {
    'panel': [(208, 210, 204), (198, 204, 206), (214, 210, 198)],
    'brick': [(186, 116, 92), (210, 190, 152), (196, 178, 150), (178, 122, 98)],
    'old':   [(226, 196, 170), (232, 214, 160), (196, 214, 196), (226, 204, 196), (214, 206, 186)],
    'house': [(236, 226, 200), (230, 214, 178), (214, 224, 206), (240, 236, 226), (222, 204, 184)],
    'shed':  [(150, 156, 160), (168, 160, 146), (132, 140, 146)],
}
GLASS, GLASS_HI, LIT = (62, 88, 118), (146, 178, 202), (236, 210, 132)

def wall_tex(tp, w, floors, Lm, rs, wall, shop, entr):
    """текстура стіни w×hp пікс (без перспективи); w — ширина на екрані, Lm — справжня довжина, м"""
    hp = floors * FLOOR
    T = np.zeros((hp, w, 3), np.uint8); T[:] = wall
    dk, lt = dark(wall, 0.82), dark(wall, 1.08)
    yy = np.arange(hp)[:, None]
    if tp == 'brick':                                                  # цегла: рядки по 2 пікс зі зсувом
        xx = np.arange(w)[None, :]
        mortar = ((yy % 2) == 0) & (((xx + 2 * ((yy // 2) % 2)) % 4) == 0)
        T[np.broadcast_to(mortar, (hp, w))] = dk
    if tp == 'shed':                                                   # профнастил: вертикальні смуги
        T[:, ::2] = dk
        if w >= 8: T[hp - 5:hp, w // 2 - 3:w // 2 + 3] = dark(wall, 0.6); T[hp - 5:hp, w // 2 - 3:w // 2 + 3:2] = dark(wall, 0.5)
        return T
    # колонки вікон: за справжньою довжиною, але не густіше ніж 5 пікс на екрані
    ncol = max(1, min(int(round(Lm / 3.2)), w // 5))
    if w < 6: ncol = 0
    cw = w / max(ncol, 1); ww = int(np.clip(round(cw * 0.5), 2, 5))
    for f in range(floors):
        yb = hp - f * FLOOR                                            # низ поверху
        if tp == 'panel':
            T[yb - 1, :] = dk                                          # шов плит
        for j in range(ncol):
            xc = int((j + 0.5) * cw); x0_ = xc - ww // 2
            ground = f == 0
            if ground and shop:
                continue
            if ground and entr and j in entr:                          # під'їзд: двері + козирок
                d0 = max(0, xc - 2)
                T[yb - 5:yb, d0:d0 + 4] = (92, 66, 50); T[yb - 5:yb, d0 + 1:d0 + 3] = (120, 88, 64)
                T[yb - 6, max(0, d0 - 1):d0 + 5] = dark(wall, 0.55)
                continue
            wy0 = yb - FLOOR + 2; wy1 = yb - 2                         # 3–4 пікс заввишки
            lit = rs.random() < 0.12
            T[wy0 - 1:wy1 + 1, x0_ - 1:x0_ + ww + 1] = (236, 234, 226) if tp in ('old', 'house') else dark(wall, 0.72)
            T[wy0:wy1, x0_:x0_ + ww] = LIT if lit else GLASS
            if not lit: T[wy0, x0_] = GLASS_HI
            T[wy1, x0_ - 1:x0_ + ww + 1] = dark(wall, 1.15)            # підвіконня
            if tp == 'panel' and j % 2 == 1 and f > 0:                  # балкон
                T[yb - 2, x0_ - 2:x0_ + ww + 2] = dark(wall, 1.12); T[yb - 1, x0_ - 2:x0_ + ww + 2] = (88, 92, 96)
    if shop:                                                           # перший поверх — вітрини й вивіска
        yb = hp
        T[yb - FLOOR + 1:yb, 1:w - 1] = GLASS
        for q in range(2, w - 2, 7): T[yb - FLOOR + 1:yb, q] = (200, 204, 206)
        for q in range(0, w - 3, 9): T[yb - FLOOR + 2, q + 1:q + 3] = GLASS_HI
        sc = shop
        T[yb - FLOOR - 2:yb - FLOOR + 1, 0:w] = sc
        for q in range(2, w - 2, 2):
            if rs.random() < 0.75: T[yb - FLOOR - 1, q] = (250, 246, 236)
    if tp in ('old', 'brick') and hp > 3: T[0, :] = lt; T[1, :] = dark(wall, 0.7)   # карниз
    T[hp - 1, :] = dark(wall, 0.55)                                    # цоколь
    return T

def blit_wall(canvas, a, c, T, shade):
    """перенести текстуру на стіну a->c: стовпчики з вертикальним зсувом"""
    hp, w = T.shape[:2]
    dx, dy = c[0] - a[0], c[1] - a[1]
    xs = range(int(round(min(a[0], c[0]))), int(round(max(a[0], c[0]))))
    for X in xs:
        if not (0 <= X < W): continue
        t = (X + 0.5 - a[0]) / dx
        if not (0 <= t <= 1): continue
        yb = int(round(a[1] + dy * t))
        u = min(w - 1, max(0, int(t * w)))
        col = (T[:, u].astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
        y0_ = yb - hp
        s0, s1 = max(0, y0_), min(H, yb)
        if s1 > s0: canvas[s0:s1, X] = col[s0 - y0_:s1 - y0_]

# ---------- піксельний шрифт 3×5 для вивісок (лише потрібні великі літери)
F35 = {
 'А': ['010', '101', '111', '101', '101'], 'В': ['110', '101', '110', '101', '110'], 'Е': ['111', '100', '110', '100', '111'],
 'І': ['111', '010', '010', '010', '111'], 'К': ['101', '110', '100', '110', '101'], 'Л': ['010', '101', '101', '101', '101'],
 'М': ['101', '111', '111', '101', '101'], 'Н': ['101', '101', '111', '101', '101'], 'О': ['010', '101', '101', '101', '010'],
 'П': ['111', '101', '101', '101', '101'], 'Р': ['110', '101', '110', '100', '100'], 'Т': ['111', '010', '010', '010', '010'],
 'Ц': ['101', '101', '101', '111', '001'], 'Ч': ['101', '101', '011', '001', '001'], 'Ш': ['101', '101', '111', '111', '111'],
 '.': ['000', '000', '000', '000', '010'], ' ': ['000', '000', '000', '000', '000'], 'Г': ['111', '100', '100', '100', '100'],
}
def text_w(t, sc=1): return (len(t) * 4 - 1) * sc
def draw_text(T, x, y, t, col, sc=1):
    for ch in t:
        g = F35.get(ch, F35[' '])
        for r in range(5):
            for c_ in range(3):
                if g[r][c_] == '1':
                    yy_, xx_ = y + r * sc, x + c_ * sc
                    if 0 <= yy_ and yy_ + sc <= T.shape[0] and 0 <= xx_ and xx_ + sc <= T.shape[1]: T[yy_:yy_ + sc, xx_:xx_ + sc] = col
        x += 4 * sc

# ---------- пам'ятки: власні фасади (стилізація; точність — за фото)
LMK = [(12040, 14430, 'tower'), (11979, 14409, 'depot'), (13149, 15918, 'cinema'), (13664, 12133, 'mall')]
def lm_tex(kind, w, hp, main, rs):
    if kind == 'cinema':                                               # модернізм: скляний вестибюль, глуха стіна з літерами, рама
        wall = (222, 218, 206)
        T = np.zeros((hp, w, 3), np.uint8); T[:] = wall
        if not main:
            for q in range(3, w, 8): T[:, q] = dark(wall, 0.9)
            T[hp - 1] = dark(wall, 0.55); return T
        g0 = hp - FLOOR * 2 + 2                                        # два поверхи скла
        T[g0:hp - 1, 2:w - 2] = GLASS
        for q in range(2, w - 2, 5): T[g0:hp - 1, q] = (214, 216, 214)   # імпости
        T[g0 + FLOOR - 1, 2:w - 2] = (214, 216, 214)
        for q in range(4, w - 4, 11): T[g0 + 1, q:q + 2] = GLASS_HI; T[g0 + 2, q + 1] = GLASS_HI
        T[g0 - 2:g0, :] = dark(wall, 0.6)                               # козирок над входом
        t = 'КІНОТЕАТР ІМ. ШЕВЧЕНКА'; sc = 2 if text_w(t, 2) < w - 6 else 1
        if text_w(t, sc) > w - 4: t = 'КІНОТЕАТР'; sc = 2 if text_w(t, 2) < w - 6 else 1
        ty = max(2, (g0 - 2 - 5 * sc) // 2)
        draw_text(T, (w - text_w(t, sc)) // 2, ty, t, (178, 40, 40), sc)
        for q in range(0, w, 3): T[0, q] = dark(wall, 0.8)
        T[hp - 1] = dark(wall, 0.5); return T
    if kind == 'mall':                                                 # гіпермаркет: сендвіч-панелі, помаранчевий пояс, вивіска, вхідний портал
        wall = (226, 228, 226)
        T = np.zeros((hp, w, 3), np.uint8); T[:] = wall
        for y in range(4, hp, 4): T[y] = dark(wall, 0.92)
        T[0:5] = (232, 104, 34); T[5] = dark((232, 104, 34), 0.7)
        if main:
            t = 'ЕПІЦЕНТР'; sc = 3 if (text_w(t, 3) < w * 0.5 and 8 + 5 * 3 + 9 < hp - FLOOR * 2) else (2 if 8 + 19 < hp - FLOOR * 2 else 1)
            bw, bh = text_w(t, sc) + 10 + 5 * sc + 6, 5 * sc + 8
            bx, by = (w - bw) // 2, 8
            T[by:by + bh, bx:bx + bw] = (232, 104, 34); T[by + bh, bx:bx + bw] = dark((232, 104, 34), 0.6)
            draw_text(T, bx + 5, by + 4, t, (255, 255, 255), sc)
            kx = bx + 5 + text_w(t, sc) + 5                             # «К» у квадраті
            T[by + 3:by + 5 + 5 * sc, kx:kx + 5 * sc] = (255, 214, 40)
            draw_text(T, kx + sc, by + 4, 'К', (232, 104, 34), sc)
            px_ = w // 2; T[hp - FLOOR * 2 + 1:hp - 1, px_ - 10:px_ + 10] = GLASS   # вхід
            for q in range(px_ - 10, px_ + 10, 4): T[hp - FLOOR * 2 + 1:hp - 1, q] = (200, 204, 206)
            T[hp - FLOOR * 2 - 1:hp - FLOOR * 2 + 1, px_ - 13:px_ + 13] = (232, 104, 34)
            for q in range(6, w - 6, 26):
                if abs(q - px_) > 20: T[hp - 8:hp - 1, q:q + 7] = (150, 154, 156); T[hp - 8:hp - 1, q:q + 7:2] = (132, 136, 138)   # ворота складу
        T[hp - 1] = dark(wall, 0.5); return T
    if kind == 'depot':
        # За фото (Подільська, 39): вохристий фасад, білі руставані пілястри між прольотами, балюстрада з тумбами
        # на даху, фриз з червоними літерами «КІНОТЕАТР ПЛАНЕТА», 2-й поверх — пари вікон з напівциркульними
        # вікнами-люнетами над ними, 1-й — темно-червоні двері по краях і афіші в червоних рамах, темний цоколь.
        OCH, WH, RF = (230, 196, 118), (244, 240, 230), (140, 58, 48)
        T = np.zeros((hp, w, 3), np.uint8); T[:] = OCH
        if w < 14:                                                       # вузька бічна смужка: лише балюстрада, руст, цоколь
            T[0:4] = RF; T[0, :] = WH; T[3, :] = WH; T[4, :] = WH; T[5:hp - 2:2, :] = dark(OCH, 0.9); T[hp - 2:hp] = (88, 84, 80)
            return T
        # ряди згори вниз (hp = 4 поверхи «ігрових» = 28 пікс)
        bal0, bal1 = 0, 4                                              # балюстрада на тлі даху
        fr0, fr1 = bal1 + 1, bal1 + 8                                    # фриз з написом
        f2_0, f2_1 = fr1 + 1, fr1 + 10                                   # 2-й поверх
        f1_0 = f2_1 + 1                                                  # 1-й поверх до цоколя
        T[bal0:bal1] = RF
        T[bal1 - 4, :] = WH; T[bal1 - 1, :] = WH
        T[bal1 - 3:bal1 - 1, ::2] = WH                                   # балясини
        T[bal1, :] = WH; T[bal1 + 1, :] = dark(OCH, 0.75)                # карниз з тінню
        nb = max(2, int(round(w / 13)))                                  # прольоти між пілястрами
        piers = [int(round(j * (w - 3) / nb)) for j in range(nb + 1)]
        for j in range(nb):
            a_, b_ = piers[j] + 3, piers[j + 1]
            cx_ = (a_ + b_) // 2
            for xw in (cx_ - 3, cx_ + 1):                               # пара вікон 2-го поверху + люнети
                if a_ <= xw and xw + 2 < b_:
                    T[f2_0 + 4:f2_1, xw:xw + 3] = (54, 64, 70); T[f2_0 + 4:f2_1, xw + 1] = (70, 84, 92)
                    T[f2_1, xw - 1:xw + 4] = WH                          # підвіконня
                    T[f2_0 + 1:f2_0 + 3, xw:xw + 3] = (54, 64, 70); T[f2_0, xw + 1] = (54, 64, 70)   # люнет
                    T[f2_0, xw] = WH; T[f2_0, xw + 2] = WH
            T[f2_1 + 1, a_:b_] = dark(OCH, 0.85)                          # міжповерховий пояс
            if main:                                                     # 1-й поверх: двері по краях, афіші посередині
                if j in (0, nb - 1):
                    T[hp - 9:hp - 2, cx_ - 2:cx_ + 2] = (124, 40, 34); T[hp - 9:hp - 2, cx_] = (104, 32, 28)
                elif abs(j - (nb - 1) / 2) > 1.01:                      # звичайні вікна 1-го поверху
                    for xw in (cx_ - 3, cx_ + 1):
                        T[hp - 9:hp - 4, xw:xw + 3] = (54, 64, 70); T[hp - 4, xw - 1:xw + 4] = WH
                else:
                    T[hp - 10:hp - 3, cx_ - 4:cx_ + 4] = (150, 36, 36)
                    T[hp - 9:hp - 4, cx_ - 3:cx_ + 3] = [(60, 90, 140), (40, 46, 60), (120, 70, 50)][j % 3]
                    T[hp - 9, cx_ - 3:cx_ + 3] = (232, 226, 214)
        for px_ in piers:                                                # руставані пілястри + тумби над ними
            T[fr0:hp - 2, px_:px_ + 3] = WH
            T[fr0:hp - 2:2, px_:px_ + 3] = dark(WH, 0.86)
            T[bal0:bal1, px_:px_ + 3] = WH; T[bal0, px_:px_ + 3] = dark(WH, 0.8)
        T[hp - 2:hp] = (88, 84, 80)                                      # цоколь
        if main:
            for t, sc in (('КІНОТЕАТР ПЛАНЕТА', 1), ('ПЛАНЕТА', 1)):
                if text_w(t, sc) < w - 8:
                    T[fr0:fr1 - 1, 3:w - 3] = OCH
                    draw_text(T, (w - text_w(t, sc)) // 2, fr0 + 1, t, (150, 42, 36), sc); break
        return T
    return None

def zpx(m): return m / MPP * KV                                        # висота (м) -> зсув угору на екрані

def tower_tex(w, hp, z0m, z_top, main):
    """За фото: вохристий стовбур, білий руст на кутах унизу, одиночні вікна по осі в білих обрамленнях, балкон,
    вікно з фронтончиком, білий карниз-пояс; годинник — жовта «сонце»-зірка з синім циферблатом під карнизом;
    вгорі ярус-альтанка з арками в білих рамках і сильно винесений карниз; низ — вхід у крамницю."""
    OCH, WH = (230, 196, 118), (244, 240, 230)
    T = np.zeros((hp, w, 3), np.uint8); T[:] = OCH
    def ry(m): return int(round(zpx(z_top - m)))
    q0 = ry(15.0)                                                        # руст на кутах до 12 м
    for r in range(q0, hp - 2):
        if (r // 2) % 2 == 0: T[r, 0:3] = WH; T[r, w - 3:w] = WH
        else: T[r, 0:2] = WH; T[r, w - 2:w] = WH
    c = w // 2
    if w < 10:                                                           # вузька бічна грань: пояси й карнизи без деталей
        for m in (24.5, 25.5, 30.5): r = ry(m); T[max(0, r - 1):r + 1] = WH
        T[hp - 2:hp] = (88, 84, 80); return T
    for m, kind_ in ((5.0, 'win'), (10.0, 'balc'), (14.5, 'ped'), (18.0, 'win')):   # вікна по осі
        r = ry(m)
        if r < 6 or r >= hp - 2: continue
        T[r - 5:r, c - 2:c + 2] = WH; T[r - 4:r, c - 1:c + 1] = (54, 64, 70)
        if kind_ == 'balc': T[r, c - 4:c + 4] = (60, 50, 46); T[r - 2, c - 4:c + 4] = (60, 50, 46); T[r - 2:r, c - 4] = (60, 50, 46); T[r - 2:r, c + 3] = (60, 50, 46)
        if kind_ == 'ped': T[r - 6, c - 3:c + 3] = WH; T[r - 7, c - 1:c + 1] = WH
    if main:                                                             # вхід у крамницю внизу
        T[hp - 7:hp - 2, c - 3:c + 3] = (210, 216, 220); T[hp - 6:hp - 2, c - 2:c + 2] = (90, 120, 150)
        T[hp - 8, c - 4:c + 4] = (200, 40, 40)
    rc = ry(24.5); T[rc - 1:rc + 1] = WH; T[rc + 1] = dark(OCH, 0.75)    # карниз-пояс під годинником
    if main and w >= 12:                                                 # годинник-«сонце»
        cy_ = ry(22.0); r_ = min(w // 2 - 3, 5)
        yy, xx = np.ogrid[:hp, :w]; dx_, dy_ = xx - (w - 1) / 2, yy - cy_
        d = np.sqrt(dx_ ** 2 + dy_ ** 2); a = np.arctan2(dy_, dx_)
        ray = d <= r_ + 2 + (np.cos(a * 12) > 0.3)
        T[ray] = (238, 190, 40); T[d <= r_ + 1] = (200, 150, 30); T[d <= r_] = (60, 110, 190)
        T[d <= r_ - 1] = (226, 232, 240)
        T[cy_ - r_ + 2:cy_ + 1, w // 2] = (30, 40, 80); T[cy_, w // 2:w // 2 + r_ - 2] = (30, 40, 80)
    a0, a1 = ry(30.5), ry(25.5)                                          # ярус-альтанка: арки в білих рамках
    na = 3 if w >= 16 else 2; pw = w / na
    for j in range(na):
        x0_ = int(j * pw) + 2; x1_ = int((j + 1) * pw) - 1
        T[a0 + 1:a1, x0_ - 1:x1_ + 1] = WH
        T[a0 + 3:a1 - 1, x0_:x1_] = (60, 64, 70); T[a0 + 2, x0_ + 1:x1_ - 1] = (60, 64, 70)
        T[a1 - 3, x0_:x1_] = WH                                          # перемичка-підвіконня
    T[a0 - 1:a0 + 1] = WH; T[a0 - 2] = dark(WH, 0.8)                      # винесений карниз
    T[a1:a1 + 1] = WH
    T[hp - 2:hp] = (88, 84, 80)                                          # цоколь
    return T

def draw_tower(canvas, x, y, z0m=0.0, ang=0.0):
    """Стара пожежна вежа — об'єм у проєкції 3/4, ПОВЕРНУТИЙ ЯК БУДИНОК, на якому стоїть (ang — напрям його стін):
    квадрат 6,5 м на даху депо (z0m), стовбур до 24 м, шатро до 31 м. Видимі стіни (нормаль до камери) —
    текстурою в площині стіни; шатро — чотири грані піраміди від дальньої до ближньої."""
    hw = 6.0 / MPP / 2
    z_top, z_roof = 31.0, 33.0                                           # стовбур+альтанка до 31 м, парапет до 33 (на фото вежа ~2,5× вища за кінотеатр)
    hp = int(round(zpx(z_top - z0m)))
    c, s_ = np.cos(ang), np.sin(ang)
    loc = [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]
    G_ = [np.array([x + u * c - v * s_, y + u * s_ + v * c]) for u, v in loc]   # кути на землі (за годинниковою на екрані)
    lift = zpx(z0m)
    walls = []
    for k in range(4):
        a, b = G_[k], G_[(k + 1) % 4]
        dx, dy = b[0] - a[0], b[1] - a[1]
        nx, ny = dy, -dx                                                   # зовнішня нормаль для обходу за годинниковою (y вниз)
        if ny <= 0: continue
        walls.append((a, b, dx, dy, nx, ny))
    main = max(range(len(walls)), key=lambda q: walls[q][5]) if walls else -1
    for q, (a, b, dx, dy, nx, ny) in enumerate(walls):
        w = int(round(abs(dx)))
        if w < 1: continue
        T = tower_tex(w, hp, z0m, z_top, q == main)
        L = np.hypot(dx, dy); shade = 1.0 if ny / L > 0.8 else (0.82 if nx > 0 else 0.94)
        aa, bb = (a, b) if dx > 0 else (b, a)
        if dx < 0: T = T[:, ::-1]
        blit_wall(canvas, aa - [0, lift], bb - [0, lift], T, shade)
        for p_ in (aa, bb):
            X, Y = int(round(p_[0])), int(round(p_[1] - lift))
            if 0 <= X < W: canvas[max(0, Y - hp):max(0, Y), X] = (150, 132, 100)
    # верх (за фото): пласка покрівля, парапет-балюстрада по периметру, тумби-пінаклі на кутах
    zt, zp = zpx(z_top), zpx(z_roof)
    top = np.array([(p_[0], p_[1] - zt) for p_ in G_])
    WH = (244, 240, 230)
    cv2.fillPoly(canvas, [np.round(top).astype(np.int32)], (150, 146, 136))          # покрівля
    for k in range(4):                                                    # парапет: дальні сторони спершу
        pass
    order = sorted(range(4), key=lambda k: (top[k][1] + top[(k + 1) % 4][1]))
    for k in order:
        a, b = top[k], top[(k + 1) % 4]
        hpp = int(round(zp - zt))
        Tp = np.zeros((hpp, max(1, int(round(abs(b[0] - a[0])))), 3), np.uint8); Tp[:] = WH
        Tp[1:hpp - 1, ::2] = (150, 146, 136)                               # балясини
        aa, bb = (a, b) if b[0] > a[0] else (b, a)
        if abs(b[0] - a[0]) >= 1: blit_wall(canvas, aa, bb, Tp, 0.92 if k in order[:2] else 1.0)
    for p_ in top:                                                        # пінаклі
        X, Y = int(round(p_[0])), int(round(p_[1] - zp))
        for dy_ in range(0, 5):
            for dx_ in (-1, 0, 1):
                if 0 <= Y + dy_ < H and 0 <= X + dx_ < W: canvas[Y + dy_, X + dx_] = WH if dx_ < 1 else dark(WH, 0.8)

B = stylize_ground(gnd)
SHOP_COLS = [(190, 52, 48), (40, 104, 168), (228, 164, 36), (54, 132, 76), (132, 60, 140)]
blist = []
for i, b, pts in [(i, b, pts) for i, b, pts, ptf in C['blds']]:
    P = np.array(pts, np.float32)
    if len(P) < 3: continue
    rb, m = roof_colors(i, b, pts)
    if rb is None: continue
    am2 = abs(cv2.contourArea(P)) * MPP * MPP
    h = float(b.get('h') or 6)
    floors = max(1, int(round(h / 3.0)))
    tp = btype(b, floors, am2)
    if tp == 'shed': floors = 1
    lm = None
    for lx_, ly_, lk in LMK:
        if lk == 'tower': continue
        if cv2.pointPolygonTest(P, (float(lx_ - x0), float(ly_ - y0)), True) > -25: lm = lk
    if lm == 'cinema': floors = 4
    if lm == 'mall': floors = 6
    if lm == 'depot': floors = 4
    blist.append(dict(i=i, b=b, P=P, rb=rb, floors=floors, tp=tp, hp=floors * FLOOR, lm=lm))
# тіні будинків на землю (світло з північного заходу: тінь праворуч-вниз)
for o in blist:
    sh = np.round(o['P'] + [o['hp'] * 0.45, 2]).astype(np.int32)
    ms = np.zeros((H, W), np.uint8); cv2.fillPoly(ms, [sh], 1); cv2.fillPoly(ms, [np.round(o['P']).astype(np.int32)], 1)
    B[ms.astype(bool)] = (B[ms.astype(bool)] * 0.74).astype(np.uint8)
objs = [(o['P'][:, 1].max(), 'b', o) for o in blist] + [(y - y0 + r * 0.3, 't', (x - x0, y - y0, r, k)) for x, y, r, k in tl]
for lx_, ly_, lk in LMK:                                                # вежа — після будинку, на якому стоїть
    if lk != 'tower' or not (-60 < lx_ - x0 < W + 60 and -60 < ly_ - y0 < H + 200): continue
    by_ = ly_ - y0 + 12; z0 = 0.0; ang = 0.0
    for o in blist:
        if cv2.pointPolygonTest(o['P'], (float(lx_ - x0), float(ly_ - y0)), True) > -30:
            by_ = max(by_, o['P'][:, 1].max() + 0.5)   # за фото вежа росте від землі на розі, малюємо після депо
            Pp = o['P']; e = [(np.hypot(*(Pp[(k + 1) % len(Pp)] - Pp[k])), Pp[(k + 1) % len(Pp)] - Pp[k]) for k in range(len(Pp))]
            d_ = max(e, key=lambda t: t[0])[1]; ang = float(np.arctan2(d_[1], d_[0])) % (np.pi / 2)   # напрям стін депо
    objs.append((by_, 'tower', (lx_ - x0, ly_ - y0, z0, ang)))
for base_y, kind, o in sorted(objs, key=lambda z: z[0]):
    if kind == 'tower':
        draw_tower(B, *o); continue
    if kind == 't':
        x, y, r, k = o
        if 1 <= x < W - 1 and 0 < y: B[max(0, int(y - r * 0.7)):min(H, int(y)), int(x) - 1:int(x) + 1] = (92, 70, 52)
        draw_tree(B, x, y, r, k, lift=r * 0.9); continue
    P, floors, tp, hp = o['P'], o['floors'], o['tp'], o['hp']
    rs = random.Random(o['i'])
    wall = jit(PAL[tp][rs.randrange(len(PAL[tp]))], rs, 6)
    n = len(P)
    area2 = sum(P[k][0] * P[(k + 1) % n][1] - P[(k + 1) % n][0] * P[k][1] for k in range(n))
    sgn = 1 if area2 > 0 else -1
    shop_here = [s for s in shops if cv2.pointPolygonTest(P, (float(s[0]), float(s[1])), True) > -16]
    shop_col = SHOP_COLS[rs.randrange(len(SHOP_COLS))] if shop_here else None
    walls = []
    for k in range(n):
        a, c = P[k], P[(k + 1) % n]
        dx, dy = c[0] - a[0], c[1] - a[1]
        nx, ny = dy * sgn, -dx * sgn
        L = np.hypot(dx, dy)
        if L < 1 or ny <= 0: continue
        walls.append((a, c, dx, dy, nx, ny, L))
    # головна (найдовша видима) стіна житлового будинку отримує під'їзди
    main_k = max(range(len(walls)), key=lambda q: abs(walls[q][2])) if walls else -1
    for q, (a, c, dx, dy, nx, ny, L) in enumerate(walls):
        w = int(round(abs(dx)))
        if w < 1: continue
        face = ny / L                                                  # 1 — прямо на камеру, 0 — вбік
        shade = 1.0 if face > 0.8 else (0.8 if nx * sgn * 0 + nx > 0 else 0.92)
        entr = None
        if q == main_k and tp in ('panel', 'brick', 'old') and floors > 1:
            ncol = max(1, min(int(round(L * MPP / 3.2)), w // 5))
            entr = set(range(ncol // 4, ncol, max(3, ncol // 3))) if ncol >= 3 else {ncol // 2}
        T = lm_tex(o['lm'], max(w, 1), hp, q == main_k or (o.get('lm') == 'mall' and w > 120), rs) if o.get('lm') else None
        if T is None: T = wall_tex(tp, max(w, 1), floors, L * MPP, rs, wall, shop_col if (shop_here and q == main_k) else None, entr)
        if dx < 0 and not o.get('lm'): T = T[:, ::-1]                   # текстура йде зліва направо по екрану (написи не дзеркалимо)
        aa, cc = (a, c) if dx > 0 else (c, a)
        blit_wall(B, aa, cc, T, shade)
        # кути будинку — темна вертикаль
        for p_ in (aa, cc):
            X, Y = int(round(p_[0])), int(round(p_[1]))
            if 0 <= X < W: B[max(0, Y - hp):max(0, min(H, Y)), X] = dark(wall, 0.5)
    draw_roof_flat(B, o['b'], P, o['rb'], dy=hp)
    # лінія карниза між дахом і стіною — у draw_roof_flat контур уже темний

for name, im in (('A', A), ('B', B)):
    crop = im[PADT:PADT + SH, :W]
    crop = ((crop.astype(np.uint16) >> 3) << 3).astype(np.uint8)          # 15 біт, як на PSP
    cv2.imwrite(os.path.join(OUT, '%s_%s.png' % (tag, name)), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
print('ok', len(C['blds']), 'будинків', len(tl), 'дерев', len(shops), 'закладів')
