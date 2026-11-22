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
KV = 0.62                                                              # стискання висоти для виду 3/4
FLOOR = 3.0 / MPP * KV                                                 # пікселів на поверх
WALLS = [(214, 206, 188), (196, 192, 184), (226, 214, 176), (204, 160, 128), (232, 222, 206), (186, 196, 200), (220, 196, 160)]
B = stylize_ground(gnd)
objs = []
for i, b, pts, ptf in C['blds']:
    P = np.array(pts, np.float32)
    objs.append((P[:, 1].max(), 'b', (i, b, pts)))
for x, y, r, k in tl:
    objs.append((y - y0 + r * 0.3, 't', (x - x0, y - y0, r, k)))
for base_y, kind, o in sorted(objs, key=lambda z: z[0]):
    if kind == 't':
        x, y, r, k = o
        cv2.line(B, (int(x), int(y)), (int(x), int(y - r * 0.6)), (92, 70, 52), 2)      # стовбур
        draw_tree(B, x, y, r, k, lift=r * 0.9); continue
    i, b, pts = o
    rb, m = roof_colors(i, b, pts)
    if rb is None: continue
    t = b.get('t', {}); bt = t.get('building', 'yes')
    h = float(b.get('h') or 6)
    small = bt in ('garage', 'garages', 'shed', 'kiosk') or cv2.contourArea(np.array(pts, np.float32)) * MPP * MPP < 30
    if small: h = min(h, 3)
    floors = max(1, int(round(h / 3.0)))
    hp = floors * FLOOR
    rs = random.Random(i)
    wall = WALLS[rs.randrange(len(WALLS))]
    if bt in ('house', 'detached', 'semidetached_house'): wall = [(232, 220, 190), (226, 206, 170), (240, 236, 222), (214, 222, 200)][rs.randrange(4)]
    if floors >= 9: wall = [(210, 210, 204), (226, 222, 212), (196, 204, 208)][rs.randrange(3)]
    P = np.array(pts, np.float32)
    if cv2.contourArea(P) < 0: P = P[::-1]
    n = len(P)
    # орієнтація контуру: зовнішня нормаль ребра (a->b) = (dy, -dx) при обході за годинниковою стрілкою в екранних координатах
    area2 = sum(P[k][0] * P[(k + 1) % n][1] - P[(k + 1) % n][0] * P[k][1] for k in range(n))
    sgn = 1 if area2 > 0 else -1
    shop_here = [s for s in shops if cv2.pointPolygonTest(P, (float(s[0]), float(s[1])), True) > -14]
    for k in range(n):
        a, c = P[k], P[(k + 1) % n]
        dx, dy = c[0] - a[0], c[1] - a[1]
        nx, ny = dy * sgn, -dx * sgn                                     # зовнішня нормаль
        L = np.hypot(dx, dy)
        if L < 1 or ny <= 0.05 * L: continue                            # видимі лише стіни, що дивляться «до камери»
        shade = 1.0 if abs(nx) < 0.5 * L else (0.86 if nx > 0 else 0.94)
        wc = dark(wall, shade)
        quad = np.round(np.array([a, c, c - [0, hp], a - [0, hp]])).astype(np.int32)
        cv2.fillPoly(B, [quad], wc)
        # вікна: колонки кожні ~3 м, ряди за поверхами
        step = 3.0 / MPP; ncol = int(L // step)
        if ncol >= 1 and not small:
            for j in range(ncol):
                tt = (j + 0.5) / ncol
                gx_, gy_ = a[0] + dx * tt, a[1] + dy * tt
                for f in range(floors):
                    yb = gy_ - f * FLOOR
                    ground = f == 0
                    if ground and shop_here:                             # вітрина
                        cv2.rectangle(B, (int(gx_ - step * 0.42), int(yb - FLOOR * 0.85)), (int(gx_ + step * 0.42), int(yb - 1)), (62, 92, 120), -1)
                        cv2.line(B, (int(gx_ - step * 0.42), int(yb - FLOOR * 0.85)), (int(gx_ + step * 0.42), int(yb - FLOOR * 0.85)), (150, 190, 214), 1)
                        continue
                    if ground and j == ncol // 2 and floors > 1:          # під'їзд
                        cv2.rectangle(B, (int(gx_ - 2), int(yb - FLOOR * 0.8)), (int(gx_ + 2), int(yb - 1)), (98, 70, 52), -1)
                        continue
                    wx0, wy0 = int(gx_ - 2), int(yb - FLOOR * 0.72)
                    cv2.rectangle(B, (wx0 - 1, wy0 - 1), (wx0 + 4, wy0 + 3), dark(wc, 0.8), -1)
                    lit_ = rs.random() < 0.15
                    cv2.rectangle(B, (wx0, wy0), (wx0 + 3, wy0 + 2), (236, 214, 140) if lit_ else (70, 96, 124), -1)
                    if 0 <= wy0 < H and 0 <= wx0 < W - 2: B[wy0, wx0:wx0 + 2] = (150, 180, 204) if not lit_ else (250, 236, 180)
            if shop_here:                                                # вивіска над вітринами
                y_s = int(min(a[1], c[1]) - FLOOR * 0.95)
                sc = [(196, 56, 52), (46, 110, 170), (232, 172, 40), (60, 140, 80)][rs.randrange(4)]
                cv2.rectangle(B, (int(min(a[0], c[0]) + 2), y_s - 3), (int(max(a[0], c[0]) - 2), y_s), sc, -1)
                for q in range(int(min(a[0], c[0]) + 5), int(max(a[0], c[0]) - 5), 3):
                    if rs.random() < 0.7 and 0 <= q < W and 2 <= y_s < H: B[y_s - 2, q] = (250, 246, 236)
        cv2.polylines(B, [quad], True, dark(wc, 0.55), 1)
        cv2.line(B, tuple(np.round(a).astype(int)), tuple(np.round(c).astype(int)), dark(wc, 0.4), 1)
    draw_roof_flat(B, b, pts, rb, dy=hp)

for name, im in (('A', A), ('B', B)):
    crop = im[PADT:PADT + SH, :W]
    crop = ((crop.astype(np.uint16) >> 3) << 3).astype(np.uint8)          # 15 біт, як на PSP
    cv2.imwrite(os.path.join(OUT, '%s_%s.png' % (tag, name)), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
print('ok', len(C['blds']), 'будинків', len(tl), 'дерев', len(shops), 'закладів')
