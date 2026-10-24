"""Дрібні деталі міста для citydraw.render (детальна сітка при надвибірці ss).
Усе детерміноване світовими координатами (однаково в будь-якому блоці, без швів).
Група 1: крайова розмітка, стоп-лінії, світлофори, ліхтарі, зупинки з павільйонами, урни, листя й дрібне сміття.
"""
import math, random
import numpy as np, cv2

MAIN = ('primary', 'secondary', 'tertiary')
WHITE = (226, 226, 214)
POLE = (58, 60, 66)


def _rng(x, y, salt=0):
    return random.Random((int(x) * 73856093) ^ (int(y) * 19349663) ^ salt)


def _ip(p):
    return (int(round(p[0])), int(round(p[1])))


class Ctx:
    """Спільні дані блоку: img (детальна), cls (детальна), S, MF (м/пікс детальної), x0, y0 (світ, сітка гри), D, ways, tofn."""
    def __init__(self, **kw): self.__dict__.update(kw)
    def m(self, metres): return metres / self.MF                       # метри -> детальні пікселі
    def world(self, p): return (p[0] / self.S + self.x0, p[1] / self.S + self.y0)   # детальні -> світ (сітка гри)


def _drive_ways(c):
    from citydraw import ROAD_W
    out = []
    for w, pts, ptf in c.ways:
        t = w['t']; h = t.get('highway')
        if h in ROAD_W and t.get('area') != 'yes' and t.get('tunnel') != 'yes': out.append((h, ptf, w))
    return out


def _junctions(dw):
    cnt = {}
    for h, ptf, w in dw:
        for k, nid in enumerate(w['nds']):
            cnt[nid] = cnt.get(nid, 0) + (1 if k in (0, len(w['nds']) - 1) else 2)
    return {n for n, v in cnt.items() if v >= 3}


def road_marks(c):
    """Крайові лінії на головних двосторонніх вулицях (з розривом на перехрестях) і стоп-лінії біля світлофорів/знаків STOP."""
    from citydraw import ROAD_W
    dw = _drive_ways(c); jn = _junctions(dw)
    img, S = c.img, c.S
    lw = max(1, int(round(c.m(0.35))))
    for h, ptf, w in dw:
        if h not in MAIN: continue
        hw = c.m(ROAD_W[h]) / 2 - c.m(0.45)
        nds = w['nds']
        for i in range(len(ptf) - 1):
            (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < 2: continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux
            cut0 = c.m(ROAD_W[h] / 2 + 4) if nds[i] in jn else 0       # не малювати через перехрестя
            cut1 = c.m(ROAD_W[h] / 2 + 4) if nds[i + 1] in jn else 0
            if L - cut0 - cut1 < 4: continue
            a = (xa + ux * cut0, ya + uy * cut0); b = (xb - ux * cut1, yb - uy * cut1)
            for sg in (-1, 1):
                if w['t'].get('oneway') == 'yes' and False: pass
                cv2.line(img, _ip((a[0] + nx * hw * sg, a[1] + ny * hw * sg)), _ip((b[0] + nx * hw * sg, b[1] + ny * hw * sg)), WHITE, lw)
    # стоп-лінії й світлофори
    sig = {nid for nid, t in c.D['n'].items() if t.get('highway') in ('traffic_signals', 'stop')}
    for h, ptf, w in dw:
        nds = w['nds']; hw = c.m(ROAD_W[h]) / 2
        for k, nid in enumerate(nds):
            if nid not in sig: continue
            tl = c.D['n'][nid].get('highway') == 'traffic_signals'
            for j in (k - 1, k + 1):
                if j < 0 or j >= len(nds): continue
                (xa, ya), (xb, yb) = ptf[j], ptf[k]                # підхід: від сусіда до вузла
                L = math.hypot(xb - xa, yb - ya)
                if L < c.m(12) or (nid not in jn and not tl): continue
                ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux        # праворуч від напрямку руху
                back = hw + c.m(3.0) if nid in jn else c.m(1.5)
                px_, py_ = xb - ux * back, yb - uy * back
                ow = w['t'].get('oneway') == 'yes'
                s0 = -hw if ow else 0.0
                cv2.line(img, _ip((px_ + nx * s0, py_ + ny * s0)), _ip((px_ + nx * hw, py_ + ny * hw)), WHITE, max(1, int(round(c.m(0.4)))))
                if tl: traffic_light(c, px_ + nx * (hw + c.m(0.9)), py_ + ny * (hw + c.m(0.9)), ux, uy)


def traffic_light(c, x, y, ux, uy):
    """Вид згори: стовп і корпус уздовж дороги, лінзи червона/жовта/зелена."""
    img = c.img; S = c.S
    cv2.circle(img, _ip((x, y)), max(1, int(round(c.m(0.3)))), POLE, -1)
    L = c.m(1.9); Wd = c.m(0.75)
    cx, cy = x - ux * c.m(0.2), y - uy * c.m(0.2)
    nx, ny = -uy, ux
    box = np.array([[cx - ux * L / 2 - nx * Wd / 2, cy - uy * L / 2 - ny * Wd / 2], [cx + ux * L / 2 - nx * Wd / 2, cy + uy * L / 2 - ny * Wd / 2],
                    [cx + ux * L / 2 + nx * Wd / 2, cy + uy * L / 2 + ny * Wd / 2], [cx - ux * L / 2 + nx * Wd / 2, cy - uy * L / 2 + ny * Wd / 2]])
    cv2.fillPoly(img, [np.round(box).astype(np.int32)], (30, 32, 36))
    for k, col in enumerate(((230, 60, 50), (240, 196, 60), (70, 210, 110))):
        t = (k - 1) * L * 0.3
        cv2.circle(img, _ip((cx + ux * t, cy + uy * t)), max(1, int(round(c.m(0.22)))), col, -1)


def lamps(c):
    """Ліхтарі вздовж головних вулиць: стовп на краю тротуару, кронштейн над дорогою, світильник."""
    from citydraw import ROAD_W
    dw = _drive_ways(c); jn = _junctions(dw)
    img = c.img; step = c.m(30)
    for h, ptf, w in dw:
        if h not in MAIN: continue
        hw = c.m(ROAD_W[h]) / 2
        acc = 0.0
        for i in range(len(ptf) - 1):
            (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < 1: continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux
            s = (step - acc) % step
            while s < L:
                px_, py_ = xa + ux * s, ya + uy * s
                wx, wy = c.world((px_, py_))
                sg = 1 if (int(wx // 60) + int(wy // 60)) % 2 == 0 else -1        # шаховий порядок по боках
                for side in ((sg,) if w['t'].get('oneway') == 'yes' else (1, -1)):
                    bx, by = px_ + nx * (hw + c.m(0.7)) * side, py_ + ny * (hw + c.m(0.7)) * side
                    if c.cls[min(c.cls.shape[0] - 1, max(0, int(by))), min(c.cls.shape[1] - 1, max(0, int(bx)))] == 5: continue   # не на дорозі
                    ax, ay = bx - nx * c.m(2.0) * side, by - ny * c.m(2.0) * side
                    cv2.line(img, _ip((bx + c.m(0.3), by + c.m(0.4))), _ip((ax + c.m(0.3), ay + c.m(0.4))), (96, 96, 96), max(1, int(round(c.m(0.22)))))   # тінь
                    cv2.line(img, _ip((bx, by)), _ip((ax, ay)), (70, 72, 78), max(1, int(round(c.m(0.22)))))
                    cv2.circle(img, _ip((bx, by)), max(1, int(round(c.m(0.32)))), POLE, -1)
                    cv2.ellipse(img, _ip((ax, ay)), (max(1, int(c.m(0.6))), max(1, int(c.m(0.3)))), math.degrees(math.atan2(uy, ux)), 0, 360, (54, 56, 60), -1)
                    cv2.ellipse(img, _ip((ax, ay)), (max(1, int(c.m(0.45))), max(1, int(c.m(0.18)))), math.degrees(math.atan2(uy, ux)), 0, 360, (226, 226, 214), -1)
                s += step
            acc = (acc + L) % step


def bus_stops(c):
    """Павільйон на тротуарі з того боку, де стоїть точка зупинки; урна поруч."""
    from citydraw import ROAD_W
    import citygen as CG
    dw = _drive_ways(c)
    img = c.img
    for nid, t in c.D['n'].items():
        if t.get('highway') != 'bus_stop' or nid not in c.D['nodes']: continue
        p = c.tofn(nid)
        if not (-200 < p[0] < img.shape[1] + 200 and -200 < p[1] < img.shape[0] + 200): continue
        best = None
        for h, ptf, w in dw:
            for i in range(len(ptf) - 1):
                (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; dx, dy = xb - xa, yb - ya; L2 = dx * dx + dy * dy
                if L2 < 1: continue
                tt = max(0, min(1, ((p[0] - xa) * dx + (p[1] - ya) * dy) / L2))
                qx, qy = xa + dx * tt, ya + dy * tt; d = math.hypot(p[0] - qx, p[1] - qy)
                if best is None or d < best[0]: best = (d, qx, qy, dx / math.sqrt(L2), dy / math.sqrt(L2), h)
        if best is None or best[0] > c.m(30): continue
        d, qx, qy, ux, uy, h = best
        nx, ny = -uy, ux
        sg = 1 if (p[0] - qx) * nx + (p[1] - qy) * ny >= 0 else -1
        hw = c.m(ROAD_W[h]) / 2
        cx, cy = qx + nx * sg * (hw + c.m(1.6)), qy + ny * sg * (hw + c.m(1.6))
        L, Wd = c.m(3.6), c.m(1.4)
        box = np.array([[-L / 2, -Wd / 2], [L / 2, -Wd / 2], [L / 2, Wd / 2], [-L / 2, Wd / 2]])
        R = np.array([[ux, nx * sg], [uy, ny * sg]])
        P = (box @ R.T) + [cx, cy]
        cv2.fillPoly(img, [np.round(P + [c.m(0.25), c.m(0.35)]).astype(np.int32)], (90, 92, 94))          # тінь
        cv2.fillPoly(img, [np.round(P).astype(np.int32)], (44, 58, 70))
        inner = (box * [0.86, 0.6]) @ R.T + [cx, cy]
        cv2.fillPoly(img, [np.round(inner).astype(np.int32)], (104, 150, 176))                            # скляний дах
        cv2.line(img, _ip(((-L * 0.43, -Wd * 0.3) @ R.T + [cx, cy])), _ip(((L * 0.43, -Wd * 0.3) @ R.T + [cx, cy])), (176, 210, 226), max(1, int(c.m(0.1))))
        bin_(c, cx + ux * (L / 2 + c.m(0.8)), cy + uy * (L / 2 + c.m(0.8)))
        # жовта розмітка «зупинка» на краю проїжджої частини
        ex, ey = qx + nx * sg * (hw - c.m(0.3)), qy + ny * sg * (hw - c.m(0.3))
        cv2.line(img, _ip((ex - ux * c.m(8), ey - uy * c.m(8))), _ip((ex + ux * c.m(8), ey + uy * c.m(8))), (226, 190, 60), max(1, int(round(c.m(0.15)))))


def bin_(c, x, y):
    cv2.circle(c.img, _ip((x + c.m(0.2), y + c.m(0.25))), max(1, int(round(c.m(0.5)))), (92, 94, 92), -1)
    cv2.circle(c.img, _ip((x, y)), max(1, int(round(c.m(0.5)))), (40, 84, 56), -1)
    cv2.circle(c.img, _ip((x, y)), max(1, int(round(c.m(0.28)))), (22, 26, 24), -1)


def street_litter(c, trees):
    """Головні вулиці й центр: урни на розі, опале листя (більше під деревами), трохи дрібного сміття. Не в приватному секторі."""
    from citydraw import ROAD_W
    img, cls, S = c.img, c.cls, c.S
    H, W = cls.shape
    main = np.zeros((H // S + 1, W // S + 1), np.uint8)
    for h, ptf, w in _drive_ways(c):
        if h in MAIN or w['t'].get('highway') == 'pedestrian':
            cv2.polylines(main, [np.round(np.array(ptf) / S).astype(np.int32)], False, 1, int(c.m(ROAD_W[h] + 12) / S))
    for w, pts, ptf in c.ways:
        if w['t'].get('highway') == 'pedestrian' or w['t'].get('place') == 'square':
            if w['closed']: cv2.fillPoly(main, [np.round(np.array(ptf) / S).astype(np.int32)], 1)
            else: cv2.polylines(main, [np.round(np.array(ptf) / S).astype(np.int32)], False, 1, 16)
    mainf = np.repeat(np.repeat(main, S, 0), S, 1)[:H, :W].astype(bool)
    G_SIDE, G_ROAD, G_PED, G_GRASS = 4, 5, 7, 2
    walk = np.isin(cls, (G_SIDE, G_PED)) & mainf
    road_edge = (cls == G_ROAD) & mainf & cv2.dilate((cls != G_ROAD).astype(np.uint8), np.ones((int(c.m(1.2)) | 1,) * 2, np.uint8)).astype(bool)
    rs = np.random.RandomState((int(c.x0) * 7919 + int(c.y0)) & 0x7FFFFFFF)
    # листя: щільніше біля дерев
    near = np.zeros((H, W), np.float32)
    for x, y, rr, k in trees:
        cv2.circle(near, (int((x + 0.5) * S), int((y + 0.5) * S)), int(rr * S * 1.4), 1.0, -1)
    near = cv2.GaussianBlur(near, (0, 0), 3 * S)
    cand = (walk | road_edge | ((cls == G_GRASS) & mainf))
    p = (0.00025 + 0.005 * near) * cand
    ys, xs = np.nonzero(rs.rand(H, W) < p)
    LEAF = [(214, 160, 52), (196, 112, 40), (150, 92, 44), (226, 196, 84)]
    for y, x in zip(ys, xs):
        col = LEAF[(x * 31 + y * 17) % 4]
        r = max(1, int(c.m(0.22)))
        cv2.ellipse(img, (int(x), int(y)), (r + (x % 2), max(1, r - 1)), (x * 37) % 180, 0, 360, col, -1)
    # сміття: зрідка світлі папірці, кришки
    ys, xs = np.nonzero((rs.rand(H, W) < 0.00012) & walk)
    LIT = [(232, 232, 226), (210, 70, 60), (70, 110, 190), (240, 210, 90)]
    for y, x in zip(ys, xs):
        cv2.rectangle(img, (int(x), int(y)), (int(x) + max(1, int(c.m(0.35))), int(y) + max(1, int(c.m(0.22)))), LIT[(x + y) % 4], -1)
    # урни: на тротуарах головних вулиць, по сітці ~35 м, лише на тротуарі
    g = int(c.m(35))
    gx0 = int(c.x0 * S) % g; gy0 = int(c.y0 * S) % g
    for yy in range(g - gy0, H, g):
        for xx in range(g - gx0, W, g):
            # найближчий тротуар біля вузла сітки
            win = walk[max(0, yy - g // 3):yy + g // 3, max(0, xx - g // 3):xx + g // 3]
            if not win.any(): continue
            ys_, xs_ = np.nonzero(win)
            k = int((xx * 13 + yy * 7) % len(xs_))
            bin_(c, xs_[k] + max(0, xx - g // 3), ys_[k] + max(0, yy - g // 3))


# ---------------------------------------------------------------- група 2: дахи
SOLAR = (34, 48, 92); SOLAR_L = (88, 112, 160)


def _orient(P):
    (cx, cy), (rw, rh), a = cv2.minAreaRect(P.astype(np.float32))
    a = math.radians(a)
    if rw < rh: a += math.pi / 2; rw, rh = rh, rw
    return cx, cy, rw, rh, math.cos(a), math.sin(a)


def _rect(out, cx, cy, L, Wd, ux, uy, col):
    nx, ny = -uy, ux
    pts = np.array([[cx - ux * L / 2 - nx * Wd / 2, cy - uy * L / 2 - ny * Wd / 2], [cx + ux * L / 2 - nx * Wd / 2, cy + uy * L / 2 - ny * Wd / 2],
                    [cx + ux * L / 2 + nx * Wd / 2, cy + uy * L / 2 + ny * Wd / 2], [cx - ux * L / 2 + nx * Wd / 2, cy - uy * L / 2 + ny * Wd / 2]])
    cv2.fillPoly(out, [np.round(pts).astype(np.int32)], col)


def _solar(out, F, cx, cy, L, Wd, ux, uy, ss):
    """Масив панелей: темно-синій, світлі шви між модулями ~1x1,7 м."""
    m = np.zeros(F.shape, np.uint8); _rect(m, cx, cy, L, Wd, ux, uy, 1)
    m = m.astype(bool) & F
    if m.sum() < 6 * ss * ss: return
    yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
    a = (xx - cx) * ux + (yy - cy) * uy; b = -(xx - cx) * uy + (yy - cy) * ux
    seam = (np.mod(a, 2.0 * ss) < 0.35 * ss) | (np.mod(b, 3.4 * ss) < 0.35 * ss)       # модулі 1 x 1,7 м (у пікс гри 2 x 3,4)
    out[m] = SOLAR; out[m & seam] = SOLAR_L


def roof_details(out, F, d, P, typ, gxy, area, b, rs, ss):
    """Комини, тарілки, сонячні панелі на хатах; кондиціонери, вентиляція, тарілки, панелі на пласких дахах."""
    if F.sum() < 30 * ss * ss: return
    cx, cy, L, Wd, ux, uy = _orient(P)
    ys, xs = np.nonzero(F & (d > 1.6))
    if len(xs) == 0: return
    if typ == 'house':
        gx, gy, ridge = gxy
        # сонячні панелі на південному скаті (градієнт відстані до гребеня вказує на північ)
        if rs.random() < 0.09 and area > 60:
            south = F & (gy < -0.05 / ss) & (d > 1.2) & ~ridge
            if south.sum() > 20 * ss * ss:
                sy, sx = np.nonzero(south)
                _solar(out, F & south, sx.mean(), sy.mean(), min(L * 0.55, 9 * ss * 2), min(Wd * 0.3, 4 * ss * 2), ux, uy, ss)
        # комин: цегла з темним отвором і тінню
        if rs.random() < 0.75:
            k = rs.randrange(len(xs)); x, y = xs[k], ys[k]
            s = 0.7 * ss * 2 / 2 * 1.0
            cv2.rectangle(out, (int(x + s * 0.6), int(y + s * 0.6)), (int(x + s * 1.8), int(y + s * 1.8)), (60, 52, 50), -1)
            cv2.rectangle(out, (int(x - s), int(y - s)), (int(x + s), int(y + s)), (148, 78, 58), -1)
            cv2.rectangle(out, (int(x - s * 0.45), int(y - s * 0.45)), (int(x + s * 0.45), int(y + s * 0.45)), (40, 34, 32), -1)
        if rs.random() < 0.18:
            k = rs.randrange(len(xs)); x, y = xs[k], ys[k]
            cv2.ellipse(out, (int(x), int(y)), (int(0.45 * ss * 2), int(0.3 * ss * 2)), rs.randrange(180), 0, 360, (210, 212, 210), -1)
            cv2.circle(out, (int(x), int(y)), max(1, int(0.12 * ss * 2)), (90, 90, 92), -1)
    elif typ == 'flat':
        if rs.random() < 0.06 and area > 300:                        # ряди панелей на великому пласкому даху
            for k in range(-2, 3):
                _solar(out, F & (d > 2.5), cx - uy * k * 3.0 * ss, cy + ux * k * 3.0 * ss, L * 0.7, 1.7 * ss * 2 * 0.6, ux, uy, ss)
        n = min(14, int(area / 160) + 1)
        for _ in range(n):                                            # кондиціонери / вентиляція
            k = rs.randrange(len(xs)); x, y = xs[k], ys[k]
            if d[y, x] < 2.5: continue
            if rs.random() < 0.6:
                _rect(out, x + ss * 0.6, y + ss * 0.8, 2.0 * ss, 1.6 * ss, ux, uy, (90, 92, 96))          # тінь
                _rect(out, x, y, 2.0 * ss, 1.6 * ss, ux, uy, (196, 198, 200))
                cv2.circle(out, (int(x), int(y)), max(1, int(0.55 * ss)), (70, 72, 78), -1)
            else:
                _rect(out, x + ss * 0.6, y + ss * 0.8, 3.0 * ss, 2.0 * ss, ux, uy, (80, 82, 86))
                _rect(out, x, y, 3.0 * ss, 2.0 * ss, ux, uy, (150, 152, 156))
                _rect(out, x, y, 2.2 * ss, 0.5 * ss, ux, uy, (110, 112, 116))
        if area > 400:
            for _ in range(rs.randrange(1, 4)):                       # тарілки біля краю
                e = np.nonzero(F & (d > 1.2) & (d < 2.5))
                if len(e[0]) == 0: break
                k = rs.randrange(len(e[0])); y, x = e[0][k], e[1][k]
                cv2.ellipse(out, (int(x), int(y)), (int(0.9 * ss), int(0.6 * ss)), rs.randrange(180), 0, 360, (214, 216, 214), -1)


# ---------------------------------------------------------------- група 2: ділянки приватного сектору
FENCES = [((122, 94, 64), 0.25), ((92, 104, 114), 0.22), ((154, 162, 152), 0.15), ((150, 84, 62), 0.4), ((70, 98, 72), 0.22)]
HOUSE_T = ('house', 'detached', 'semidetached_house', 'yes', 'residential', 'bungalow')


def _is_house(b):
    bt = b['t'].get('building', 'yes')
    return b['area'] <= 320 and bt in HOUSE_T and b.get('h', 6) <= 10


def plots(c, blds, park_b, bmask_b, solid_out=None):
    """Ділянка кожної хати (найближча до неї територія двору, до ~20 м): паркан, ворота з боку вулиці,
    у глибині — город рядами / теплиця / сарайчик, перед хатою — плитка. Склад визначається місцем (без повторів)."""
    S, img, cls = c.S, c.img, c.cls
    H, W = cls.shape; Hb, Wb = H // S, W // S
    clsb = cls[S // 2::S, S // 2::S][:Hb, :Wb]
    houses = [(b, np.array(ptf, np.float32) / S) for i, b, pts, ptf in blds if _is_house(b)]
    if len(houses) < 4: return
    cen = np.array([P.mean(0) for b, P in houses])
    # лише там, де хат густо (приватний сектор), а не одинокі будиночки серед багатоповерхівок
    dense = np.array([(np.hypot(*(cen - p).T) < 120).sum() >= 6 for p in cen])
    hm = np.zeros((Hb, Wb), np.int32)
    for k, (b, P) in enumerate(houses):
        if dense[k]: cv2.fillPoly(hm, [np.round(P).astype(np.int32)], k + 1)
    if not hm.any(): return
    dist, lab = cv2.distanceTransformWithLabels((hm == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # мітка найближчого пікселя хати -> номер хати
    seeds = np.zeros(lab.max() + 1, np.int32)
    zy, zx = np.nonzero(hm == 0)
    ny_, nx_ = np.nonzero(hm > 0)
    lab_h = lab[ny_, nx_]; seeds[lab_h] = hm[ny_, nx_]
    owner = seeds[lab]
    yard = np.isin(clsb, (1, 2, 9, 13)) & ~park_b[:Hb, :Wb] & ~bmask_b[:Hb, :Wb] & (hm == 0) & (dist < 40) & (owner > 0)
    road = (clsb == 5).astype(np.uint8)
    rdist, rlab = cv2.distanceTransformWithLabels(1 - road, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ry, rx = np.nonzero(road)
    rpos = np.zeros((rlab.max() + 1, 2), np.float32); rpos[rlab[ry, rx]] = np.stack([rx, ry], 1)
    for k, (b, P) in enumerate(houses):
        if not dense[k]: continue
        R = yard & (owner == k + 1)
        n = int(R.sum())
        if n < 60: continue
        if np.isin(clsb[R], (2, 9)).mean() < 0.12: continue           # двір вимощений (центр) — не город і не паркан
        ys, xs = np.nonzero(R)
        hx, hy = P.mean(0)
        wx, wy = hx + c.x0, hy + c.y0
        rs = _rng(wx, wy, 11)
        ix, iy = int(np.clip(hx, 0, Wb - 1)), int(np.clip(hy, 0, Hb - 1))
        rp = rpos[rlab[iy, ix]] if road.any() else np.array([hx, hy - 1])
        v = np.array([hx - rp[0], hy - rp[1]], np.float32); v /= (np.linalg.norm(v) + 1e-6)      # від вулиці вглиб
        proj = (xs - hx) * v[0] + (ys - hy) * v[1]
        ux, uy = -v[1], v[0]
        # ---- глибина ділянки: город рядами / садок / газон / нічого
        back = proj > 4
        choice = rs.random()
        if back.sum() > 80 and choice < 0.75:
            m = np.zeros((Hb, Wb), np.uint8); m[ys[back], xs[back]] = 1
            mf = cv2.erode(np.repeat(np.repeat(m, S, 0), S, 1), np.ones((2 * S + 1,) * 2, np.uint8)).astype(bool)
            if mf.any():
                fy, fx = np.nonzero(mf)
                if choice < 0.4:                                       # город рядами
                    t = ((fx / S - hx) * v[0] + (fy / S - hy) * v[1])
                    per = 2.4 + rs.random() * 0.8
                    row = np.mod(t, per) < per * 0.45
                    soil = np.array((116 + rs.randrange(-10, 10), 94, 70), np.float32)
                    green = np.array((98, 136 + rs.randrange(-10, 10), 70), np.float32)
                    base = img[fy, fx].astype(np.float32)
                    img[fy, fx] = np.clip(base * 0.25 + np.where(row[:, None], green, soil) * 0.75, 0, 255).astype(np.uint8)
                else:                                                  # газон (і, можливо, садок)
                    gcol = np.array((96 + rs.randrange(-8, 8), 142 + rs.randrange(-10, 10), 64), np.float32)
                    base = img[fy, fx].astype(np.float32)
                    img[fy, fx] = np.clip(base * 0.35 + gcol * 0.65, 0, 255).astype(np.uint8)
                    if choice < 0.6:                                   # садок: кілька невеликих дерев
                        from citydraw import _tree_v3
                        for _ in range(min(6, len(fx) // (60 * S * S) + 1)):
                            j = rs.randrange(len(fx))
                            _tree_v3(img, int(fx[j]), int(fy[j]), (3.0 + rs.random() * 2) * S, 'g' if rs.random() < 0.8 else 'y', rs, S)
        # ---- плитка перед хатою
        front = (proj < -1) & (proj > -10)
        if front.sum() > 30 and rs.random() < 0.5:
            fyb, fxb = ys[front], xs[front]
            for by_, bx_ in zip(fyb, fxb):
                y0_, x0_ = by_ * S, bx_ * S
                tile = (172, 170, 164) if ((bx_ + by_) % 2) else (160, 158, 152)
                img[y0_:y0_ + S, x0_:x0_ + S] = tile
        # ---- теплиця / сарайчик у глибині
        if back.sum() > 120:
            bys, bxs = ys[back], xs[back]
            far = np.argsort(-proj[back])
            if rs.random() < 0.4:
                j = far[min(len(far) - 1, len(far) // 4)]
                _box(c, bxs[j], bys[j], 12, 6, ux, uy, 'green', solid_out)      # теплиця 6 x 3 м
            if rs.random() < 0.55:
                j = far[0]
                _box(c, bxs[j], bys[j], 6, 5, ux, uy, 'shed', solid_out, rs)     # сарай 3 x 2,5 м
        # ---- паркан по межі ділянки
        m = R.astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col, wm = FENCES[rs.randrange(len(FENCES))]
        for cn in cnts:
            if len(cn) < 8: continue
            q = cv2.approxPolyDP(cn.astype(np.float32), 1.2, True).reshape(-1, 2)
            qf = q * S + S / 2
            cv2.polylines(img, [np.round(qf).astype(np.int32)], True, col, max(1, int(round(c.m(wm)))))
            if solid_out is not None: solid_out.append(('line', [(float(x + c.x0), float(y + c.y0)) for x, y in q]))
            # ворота: ділянка межі, найближча до вулиці, — темніша смуга
            dq = [rdist[int(np.clip(y, 0, Hb - 1)), int(np.clip(x, 0, Wb - 1))] for x, y in q]
            j = int(np.argmin(dq))
            a, b2 = qf[j], qf[(j + 1) % len(qf)]
            L = np.hypot(*(b2 - a))
            if L > c.m(3.5):
                mid = (a + b2) / 2; dv = (b2 - a) / L
                cv2.line(img, tuple(np.round(mid - dv * c.m(1.7)).astype(int)), tuple(np.round(mid + dv * c.m(1.7)).astype(int)), (60, 62, 66), max(1, int(round(c.m(0.35)))))


def _box(c, bx, by, L, Wd, ux, uy, kind, solid_out, rs=None):
    """Невелика споруда на ділянці (розміри в пікселях гри): теплиця з ребрами або сарай з двоскатним дахом; з тінню."""
    S, img = c.S, c.img
    cx, cy = bx * S + S / 2, by * S + S / 2
    Lf, Wf = L * S, Wd * S
    nx, ny = -uy, ux
    corners = np.array([[-Lf / 2, -Wf / 2], [Lf / 2, -Wf / 2], [Lf / 2, Wf / 2], [-Lf / 2, Wf / 2]])
    Rm = np.array([[ux, nx], [uy, ny]])
    P = corners @ Rm.T + [cx, cy]
    cv2.fillPoly(img, [np.round(P + [c.m(0.6), c.m(0.8)]).astype(np.int32)], (96, 96, 92))                 # тінь
    if kind == 'green':
        cv2.fillPoly(img, [np.round(P).astype(np.int32)], (214, 226, 220))
        for t in np.arange(-Lf / 2 + c.m(1.0), Lf / 2, c.m(1.0)):
            a = np.array([t, -Wf / 2]) @ Rm.T + [cx, cy]; b = np.array([t, Wf / 2]) @ Rm.T + [cx, cy]
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (176, 190, 186), max(1, S // 3))
        a = np.array([-Lf / 2, 0]) @ Rm.T + [cx, cy]; b = np.array([Lf / 2, 0]) @ Rm.T + [cx, cy]
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (240, 246, 244), max(1, S // 3))
        cv2.polylines(img, [np.round(P).astype(np.int32)], True, (150, 162, 158), max(1, S // 3))
    else:
        roof = [(116, 84, 64), (136, 140, 146), (92, 96, 104), (150, 74, 54)][rs.randrange(4) if rs else 0]
        r = np.array(roof, np.float32)
        half1 = np.array([[-Lf / 2, -Wf / 2], [Lf / 2, -Wf / 2], [Lf / 2, 0], [-Lf / 2, 0]]) @ Rm.T + [cx, cy]
        half2 = np.array([[-Lf / 2, 0], [Lf / 2, 0], [Lf / 2, Wf / 2], [-Lf / 2, Wf / 2]]) @ Rm.T + [cx, cy]
        light = (ux * 0 + nx * -1) * -0.7 + (uy * 0 + ny * -1) * -0.7
        cv2.fillPoly(img, [np.round(half1).astype(np.int32)], tuple(int(v) for v in np.clip(r * (1.12 if light > 0 else 0.86), 0, 255)))
        cv2.fillPoly(img, [np.round(half2).astype(np.int32)], tuple(int(v) for v in np.clip(r * (0.86 if light > 0 else 1.12), 0, 255)))
        cv2.polylines(img, [np.round(P).astype(np.int32)], True, tuple(int(v) for v in r * 0.55), max(1, S // 3))
    if solid_out is not None:
        solid_out.append(('poly', [(float(x / S + c.x0), float(y / S + c.y0)) for x, y in P]))


# ---------------------------------------------------------------- група 3: центр, двори, вода
AWN = [((196, 52, 44), (236, 232, 222)), ((44, 120, 70), (236, 232, 222)), ((40, 86, 160), (236, 232, 222)),
       ((214, 160, 40), (120, 70, 40)), ((150, 40, 60), None), ((60, 60, 66), None), ((214, 104, 40), None)]
UMB = [(196, 52, 44), (40, 120, 70), (236, 232, 222), (214, 170, 40), (40, 86, 160), (110, 70, 50)]
TENT = [(52, 96, 176), (196, 56, 48), (232, 230, 220), (46, 128, 74), (226, 178, 46), (140, 60, 130)]


def _node_pts(c, pred):
    out = []
    H, W = c.cls.shape
    for nid, t in c.D['n'].items():
        if nid in c.D['nodes'] and pred(t):
            p = c.tofn(nid)
            if -100 < p[0] < W + 100 and -100 < p[1] < H + 100: out.append((t, p))
    for w, pts, ptf in c.ways:
        if w['closed'] and pred(w['t']):
            P = np.array(ptf); out.append((w['t'], tuple(P.mean(0)), P))
    return out


def awnings(c, blds):
    """Маркіза над входом магазину/кафе: смуга на ребрі будинку, найближчому до точки, назовні (над тротуаром)."""
    img, cls = c.img, c.cls
    H, W = cls.shape
    polys = [np.array(ptf, np.float32) for i, b, pts, ptf in blds]
    if not polys: return []
    cents = np.array([P.mean(0) for P in polys])
    placed = []
    for item in _node_pts(c, lambda t: 'shop' in t or t.get('amenity') in ('cafe', 'restaurant', 'fast_food', 'bar', 'pub', 'pharmacy', 'bank')):
        t, p = item[0], item[1]
        if len(item) > 2: continue                                     # лише точки (полігони — самі будівлі)
        near = np.nonzero(np.hypot(*(cents - p).T) < c.m(40))[0]
        best = None
        for k in near:
            P = polys[k]
            for i in range(len(P)):
                a, b = P[i], P[(i + 1) % len(P)]; d = b - a; L = np.hypot(*d)
                if L < c.m(3): continue
                tt = np.clip(np.dot(p - a, d) / (L * L), 0, 1); q = a + d * tt
                dist = np.hypot(*(p - q))
                if best is None or dist < best[0]: best = (dist, a, b, q, k)
        if best is None or best[0] > c.m(12): continue
        dist, a, b, q, k = best
        d = b - a; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
        if np.dot(cents[k] - q, n) > 0: n = -n                         # назовні від будинку
        probe = q + n * c.m(2.5)
        if not (0 <= probe[0] < W and 0 <= probe[1] < H) or cls[int(probe[1]), int(probe[0])] not in (4, 7, 13, 1): continue
        if any(np.hypot(*(q - pp)) < c.m(5) for pp in placed): continue
        placed.append(q)
        rs = _rng(*c.world(q), 21)
        col, col2 = AWN[rs.randrange(len(AWN))]
        Lw = min(L * 0.7, c.m(4 + rs.random() * 3)); Dp = c.m(1.4)
        quad = np.array([q - u * Lw / 2, q + u * Lw / 2, q + u * Lw / 2 + n * Dp, q - u * Lw / 2 + n * Dp])
        cv2.fillPoly(img, [np.round(quad + [c.m(0.4), c.m(0.6)]).astype(np.int32)], (84, 84, 84))
        cv2.fillPoly(img, [np.round(quad).astype(np.int32)], col)
        if col2:
            for s_ in np.arange(-Lw / 2, Lw / 2, c.m(0.6)):
                if int(round(s_ / c.m(0.6))) % 2:
                    st = np.array([q + u * s_, q + u * min(s_ + c.m(0.6), Lw / 2), q + u * min(s_ + c.m(0.6), Lw / 2) + n * Dp, q + u * s_ + n * Dp])
                    cv2.fillPoly(img, [np.round(st).astype(np.int32)], col2)
        cv2.line(img, tuple(np.round(quad[2]).astype(int)), tuple(np.round(quad[3]).astype(int)), tuple(int(v * 0.6) for v in col), max(1, int(c.m(0.15))))
        if t.get('amenity') in ('cafe', 'restaurant', 'bar', 'pub') and rs.random() < 0.8:
            for j in range(rs.randrange(2, 5)):                        # тераса: парасольки
                cpos = q + n * c.m(4.2) + u * (j - 1.5) * c.m(3.2)
                if not (0 <= cpos[0] < W and 0 <= cpos[1] < H) or cls[int(cpos[1]), int(cpos[0])] not in (4, 7, 13, 1): continue
                umbrella(c, cpos[0], cpos[1], UMB[rs.randrange(len(UMB))], rs)
    return placed


def umbrella(c, x, y, col, rs):
    img = c.img; r = c.m(1.3)
    cv2.circle(img, (int(x + c.m(0.5)), int(y + c.m(0.7))), int(r), (90, 90, 88), -1)
    cv2.circle(img, (int(x), int(y)), int(r), col, -1)
    light = tuple(int(min(255, v * 1.18 + 10)) for v in col)
    for k in range(8):
        if k % 2: continue
        a0 = math.radians(k * 45 + rs.random() * 5)
        pts = np.array([[x, y], [x + math.cos(a0) * r, y + math.sin(a0) * r], [x + math.cos(a0 + math.radians(45)) * r, y + math.sin(a0 + math.radians(45)) * r]])
        cv2.fillPoly(img, [np.round(pts).astype(np.int32)], light)
    cv2.circle(img, (int(x), int(y)), max(1, int(c.m(0.15))), (60, 60, 60), -1)


def markets(c, bmask):
    """Ринок: ряди наметів із гребенем (різні кольори), проходи між рядами; лише на вільному мощенні."""
    img, cls = c.img, c.cls
    H, W = cls.shape
    for item in _node_pts(c, lambda t: t.get('amenity') == 'marketplace'):
        if len(item) < 3: continue
        P = item[2]
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(P).astype(np.int32)], 1)
        m = m.astype(bool) & ~cv2.dilate(bmask.astype(np.uint8), np.ones((c.S * 3 + 1,) * 2, np.uint8)).astype(bool) & np.isin(cls, (1, 7, 13, 9, 12))
        if m.sum() < c.m(10) ** 2: continue
        (cx, cy), (rw, rh), a = cv2.minAreaRect(P.astype(np.float32))
        a = math.radians(a); ux, uy = math.cos(a), math.sin(a); nx, ny = -uy, ux
        tw, td, aisle = c.m(3.0), c.m(2.6), c.m(3.2)
        ext = max(rw, rh)
        for i in np.arange(-ext / 2, ext / 2, tw):
            for j in np.arange(-ext / 2, ext / 2, 2 * td + aisle):
                for side in (0, 1):
                    px_ = cx + ux * (i + tw / 2) + nx * (j + side * td + td / 2); py_ = cy + uy * (i + tw / 2) + ny * (j + side * td + td / 2)
                    if not (0 <= px_ < W and 0 <= py_ < H): continue
                    if not m[int(py_), int(px_)]: continue
                    rs = _rng(*c.world((px_, py_)), 31)
                    col = np.array(TENT[rs.randrange(len(TENT))], np.float32)
                    q = np.array([[-tw / 2 + 1, -td / 2 + 1], [tw / 2 - 1, -td / 2 + 1], [tw / 2 - 1, td / 2 - 1], [-tw / 2 + 1, td / 2 - 1]])
                    R = np.array([[ux, nx], [uy, ny]])
                    Q = q @ R.T + [px_, py_]
                    cv2.fillPoly(img, [np.round(Q + [c.m(0.4), c.m(0.6)]).astype(np.int32)], (90, 88, 84))
                    h1 = np.array([q[0], q[1], (q[1] + q[2]) / 2, (q[0] + q[3]) / 2]) @ R.T + [px_, py_]
                    h2 = np.array([(q[0] + q[3]) / 2, (q[1] + q[2]) / 2, q[2], q[3]]) @ R.T + [px_, py_]
                    cv2.fillPoly(img, [np.round(h1).astype(np.int32)], tuple(int(v) for v in np.clip(col * 1.1 + 8, 0, 255)))
                    cv2.fillPoly(img, [np.round(h2).astype(np.int32)], tuple(int(v) for v in col * 0.82))


def fountains(c):
    img = c.img
    for item in _node_pts(c, lambda t: t.get('amenity') == 'fountain'):
        if len(item) > 2:
            P = item[2]
            cv2.fillPoly(img, [np.round(P).astype(np.int32)], (204, 200, 192))
            inner = (P - P.mean(0)) * 0.86 + P.mean(0)
            cv2.fillPoly(img, [np.round(inner).astype(np.int32)], (86, 148, 196))
            x, y = P.mean(0); r = max(c.m(1.0), np.hypot(*(P - P.mean(0)).T).mean() * 0.2)
        else:
            x, y = item[1]; R = c.m(3.2)
            cv2.circle(img, (int(x), int(y)), int(R), (204, 200, 192), -1)
            cv2.circle(img, (int(x), int(y)), int(R - c.m(0.5)), (86, 148, 196), -1)
            r = c.m(0.9)
        cv2.circle(img, (int(x), int(y)), int(r), (232, 242, 248), -1)
        cv2.circle(img, (int(x), int(y)), int(r * 1.8), (168, 208, 232), max(1, int(c.m(0.2))))


def flowerbeds(c):
    img, S = c.img, c.S
    FL = [(226, 64, 74), (240, 200, 60), (236, 236, 236), (186, 90, 200), (240, 130, 60)]
    for item in _node_pts(c, lambda t: t.get('landuse') == 'flowerbed' or t.get('leisure') == 'garden'):
        if len(item) < 3: continue
        P = item[2]
        m = np.zeros(c.cls.shape, np.uint8); cv2.fillPoly(m, [np.round(P).astype(np.int32)], 1)
        ys, xs = np.nonzero(m)
        if len(xs) == 0: continue
        img[ys, xs] = (img[ys, xs] * 0.5 + np.array((72, 120, 52)) * 0.5).astype(np.uint8)
        rs = np.random.RandomState(int(abs(c.world((xs[0], ys[0]))[0] * 31 + c.world((xs[0], ys[0]))[1])) & 0x7FFFFFFF)
        k = rs.rand(len(xs)) < 0.05
        for x, y in zip(xs[k], ys[k]):
            cv2.circle(img, (int(x), int(y)), max(1, S // 4), FL[(x * 7 + y * 3) % len(FL)], -1)


def playgrounds(c):
    """Дитмайданчик: м'яке покриття, гірка, гойдалки, пісочниця, лазанка."""
    img, cls = c.img, c.cls
    H, W = cls.shape
    for item in _node_pts(c, lambda t: t.get('leisure') == 'playground'):
        rs = _rng(*c.world(item[1]), 41)
        if len(item) > 2:
            P = item[2]
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(P).astype(np.int32)], 1)
        else:
            m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(item[1][0]), int(item[1][1])), int(c.m(7)), 1, -1)
        ys, xs = np.nonzero(m)
        if len(xs) < c.m(3) ** 2: continue
        surf = (196, 98, 72) if rs.random() < 0.6 else (214, 196, 150)
        img[ys, xs] = surf
        cv2.polylines(img, [cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], True, (150, 150, 146), max(1, int(c.m(0.2))))
        inner = cv2.erode(m, np.ones((int(c.m(2.5)) | 1,) * 2, np.uint8))
        iy, ix = np.nonzero(inner)
        if len(ix) == 0: continue
        for kind in rs.sample(['slide', 'swing', 'sand', 'climb', 'swing2'], 3):
            j = rs.randrange(len(ix)); x, y = int(ix[j]), int(iy[j])
            if kind == 'slide':
                cv2.rectangle(img, (x, y), (x + int(c.m(1.0)), y + int(c.m(3.2))), (220, 60, 50), -1)
                cv2.rectangle(img, (x - int(c.m(0.6)), y - int(c.m(1.4))), (x + int(c.m(1.6)), y), (240, 200, 60), -1)
            elif kind in ('swing', 'swing2'):
                cv2.line(img, (x - int(c.m(1.8)), y), (x + int(c.m(1.8)), y), (60, 90, 160), max(1, int(c.m(0.25))))
                for dx in (-0.8, 0.8):
                    cv2.rectangle(img, (x + int(c.m(dx - 0.3)), y + int(c.m(0.3))), (x + int(c.m(dx + 0.3)), y + int(c.m(0.7))), (90, 60, 40), -1)
            elif kind == 'sand':
                cv2.rectangle(img, (x, y), (x + int(c.m(3)), y + int(c.m(3))), (140, 96, 60), -1)
                cv2.rectangle(img, (x + int(c.m(0.3)), y + int(c.m(0.3))), (x + int(c.m(2.7)), y + int(c.m(2.7))), (228, 210, 160), -1)
            else:
                cv2.rectangle(img, (x, y), (x + int(c.m(2.5)), y + int(c.m(2.5))), (60, 140, 90), max(1, int(c.m(0.25))))
                cv2.line(img, (x, y), (x + int(c.m(2.5)), y + int(c.m(2.5))), (60, 140, 90), max(1, int(c.m(0.25))))


def benches(c):
    img = c.img
    for item in _node_pts(c, lambda t: t.get('amenity') == 'bench'):
        if len(item) > 2: continue
        x, y = item[1]; rs = _rng(*c.world((x, y)), 51)
        a = rs.random() * math.pi
        ux, uy = math.cos(a), math.sin(a)
        L, Wd = c.m(1.8), c.m(0.55)
        _rect(img, x + c.m(0.3), y + c.m(0.4), L, Wd, ux, uy, (96, 96, 92))
        _rect(img, x, y, L, Wd, ux, uy, (140, 96, 60))
        _rect(img, x - uy * Wd * 0.3, y + ux * Wd * 0.3, L, Wd * 0.25, ux, uy, (110, 74, 46))


def shore(c):
    """Очерет біля природних берегів і кілька човнів біля берега (детерміновано)."""
    img, cls, S = c.img, c.cls, c.S
    H, W = cls.shape
    water = (cls == 3).astype(np.uint8)
    if not water.any(): return
    soft = np.isin(cls, (2, 14, 9)).astype(np.uint8)
    band = water.astype(bool) & cv2.dilate(soft, np.ones((int(c.m(3)) | 1,) * 2, np.uint8)).astype(bool)
    ys, xs = np.nonzero(band)
    rs = np.random.RandomState((int(c.x0) * 31 + int(c.y0)) & 0x7FFFFFFF)
    k = rs.rand(len(xs)) < 0.18
    for x, y in zip(xs[k], ys[k]):
        col = ((150, 156, 80), (176, 168, 96), (110, 128, 62))[(x + y) % 3]
        cv2.line(img, (int(x), int(y)), (int(x + (x % 3 - 1) * S // 2), int(y - S * 1.6)), col, max(1, S // 3))
    # човни: на воді в ~6 м від берега, рідко, за сіткою світу
    edge = water.astype(bool) & ~cv2.erode(water, np.ones((int(c.m(6)) | 1,) * 2, np.uint8)).astype(bool) & cv2.erode(water, np.ones((int(c.m(2)) | 1,) * 2, np.uint8)).astype(bool)
    g = int(c.m(120))
    for gy in range(-(int(c.y0 * S) % g), H, g):
        for gx in range(-(int(c.x0 * S) % g), W, g):
            win = edge[max(0, gy):gy + g, max(0, gx):gx + g]
            if not win.any(): continue
            wx, wy = c.world((gx, gy))
            rs2 = _rng(wx, wy, 61)
            if rs2.random() > 0.45: continue
            ey, ex = np.nonzero(win); j = rs2.randrange(len(ex)); x, y = ex[j] + max(0, gx), ey[j] + max(0, gy)
            a = rs2.random() * 180
            cv2.ellipse(img, (int(x + c.m(0.4)), int(y + c.m(0.5))), (int(c.m(2.2)), int(c.m(0.8))), a, 0, 360, (40, 70, 100), -1)
            cv2.ellipse(img, (int(x), int(y)), (int(c.m(2.2)), int(c.m(0.8))), a, 0, 360, (236, 236, 230) if rs2.random() < 0.6 else (190, 70, 50), -1)
            cv2.ellipse(img, (int(x), int(y)), (int(c.m(1.4)), int(c.m(0.45))), a, 0, 360, (150, 110, 70), -1)


def garage_seams(out, F, P, b, ss):
    """Ряд гаражів: шви між боксами (~3,5 м) поперек довгої сторони."""
    cx, cy, L, Wd, ux, uy = _orient(P)
    yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
    a = (xx - cx) * ux + (yy - cy) * uy
    seam = F & (np.mod(a, 3.5 * 2 * ss) < 0.35 * ss)
    out[seam] = (out[seam] * 0.62).astype(np.uint8)


# ---------------------------------------------------------------- група 4: широкі вулиці й парки
def main_mask(c, extra_m=12):
    """Смуга навколо широких вулиць (головні + пішохідні + площі), у детальній сітці."""
    from citydraw import ROAD_W
    S = c.S; H, W = c.cls.shape
    main = np.zeros((H // S + 1, W // S + 1), np.uint8)
    for h, ptf, w in _drive_ways(c):
        if h in MAIN: cv2.polylines(main, [np.round(np.array(ptf) / S).astype(np.int32)], False, 1, int(c.m(ROAD_W[h] + extra_m) / S))
    for w, pts, ptf in c.ways:
        t = w['t']
        if t.get('highway') == 'pedestrian' or t.get('place') == 'square':
            if w['closed']: cv2.fillPoly(main, [np.round(np.array(ptf) / S).astype(np.int32)], 1)
            else: cv2.polylines(main, [np.round(np.array(ptf) / S).astype(np.int32)], False, 1, 16)
    return np.repeat(np.repeat(main, S, 0), S, 1)[:H, :W].astype(bool)


def road_wear(c, mainf):
    """Латки асфальту й люки на широких вулицях: менше «сірого полотна»."""
    from citydraw import ROAD_W
    img, cls = c.img, c.cls
    H, W = cls.shape
    road = (cls == 5)
    for h, ptf, w in _drive_ways(c):
        if h not in MAIN: continue
        hw = c.m(ROAD_W[h]) / 2
        for i in range(len(ptf) - 1):
            (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < c.m(6): continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux
            for s_ in np.arange(c.m(4), L - c.m(4), c.m(9)):
                px_, py_ = xa + ux * s_, ya + uy * s_
                rs = _rng(*c.world((px_, py_)), 71)
                r = rs.random()
                off = (rs.random() * 2 - 1) * hw * 0.8
                qx, qy = px_ + nx * off, py_ + ny * off
                if not (0 <= qx < W and 0 <= qy < H) or not road[int(qy), int(qx)]: continue
                if r < 0.18:                                           # латка
                    Lp, Wp = c.m(1.5 + rs.random() * 3), c.m(1 + rs.random() * 1.6)
                    shade = 0.82 if rs.random() < 0.6 else 1.12
                    m = np.zeros((int(Lp + Wp) * 2 + 4,) * 2, np.uint8); k = m.shape[0] // 2
                    _rect(m, k, k, Lp, Wp, ux, uy, 1)
                    ys, xs = np.nonzero(m); ys = ys + int(qy) - k; xs = xs + int(qx) - k
                    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                    ys, xs = ys[ok], xs[ok]; ok = road[ys, xs]; ys, xs = ys[ok], xs[ok]
                    img[ys, xs] = np.clip(img[ys, xs] * shade, 0, 255).astype(np.uint8)
                elif r < 0.30:                                         # люк
                    cv2.circle(img, (int(qx), int(qy)), int(c.m(0.4)), (52, 54, 58), -1)
                    cv2.circle(img, (int(qx), int(qy)), int(c.m(0.4)), (96, 98, 102), max(1, int(c.m(0.1))))


def sidewalk_tiles(c, mainf):
    """Плитка тротуарів широких вулиць: смуги кольору (сіра/теракотова/пісочна) — по-різному на різних кварталах."""
    img, cls, S = c.img, c.cls, c.S
    H, W = cls.shape
    walk = np.isin(cls, (4, 7)) & mainf
    if not walk.any(): return
    yy, xx = np.mgrid[0:H, 0:W]
    gx = xx // S + int(c.x0); gy = yy // S + int(c.y0)
    zone = ((gx // 160) * 7 + (gy // 160) * 13) % 4                  # квартал ~80 м — свій візерунок
    PAL = [((168, 108, 94), (176, 172, 164)), ((196, 178, 140), (168, 166, 160)), ((142, 144, 150), (182, 180, 174)), ((160, 96, 82), (196, 178, 140))]
    out = img.astype(np.float32)
    for z, (a, b) in enumerate(PAL):
        m = walk & (zone == z)
        if not m.any(): continue
        band = ((gx + gy * 0) // 3 + (gy // 3)) % 4 == 0 if z % 2 else ((gx // 2 + gy // 2) % 3 == 0)
        col = np.where(band[..., None], np.array(a, np.float32), np.array(b, np.float32))
        out[m] = out[m] * 0.45 + col[m] * 0.55
    img[:] = np.clip(out, 0, 255).astype(np.uint8)


def park_furniture(c, park_b):
    """Парки: лавки вздовж доріжок (по черзі з боків), невеликі ліхтарі, урни; клумба на перетині доріжок."""
    img, S = c.img, c.S
    H, W = c.cls.shape
    pk = np.repeat(np.repeat(cv2.dilate(park_b.astype(np.uint8), np.ones((9, 9), np.uint8)), S, 0), S, 1)[:H, :W].astype(bool)
    if not pk.any(): return
    cnt = {}
    for w, pts, ptf in c.ways:
        t = w['t']
        if t.get('highway') not in ('footway', 'path', 'pedestrian', 'cycleway'): continue
        for nid in w['nds']: cnt[nid] = cnt.get(nid, 0) + 1
        for i in range(len(ptf) - 1):
            (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < c.m(5): continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux
            for s_ in np.arange(c.m(6), L - c.m(3), c.m(13)):
                px_, py_ = xa + ux * s_, ya + uy * s_
                if not (0 <= px_ < W and 0 <= py_ < H) or not pk[int(py_), int(px_)]: continue
                rs = _rng(*c.world((px_, py_)), 81)
                sg = 1 if rs.random() < 0.5 else -1
                r = rs.random()
                bx, by = px_ + nx * sg * c.m(1.9), py_ + ny * sg * c.m(1.9)
                if r < 0.45:
                    _rect(img, bx + c.m(0.3), by + c.m(0.4), c.m(1.8), c.m(0.6), ux, uy, (90, 96, 84))
                    _rect(img, bx, by, c.m(1.8), c.m(0.6), ux, uy, (146, 102, 64))
                    _rect(img, bx + nx * sg * c.m(0.22), by + ny * sg * c.m(0.22), c.m(1.8), c.m(0.15), ux, uy, (112, 76, 46))
                    if rs.random() < 0.4: bin_(c, bx + ux * c.m(1.5), by + uy * c.m(1.5))
                elif r < 0.65:
                    cv2.circle(img, (int(bx), int(by)), max(1, int(c.m(0.25))), POLE, -1)
                    cv2.circle(img, (int(bx), int(by)), max(1, int(c.m(0.45))), (226, 226, 210), max(1, int(c.m(0.12))))
    # клумби на перетинах доріжок у парках
    FL = [(220, 60, 70), (240, 200, 60), (236, 236, 236), (186, 90, 200), (240, 130, 60)]
    for nid, n in cnt.items():
        if n < 3 or nid not in c.D['nodes']: continue
        x, y = c.tofn(nid)
        if not (0 <= x < W and 0 <= y < H) or not pk[int(y), int(x)]: continue
        rs = _rng(*c.world((x, y)), 91)
        if rs.random() < 0.5: continue
        R = c.m(2.2 + rs.random() * 1.5)
        cv2.circle(img, (int(x), int(y)), int(R + c.m(0.3)), (190, 186, 178), -1)
        cv2.circle(img, (int(x), int(y)), int(R), (70, 112, 48), -1)
        for k in range(int(R * R / (S * S) * 0.6)):
            a = rs.random() * 6.283; d = math.sqrt(rs.random()) * R * 0.92
            cv2.circle(img, (int(x + math.cos(a) * d), int(y + math.sin(a) * d)), max(1, S // 3), FL[rs.randrange(len(FL))], -1)


def kiosks(c, solid_out=None):
    img = c.img
    for item in _node_pts(c, lambda t: t.get('shop') == 'kiosk' or t.get('amenity') == 'vending_machine' and False):
        if len(item) > 2: continue
        x, y = item[1]; rs = _rng(*c.world((x, y)), 101)
        a = rs.random() * math.pi; ux, uy = math.cos(a), math.sin(a)
        col = [(226, 226, 220), (60, 120, 180), (196, 64, 52), (80, 140, 80)][rs.randrange(4)]
        _rect(img, x + c.m(0.5), y + c.m(0.7), c.m(3.0), c.m(2.4), ux, uy, (88, 88, 86))
        _rect(img, x, y, c.m(3.0), c.m(2.4), ux, uy, col)
        _rect(img, x, y, c.m(2.4), c.m(1.8), ux, uy, tuple(int(v * 0.86) for v in col))


def planters(c):
    """Вазони з квітами вздовж пішохідних вулиць."""
    img = c.img; H, W = c.cls.shape
    FL = [(220, 60, 70), (240, 200, 60), (236, 236, 236), (186, 90, 200)]
    for w, pts, ptf in c.ways:
        if w['t'].get('highway') != 'pedestrian' or w['closed']: continue
        for i in range(len(ptf) - 1):
            (xa, ya), (xb, yb) = ptf[i], ptf[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < c.m(8): continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; nx, ny = -uy, ux
            for s_ in np.arange(c.m(5), L - c.m(3), c.m(16)):
                for sg in (-1, 1):
                    x, y = xa + ux * s_ + nx * sg * c.m(3.2), ya + uy * s_ + ny * sg * c.m(3.2)
                    if not (0 <= x < W and 0 <= y < H) or c.cls[int(y), int(x)] not in (4, 7, 13): continue
                    rs = _rng(*c.world((x, y)), 111)
                    cv2.circle(img, (int(x + c.m(0.25)), int(y + c.m(0.3))), int(c.m(0.8)), (96, 96, 92), -1)
                    cv2.circle(img, (int(x), int(y)), int(c.m(0.8)), (150, 146, 140), -1)
                    cv2.circle(img, (int(x), int(y)), int(c.m(0.62)), (66, 104, 46), -1)
                    for k in range(6):
                        a = rs.random() * 6.283; d = rs.random() * c.m(0.5)
                        cv2.circle(img, (int(x + math.cos(a) * d), int(y + math.sin(a) * d)), max(1, c.S // 3), FL[rs.randrange(4)], -1)
