"""Латки поверх готової карти (після citybuild, до bake): унікальні дахи пам'яток, пожежна вежа, трибуни стадіону,
білий міст і колесо огляду в парку Чекмана, скульптури Миколи Мазура в сквері Шевченка, тверді кіоски речового ринку.
Малює з надвибіркою (SS=4) і змішує з картою за покриттям; повторний запуск дає той самий результат.
python3 -I tools/mappatch.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import citydraw as CD, citygen as CG

K = CD.K
SS = 4
MPP = CD.MPP                                     # м на піксель карти (з 31.10.2026 ≈0,278)
LV = np.array([-0.5, -0.5, 0.7]); LV /= np.linalg.norm(LV)     # світло з північного заходу (тіні на південний схід)
GOLD = (226, 182, 64)


def m(x): return x / MPP                          # метри -> пікселі карти
F = 0.5 / MPP                                    # старі пікселі (0,5 м; усі ручні координати нижче) -> пікселі карти
def O(*v): return tuple(a * F for a in v)


class Canvas:
    """Ділянка карти з надвибіркою: колір (премультиплікований) + покриття."""
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = int(math.floor(x0)), int(math.floor(y0))
        self.w, self.h = int(math.ceil(x1)) - self.x0, int(math.ceil(y1)) - self.y0
        self.C = np.zeros((self.h * SS, self.w * SS, 3), np.float32)
        self.A = np.zeros((self.h * SS, self.w * SS), np.float32)

    def P(self, pts):                             # світові пікс -> пікс полотна (фіксована кома для fillPoly, shift=4)
        return np.round((np.asarray(pts, np.float64) - [self.x0, self.y0]) * SS * 16).astype(np.int32)

    def mask(self, pts):
        mk = np.zeros(self.A.shape, np.uint8)
        cv2.fillPoly(mk, [self.P(pts)], 1, shift=4)
        return mk

    def grid(self):
        yy, xx = np.mgrid[0:self.h * SS, 0:self.w * SS].astype(np.float32)
        return (xx + 0.5) / SS + self.x0, (yy + 0.5) / SS + self.y0

    def put(self, sel, col, a=1.0):
        """sel — булева маска; col — колір (3,) або (N,3) для пікселів маски"""
        col = np.asarray(col, np.float32)
        self.C[sel] = self.C[sel] * (1 - a) + col * a
        self.A[sel] = self.A[sel] * (1 - a) + a

    def commit(self, rgb):
        """змішати з картою (rgb — memmap (H, W, 3))"""
        A = cv2.resize(self.A, (self.w, self.h), interpolation=cv2.INTER_AREA)
        C = cv2.resize(self.C * self.A[..., None], (self.w, self.h), interpolation=cv2.INTER_AREA)
        sl = (slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w))
        base = np.asarray(rgb[sl]).astype(np.float32)
        out = base * (1 - A[..., None]) + C
        rgb[sl] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return A


def hip_roof(cv, pts, col, pitch_m=4.0, steep=1.0, edge=0.78):
    """Дах за формою контуру: висота = відстань до краю (до pitch_m), затінення за нормаллю — коньки й скати самі."""
    mk = cv.mask(pts)
    if not mk.any(): return mk.astype(bool)
    dist = cv2.distanceTransform(mk, cv2.DIST_L2, 5) / SS          # пікс карти
    hgt = np.minimum(dist, m(pitch_m)) * steep
    hgt = cv2.GaussianBlur(hgt, (0, 0), 0.6 * SS / 2)
    gy, gx = np.gradient(hgt * SS)
    n = np.dstack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    lam = n @ LV
    f = np.clip(lam / LV[2], 0.25, 1.45)
    c = np.asarray(col, np.float32)[None, None, :] * (0.5 + 0.5 * f)[..., None]
    sel = mk.astype(bool)
    rim = sel & ~cv2.erode(mk, np.ones((int(SS * 1.6),) * 2, np.uint8)).astype(bool)
    c[rim] *= edge
    cv.put(sel, c[sel])
    return sel


def dome(cv, cx, cy, r_m, col, cross=True, onion=True):
    """Баня згори: куля з відблиском; цибулина — зі світлішою «шийкою» в центрі; хрест."""
    X, Y = cv.grid()
    R = m(r_m)
    dx, dy = X - cx, Y - cy
    d2 = dx * dx + dy * dy
    sel = d2 <= R * R
    if not sel.any(): return
    z = np.sqrt(np.maximum(R * R - d2, 0))
    n = np.dstack([dx, dy, z]) / R
    lam = np.clip(n @ LV, 0, 1)
    c = np.asarray(col, np.float32)[None, None, :] * (0.3 + 0.95 * lam)[..., None]
    c += 255 * 0.55 * (lam ** 14)[..., None]                       # відблиск
    if onion:                                                      # гострий верх цибулини — темніше кільце й світла цятка
        rr = np.sqrt(d2) / R
        c *= (1 - 0.18 * np.exp(-((rr - 0.62) / 0.12) ** 2))[..., None]
    edge = sel & (d2 > (R - 0.6 * F) ** 2)
    c[edge] *= 0.6
    cv.put(sel, np.clip(c[sel], 0, 255))
    if cross:
        s = max(R * 0.55, 1.6 * F)
        for (ax, ay, bx, by) in ((cx - s * 0.6, cy, cx + s * 0.6, cy), (cx, cy - s, cx, cy + s)):
            mk = np.zeros(cv.A.shape, np.uint8)
            cv2.line(mk, tuple(cv.P([(ax, ay)])[0]), tuple(cv.P([(bx, by)])[0]), 1, max(1, int(SS * 0.7)), cv2.LINE_8, 4)
            cv.put(mk.astype(bool), GOLD)


def tower(cv, cx, cy, side_m, wall, roof, ang=0.0, spire=True):
    """Квадратна вежа згори: стіни-парапет по краю, всередині шатровий дах (піраміда) і шпиль."""
    s = m(side_m) / 2
    ca, sa = math.cos(ang), math.sin(ang)
    sq = [(cx + (ux * ca - uy * sa), cy + (ux * sa + uy * ca)) for ux, uy in ((-s, -s), (s, -s), (s, s), (-s, s))]
    sel = hip_roof(cv, sq, roof, pitch_m=side_m, steep=1.6, edge=1.0)
    mk = cv.mask(sq)
    ring = mk.astype(bool) & ~cv2.erode(mk, np.ones((int(SS * 2.2),) * 2, np.uint8)).astype(bool)
    X, Y = cv.grid()
    lit = ((X - cx) + (Y - cy)) < 0
    cv.put(ring & lit, np.asarray(wall, np.float32) * 1.08)
    cv.put(ring & ~lit, np.asarray(wall, np.float32) * 0.8)
    if spire:
        sp = (X - cx) ** 2 + (Y - cy) ** 2 < (m(0.5)) ** 2
        cv.put(sp, GOLD)


def shadow(rgb, pts, h_m, strength=0.38):
    """Тінь високого об'єкта на південний схід (як у карті)."""
    P = np.asarray(pts, np.float32)
    k = min(14.0, h_m * 0.22 * 1.6) * F
    x0, y0 = P.min(0) - 2; x1, y1 = P.max(0) + k + 3
    cv = Canvas(x0, y0, x1, y1)
    mk = np.zeros(cv.A.shape, np.uint8)
    for t in np.linspace(0, k, int(k * 2) + 2):
        cv2.fillPoly(mk, [cv.P(P + t)], 1, shift=4)
    mk[cv.mask(P).astype(bool)] = 0
    a = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), SS * 0.8) * strength
    A = cv2.resize(a, (cv.w, cv.h), interpolation=cv2.INTER_AREA)
    sl = (slice(cv.y0, cv.y0 + cv.h), slice(cv.x0, cv.x0 + cv.w))
    base = np.asarray(rgb[sl]).astype(np.float32)
    rgb[sl] = np.clip(base * (1 - A[..., None]), 0, 255).astype(np.uint8)


def sat_color(sat, pts, default):
    """медіанний колір даху зі знімка (без найтемніших тіней), трохи світліший — як інші дахи карти"""
    if sat is None: return default
    f = MPP / 0.32
    P = np.asarray(pts, np.float32) * f
    x0, y0 = np.floor(P.min(0)).astype(int); x1, y1 = np.ceil(P.max(0)).astype(int) + 1
    if x0 < 0 or y0 < 0: return default
    s = np.asarray(sat[y0:y1, x0:x1]).astype(np.float32)
    mk = np.zeros(s.shape[:2], np.uint8); cv2.fillPoly(mk, [np.round(P - [x0, y0]).astype(np.int32)], 1)
    px = s[mk.astype(bool)]
    if len(px) < 10: return default
    lum = px.mean(1); px = px[lum > np.percentile(lum, 40)]
    c = np.median(px, 0)
    c = c * (150 / max(40, c.mean())) ** 0.6
    return tuple(np.clip(c, 40, 235).astype(int))


def map_color(rgb, pts, default):
    """медіанний колір даху з самої карти (кольори карти вже підібрані зі знімків і узгоджені між собою)"""
    P = np.asarray(pts, np.float32)
    x0, y0 = np.floor(P.min(0)).astype(int); x1, y1 = np.ceil(P.max(0)).astype(int) + 1
    s = np.asarray(rgb[y0:y1, x0:x1]).astype(np.float32)
    mk = np.zeros(s.shape[:2], np.uint8); cv2.fillPoly(mk, [np.round(P - [x0, y0]).astype(np.int32)], 1)
    mk = cv2.erode(mk, np.ones((int(5 * F) | 1,) * 2, np.uint8))
    px = s[mk.astype(bool)]
    if len(px) < 10: return default
    return tuple(int(v) for v in np.median(px, 0))


def find_poly(D, x, y, src=('B', 'SAT')):
    best = None
    for s in src:
        for b in D[s]:
            Q = np.array(b['q'], np.float32) * K
            if abs(Q[:, 0].mean() - x) > 300 * F or abs(Q[:, 1].mean() - y) > 300 * F: continue
            d = cv2.pointPolygonTest(Q, (float(x), float(y)), True)
            if best is None or d > best[0]: best = (d, Q)
    return best[1]


def rect_frame(P):
    """центр, одиничні осі (уздовж довшої сторони, поперек), довжина, ширина"""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(P, np.float32))
    a = math.radians(a)
    u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
    if w < h: u, v, w, h = v, -u, h, w
    return np.array([cx, cy]), u, v, w, h


# ---------------------------------------------------------------- пам'ятки: дахи
CHURCHES = {
    # за сирими знімками Івана (ref/khn): колір даху, колір бань, бані — абсолютні точки карти (x, y, радіус м)
    'pokrov':   ((5749, 8694), (112, 192, 170), (96, 176, 152), [(5747, 8692, 3.4), (5728, 8686, 1.8), (5766, 8701, 1.8)]),
    'rizdva':   ((6779, 7049), (78, 126, 206), (96, 150, 228), [(6774, 7048, 6.0)]),
    'ugcc':     ((10328, 5899), (226, 226, 220), (64, 76, 104), [(10326, 5893, 5.2)]),
    'mykol':    ((9600, 4840), (176, 170, 156), (52, 96, 196), [(9598, 4802, 3.6), (9584, 4824, 3.6), (9620, 4813, 3.2), (9612, 4834, 3.0)]),
    'triitska': ((9205, 6362), (52, 88, 196), (84, 124, 220), [(9235, 6335, 2.6)]),
}
GOLDMAIN = {'mykol': (9606, 4819, 3.0)}         # центральна золота баня
PLAIN = {      # будівлі без бань: колір зі знімка, скати за контуром
    'arkhiv': ((6349, 8535), 3.5), 'vokzal': ((10955, 9805), 5.0),
}


def landmark_roofs(D, rgb, sat):
    for k, (pt, roofc, domec, domes) in CHURCHES.items():
        P = find_poly(D, *O(*pt), src=('B',))
        c, u, v, L, W = rect_frame(P)
        cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
        hip_roof(cv, P, roofc, pitch_m=min(6.0, W * MPP * 0.45), steep=1.2)
        for x, y, r in domes:
            dome(cv, x * F, y * F, r, domec)
        if k in GOLDMAIN: g = GOLDMAIN[k]; dome(cv, g[0] * F, g[1] * F, g[2], (226, 182, 64))
        cv.commit(rgb)
        print('  дах', k)
    # костел: хрестоподібний сірий дах + вежа з гострим шпилем на південному кінці
    P = find_poly(D, *O(8027, 3710), src=('B',))
    c, u, v, L, W = rect_frame(P)
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    hip_roof(cv, P, (96, 100, 110), pitch_m=6.0, steep=1.3)
    end = c + u * L * 0.4 * (1 if u[1] > 0 else -1)
    end = c + (end - c) * 0.9
    tower(cv, *end, 8.0, (196, 198, 204), (70, 76, 90), ang=math.atan2(u[1], u[0]))
    cv.commit(rgb); print('  дах kostel')
    for k, (pt, pitch) in PLAIN.items():
        P = find_poly(D, *O(*pt), src=('B',))
        col = map_color(rgb, P, (180, 180, 184))
        cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
        hip_roof(cv, P, col, pitch_m=pitch, steep=1.0)
        cv.commit(rgb); print('  дах', k, col)
    # стара пожежна вежа (каланча): будівля + квадратна вежа над вузлом OSM з довгою тінню
    P = find_poly(D, *O(6773, 7489), src=('B',))
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    hip_roof(cv, P, (196, 82, 62), pitch_m=4.0)                   # червоний дах (сирий знімок)
    cv.commit(rgb)
    # вежа — в куті будівлі (найближчому до точки OSM), сторони вздовж осей будівлі, зовнішній кут = кут будівлі
    side = m(8.0); hs = side / 2
    nx_, ny_ = O(6772, 7489)
    _, ue, ve, _, _ = rect_frame(P)
    cn = P[np.argmin(((P - [nx_, ny_]) ** 2).sum(1))]
    best = None
    for su in (-1, 1):
        for sv in (-1, 1):
            cc = cn + ue * su * hs + ve * sv * hs
            d_ = cv2.pointPolygonTest(P.astype(np.float32), (float(cc[0]), float(cc[1])), True)
            if best is None or d_ > best[0]: best = (d_, cc)
    tx, ty = best[1]
    ang = math.atan2(ue[1], ue[0])
    sq = [tuple(np.array([tx, ty]) + ue * a_ * hs + ve * b_ * hs) for a_, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    shadow(rgb, sq, 30)
    cv = Canvas(tx - 14 * F, ty - 14 * F, tx + 14 * F, ty + 14 * F)
    tower(cv, tx, ty, 8.0, (232, 196, 92), (120, 64, 50), ang=ang)
    cv.commit(rgb); print('  пожежна вежа')
    # трибуни стадіону «Поділля»: ряди сидінь уздовж поля + дах-козирок
    P = find_poly(D, *O(8742, 8930), src=('B',))
    c, u, v, L, W = rect_frame(P)
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    sel = cv.mask(P).astype(bool)
    X, Y = cv.grid()
    # трибуна під синім напівпрозорим дахом-навісом (сирий знімок): панелі з ребрами
    al = (X - c[0]) * u[0] + (Y - c[1]) * u[1]
    ac = (X - c[0]) * v[0] + (Y - c[1]) * v[1]
    col = np.dstack([np.full(X.shape, 56.0), np.full(X.shape, 88.0), np.full(X.shape, 160.0)])
    col *= (0.92 + 0.12 * (ac / (W / 2)))[..., None]
    cv.put(sel, col[sel])
    ribs = sel & (np.mod(al, m(9)) < m(0.5))
    cv.put(ribs, (128, 150, 196))
    edge = sel & ~cv2.erode(sel.astype(np.uint8), np.ones((SS * 2,) * 2, np.uint8)).astype(bool)
    cv.put(edge, (40, 58, 110))
    cv.commit(rgb); print('  трибуни')


# ---------------------------------------------------------------- дахи з фото (Apple Карти від Івана)
def photo_roofs(D, rgb):
    """Фото згори (Apple Карти) — лише зразок: прив'язується до контуру двома точками (масштаб + поворот + зсув),
    написи прибираються, далі дах розкладається на області (плоскі рівні, світлі парапети/ліхтарі, синій дах)
    і малюється кольорами й світлом гри — без фотофактури."""
    cfg = json.load(open('data/photo_roofs.json'))
    for k, e in cfg.items():
        if k.startswith('_') or not os.path.exists(e['img']): continue
        img = cv2.imread(e['img'])
        F = img.shape[1] / e['scr_w']
        # написи: жовтий/рожевий текст і круглі значки + темна обвідка навколо
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yel = (hsv[..., 1] > 90) & (hsv[..., 2] > 150) & (hsv[..., 0] > 12) & (hsv[..., 0] < 40)
        pink = (hsv[..., 1] > 35) & (hsv[..., 2] > 140) & (hsv[..., 0] > 135)
        lab = (yel | pink).astype(np.uint8)
        near = cv2.dilate(lab, np.ones((41, 41), np.uint8)).astype(bool)
        lab |= (near & (img.min(-1) > 200)).astype(np.uint8)                           # білі/світлі літери поруч
        lab = cv2.dilate(lab, np.ones((15, 15), np.uint8))                              # разом із темною обвідкою
        img = cv2.inpaint(img, lab, 9, cv2.INPAINT_TELEA)
        img = cv2.bilateralFilter(img, 7, 30, 5)[..., ::-1].astype(np.float32)
        a, b_ = complex(*e['scr'][0]) * F, complex(*e['scr'][1]) * F
        A, B = complex(*e['map'][0]) * F, complex(*e['map'][1]) * F
        kk = (B - A) / (b_ - a); tt = A - kk * a
        P = find_poly(D, *O(*e['at']), src=('B',))
        cv = Canvas(*(P.min(0) - 2), *(P.max(0) + 2))
        X, Y = cv.grid()
        z = (X + 1j * Y - tt) / kk
        tex = cv2.remap(img, z.real.astype(np.float32), z.imag.astype(np.float32), cv2.INTER_LINEAR)
        sel = cv.mask(P).astype(bool)
        # фото — лише зразок: розкласти дах на області й намалювати їх кольорами гри (плоскі дахи, світлі парапети/ліхтарі, синій дах)
        tex8 = np.clip(tex, 0, 255).astype(np.uint8)
        sm = cv2.medianBlur(tex8, 2 * (SS // 2) + 3)
        hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV)
        lum = sm.astype(np.float32).mean(-1)
        med = np.median(lum[sel])
        blue = sel & (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 70) & (hsv[..., 2] > 70)
        light = sel & ~blue & (lum > med + 40)
        dark = sel & ~blue & ~light & (cv2.GaussianBlur(lum, (0, 0), SS * 1.5) < med - 10)
        def clean(mk, amin_m2, eps_px=1.0, close=3):
            """області -> рівні багатокутники (архітектурні прямі краї), без дрібних плям"""
            mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close * SS // 2 + 1,) * 2, np.uint8))
            mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((SS + 1,) * 2, np.uint8))
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_ = np.zeros_like(mk)
            for c_ in cs:
                if cv2.contourArea(c_) < amin_m2 / (MPP * MPP) * SS * SS: continue
                cv2.fillPoly(out_, [cv2.approxPolyDP(c_, eps_px * SS, True)], 1)
            return out_.astype(bool) & sel
        blue = clean(blue, 300, 1.5, 5)                              # лише великий синій дах (Multiplex), не скло вежі
        light = clean(light, 3, 0.8, 2) & ~blue
        dark = np.zeros_like(sel)                                    # темні плями на фото — тіні, а не рівні даху: не беремо
        X_, Y_ = X, Y
        out = np.zeros(tex.shape, np.float32); out[:] = (156, 158, 164)
        out[dark] = (134, 136, 142)
        grad = (0.97 + 0.06 * np.sin((X_ * 0.9 + Y_ * 0.4)))[..., None]            # ледь помітні листи покрівлі синього даху
        out[blue] = (np.array((66, 100, 186), np.float32) * grad)[blue]
        out[light] = (230, 232, 234)
        # тінь від світлих виступів і синього даху на південний схід + світла кромка з північного заходу
        raised = (light | blue).astype(np.uint8)
        d = int(SS * 0.9)
        sh = np.zeros_like(raised); sh[d:, d:] = raised[:-d, :-d]
        shm = sel & (sh > 0) & ~raised.astype(bool)
        out[shm] *= 0.78
        # парапет по контуру будинку: світла лінія всередині, тонка темна зовні — як у решти дахів гри
        er = cv2.erode(sel.astype(np.uint8), np.ones((int(SS * 1.6),) * 2, np.uint8)).astype(bool)
        rim = sel & ~er
        out[rim] = (214, 216, 218)
        rim2 = er & ~cv2.erode(er.astype(np.uint8), np.ones((SS,) * 2, np.uint8)).astype(bool)
        out[rim2] *= 0.85
        tex = out
        cv.put(sel, tex[sel])
        # скляні ліхтарі-піраміди (з фото: прямокутники в пікселях знімка) — шатрові дахи зі скла
        def to_map(x, y): z = kk * complex(x * F, y * F) + tt; return (z.real, z.imag)
        for x0_, y0_, x1_, y1_ in e.get('glass', []):
            q = [to_map(x0_, y0_), to_map(x1_, y0_), to_map(x1_, y1_), to_map(x0_, y1_)]
            hip_roof(cv, q, (196, 220, 232), pitch_m=6, steep=0.9, edge=0.7)
        for cx_, cy_, r_ in e.get('glass_diamond', []):
            q = [to_map(cx_, cy_ - r_), to_map(cx_ + r_, cy_), to_map(cx_, cy_ + r_), to_map(cx_ - r_, cy_)]
            hip_roof(cv, q, (196, 220, 232), pitch_m=6, steep=0.9, edge=0.7)
        cv.commit(rgb); print('  дах з фото', k)


# ---------------------------------------------------------------- ТРЦ «Монблан»: вручну за знімком згори (ref/photos/monblan_top.jpg)
def monblan(D, rgb):
    """Подіум ТРЦ — плаский сірий дах із білими парапетами, відкритий паркінг на даху, скляний ліхтар-«стадіон»,
    надбудови; висотна вежа з білим дахом, скляною ротондою на захід і довгою тінню на дах подіуму."""
    e = json.load(open('data/photo_roofs.json'))['_monblan']
    a, b_ = complex(*e['scr'][0]), complex(*e['scr'][1]); A, B = complex(*e['map'][0]) * F, complex(*e['map'][1]) * F
    kk = (B - A) / (b_ - a); tt = A - kk * a
    def G(pts): return [((kk * complex(x, y) + tt).real, (kk * complex(x, y) + tt).imag) for x, y in pts]
    P = find_poly(D, *O(6885, 5875), src=('B',))
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    sel = cv.mask(P).astype(bool)
    X, Y = cv.grid()
    ROOF, CONC, WHITE, GLASS = (150, 152, 158), (176, 176, 174), (232, 234, 236), (176, 204, 220)
    cv.put(sel, ROOF)
    def poly(pts, col, outline=None, w=0.5):
        mk = cv.mask(G(pts)).astype(bool) & sel
        cv.put(mk, col)
        if outline is not None:
            ed = mk & ~cv2.erode(mk.astype(np.uint8), np.ones((max(2, int(SS * w * 2)),) * 2, np.uint8)).astype(bool)
            cv.put(ed, outline)
        return mk
    def line(pts, col, w=0.5):
        mk = np.zeros(cv.A.shape, np.uint8)
        cv2.polylines(mk, [cv.P(G(pts))], False, 1, max(1, int(SS * w * 2)), cv2.LINE_8, 4)
        cv.put(mk.astype(bool) & sel, col)
    # відкритий паркінг на даху (світлий бетон, розмітка місць)
    park = [(535, 450), (755, 555), (790, 820), (700, 990), (640, 1060), (655, 860), (640, 760), (600, 700), (545, 690)]
    pk = poly(park, CONC, WHITE)
    stripes = pk & (np.mod((X - Y * 0.3), m(2.6)) < m(0.25))
    cv.put(stripes, (214, 214, 210))
    # прибудова на північному сході (окремий рівень даху)
    poly([(900, 790), (1100, 790), (1140, 840), (1050, 1060), (900, 990)], (162, 164, 170), WHITE)
    poly([(905, 935), (1010, 990), (975, 1060), (900, 1030)], (142, 144, 150), WHITE)
    # надбудови на даху (вентиляційні будки)
    for bx in ([(712, 342), (800, 342), (800, 435), (712, 435)], [(668, 320), (712, 320), (712, 410), (668, 410)], [(790, 405), (835, 405), (835, 445), (790, 445)]):
        poly(bx, (188, 190, 194), WHITE)
    # скляний ліхтар-«стадіон»
    Lc = G([(930, 708)])[0]; Lr = G([(930 + 45, 708)])[0]; La = G([(930, 708 - 100)])[0]
    rr = math.hypot(Lr[0] - Lc[0], Lr[1] - Lc[1]); hl_ = math.hypot(La[0] - Lc[0], La[1] - Lc[1]) - rr
    ux, uy = (La[0] - Lc[0]), (La[1] - Lc[1]); n_ = math.hypot(ux, uy); ux, uy = ux / n_, uy / n_
    al = (X - Lc[0]) * ux + (Y - Lc[1]) * uy; ac = -(X - Lc[0]) * uy + (Y - Lc[1]) * ux
    d = np.sqrt(np.maximum(np.abs(al) - hl_, 0) ** 2 + ac ** 2)
    sk = d < rr
    shade = np.clip(1.0 - 0.35 * (ac / rr) - 0.15 * np.sign(al) * np.maximum(np.abs(al) - hl_, 0) / rr, 0.6, 1.25)
    cv.put(sk, (np.array(GLASS, np.float32)[None, None, :] * shade[..., None])[sk])
    cv.put(sk & (np.abs(ac) < m(0.25)), (240, 244, 246))
    cv.put(sk & (d > rr - m(0.4)), WHITE)
    # парапети між секціями
    line([(870, 440), (760, 555)], WHITE); line([(790, 820), (900, 930)], WHITE); line([(640, 1060), (630, 1160), (880, 1250)], WHITE)
    # вежа: тінь на подіум (на південний схід, як у всьому місті), білий дах, скляна ротонда на захід, надбудови
    tower = [(470, 740), (520, 702), (600, 700), (640, 760), (655, 850), (650, 980), (625, 1060), (560, 1100), (500, 1088), (455, 1010), (440, 900), (450, 800)]
    tq = G(tower)
    cv.commit(rgb)
    shadow(rgb, tq, 60, 0.45)
    cv = Canvas(*(np.min(tq, 0) - 4), *(np.max(tq, 0) + 4))
    hip_roof(cv, tq, (206, 208, 210), pitch_m=1.5, steep=0.6, edge=1.0)
    X, Y = cv.grid()
    mk = cv.mask(tq).astype(bool)
    ring = mk & ~cv2.erode(mk.astype(np.uint8), np.ones((int(SS * 1.6),) * 2, np.uint8)).astype(bool)
    cv.put(ring, (246, 246, 244))
    rot_c = G([(470, 900)])[0]; rot_r = abs(kk) * 120
    rot = mk & ((X - rot_c[0]) ** 2 + (Y - rot_c[1]) ** 2 < rot_r ** 2) & (X < rot_c[0] + abs(kk) * 30)
    cv.put(rot, (120, 156, 186))
    cv.put(rot & (((X - rot_c[0]) ** 2 + (Y - rot_c[1]) ** 2) > (rot_r - m(0.5)) ** 2), (200, 220, 236))
    for bx in ([(482, 800), (540, 800), (540, 950), (482, 950)], [(548, 830), (632, 830), (632, 950), (548, 950)]):
        b_m = cv.mask(G(bx)).astype(bool) & mk
        cv.put(b_m, (226, 226, 222))
        cv.put(b_m & ~cv2.erode(b_m.astype(np.uint8), np.ones((SS + 1,) * 2, np.uint8)).astype(bool), (170, 172, 176))
    cv.commit(rgb)
    print('  Монблан (вручну)')


# ---------------------------------------------------------------- парк Чекмана: білий міст, колесо огляду
def chekman(D, rgb):
    for w in D['w']:
        t = w['t']
        P = np.array(w['pts']) * K
        if t.get('man_made') == 'bridge' and t.get('bridge:structure') == 'humpback' and abs(P[:, 0].mean() - 5465 * F) < 40 * F:
            break
    else:
        print('  !! білий міст не знайдено'); return
    c, u, v, L, W = rect_frame(P)
    W2 = m(1.3); L2 = L / 2                                            # пішохідний місток ≈3 м завширшки
    deck = [c - u * L2 - v * W2, c + u * L2 - v * W2, c + u * L2 + v * W2, c - u * L2 + v * W2]
    shadow(rgb, deck, 2, 0.3)
    cv = Canvas(*(np.min(deck, 0) - 3), *(np.max(deck, 0) + 3))
    sel = cv.mask(deck).astype(bool)
    X, Y = cv.grid()
    al = ((X - c[0]) * u[0] + (Y - c[1]) * u[1]) / L2                 # -1..1 уздовж
    ac = ((X - c[0]) * v[0] + (Y - c[1]) * v[1]) / W2                 # -1..1 поперек
    hump = 1 - al ** 2                                                  # горб: середина світліша
    col = np.array((222, 220, 214), np.float32)[None, None, :] * (0.9 + 0.1 * hump)[..., None]
    cv.put(sel, col[sel])
    rail = sel & (np.abs(ac) > 0.78)
    cv.put(rail, (240, 240, 236))
    rail_sh = sel & (np.abs(ac) > 0.7) & (np.abs(ac) <= 0.78)
    cv.put(rail_sh, (186, 188, 190))
    posts = rail & (np.mod((al + 1) * L2 / m(1.6), 1) < 0.22)
    cv.put(posts, (150, 152, 156))
    cv.commit(rgb); print('  білий міст')
    # колесо огляду: згори — вузький обід із синіми кабінками, опори, довга тінь на південний схід
    x, y = O(4278, 6643)
    Rw = m(16)
    ax = np.array([1.0, 0.0])
    rim = [(x - Rw, y - m(1.2)), (x + Rw, y - m(1.2)), (x + Rw, y + m(1.2)), (x - Rw, y + m(1.2))]
    shadow(rgb, rim, 34, 0.32)
    cv = Canvas(x - Rw - 6 * F, y - 10 * F, x + Rw + 6 * F, y + 10 * F)
    X, Y = cv.grid()
    base = (np.abs(X - x) < m(5)) & (np.abs(Y - y) < m(4))
    cv.put(base, (170, 172, 176))
    for sy in (-1, 1):
        leg = (np.abs(Y - (y + sy * m(2.6))) < m(0.35)) & (np.abs(X - x) < m(6))
        cv.put(leg, (230, 232, 236))
    rimm = (np.abs(X - x) < Rw) & (np.abs(Y - y) < m(0.5))
    cv.put(rimm, (240, 242, 246))
    hub = (X - x) ** 2 + (Y - y) ** 2 < m(1.0) ** 2
    cv.put(hub, (120, 124, 132))
    for k in range(-8, 9):
        cx_ = x + k * Rw / 8.5
        cab = (np.abs(X - cx_) < m(0.9)) & (np.abs(Y - y) < m(1.3))
        cv.put(cab, (52, 104, 196) if k % 2 else (70, 126, 214))
    cv.commit(rgb); print('  колесо огляду')


# ---------------------------------------------------------------- сквер Шевченка: яскраві скульптури Миколи Мазура
def mazur(D, rgb):
    pg = None
    for w in D['w']:
        t = w['t']
        P = np.array(w['pts']) * K
        if t.get('leisure') == 'playground' and abs(P[:, 0].mean() - 7140) < 40 and abs(P[:, 1].mean() - 8613) < 40:
            pg = P; break
    c = pg.mean(0) if pg is not None else np.array([7140.0, 8613.0])
    x, y = c
    cv = Canvas(x - 30, y - 24, x + 30, y + 24)
    X, Y = cv.grid()
    # «Дракон»: звивисте тіло з кілець, гребінь, голова
    pts = [(x - 20 + i * 1.6, y - 6 + 4.5 * math.sin(i * 0.55)) for i in range(18)]
    for i, (px, py) in enumerate(pts):
        r = m(1.3 if i < 15 else 1.0)
        d = (X - px) ** 2 + (Y - py) ** 2
        cv.put(d < r * r, (64, 168, 74) if i % 2 else (88, 196, 84))
        if i % 2 == 0: cv.put(d < (r * 0.45) ** 2, (230, 64, 52))
    hx, hy = pts[-1][0] + 2.2, pts[-1][1]
    cv.put((X - hx) ** 2 + (Y - hy) ** 2 < m(1.7) ** 2, (246, 196, 40))
    cv.put((X - hx - 1) ** 2 + (Y - hy + 0.6) ** 2 < m(0.4) ** 2, (30, 30, 30))
    # «Риба-кит»: синій кит з хвостом
    kx, ky = x - 10, y + 10
    body = ((X - kx) / m(3.6)) ** 2 + ((Y - ky) / m(2.0)) ** 2 < 1
    cv.put(body, (52, 120, 210))
    cv.put(((X - kx + 1.5) / m(2.4)) ** 2 + ((Y - ky - 0.6) / m(1.0)) ** 2 < 1, (150, 200, 240))
    tail = (np.abs(Y - ky) < (X - (kx + m(3.0))) * 0.9) & (X < kx + m(5.2)) & (X > kx + m(3.0))
    cv.put(tail, (40, 98, 186))
    # «Бабка»: тонке тіло й чотири прозорі крила
    bx, by = x + 12, y + 9
    for sgn in (-1, 1):
        for off in (-1.2, 1.2):
            wing = ((X - bx - off) / m(0.9)) ** 2 + ((Y - by - sgn * 3.2) / m(3.2)) ** 2 < 1
            cv.put(wing, (120, 200, 240))
    cv.put((np.abs(X - bx) < m(0.5)) & (np.abs(Y - by) < m(6)) , (226, 60, 150))
    cv.put((X - bx) ** 2 + (Y - by + m(5.5)) ** 2 < m(0.9) ** 2, (250, 150, 40))
    cv.commit(rgb); print('  скульптури Мазура', int(x), int(y))
    return int(x), int(y)


# ---------------------------------------------------------------- розв'язка: Кам'янецька мостом над Святослава Хороброго
def interchange(D, rgb):
    """Проліт мосту (з OSM: Кам'янецька, bridge=yes над Святослава Хороброго): асфальт із розміткою, бетонні
    огорожі зі стовпчиками, тінь на дорогу внизу. -> геометрія для гри (два рівні: верх/низ)."""
    best = None
    for w in D['w']:
        t = w['t']
        if 'янецька' in t.get('name', '') and t.get('bridge') == 'yes':
            P = np.array(w['pts']) * K
            d = np.abs(P.mean(0) - np.array([5659, 9951]) * F).sum()
            if best is None or d < best[0]: best = (d, P, t)
    _, P, t = best
    A, B = P[0], P[-1]
    c = (A + B) / 2; L = np.linalg.norm(B - A); u = (B - A) / L; v = np.array([-u[1], u[0]])
    hl = L / 2 + m(1.0)
    hw = CD.ROAD_W[t['highway']] / MPP / 2 + m(0.8)
    deck = [c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw]
    # проліт = продовження самої Кам'янецької: беремо чистий відрізок дороги перед мостом (той самий асфальт, бордюри,
    # розмітка) і повторюємо його вздовж прольоту з кроком, кратним кроку переривчастої лінії, — без швів і чужих кольорів
    x0, y0 = np.floor(np.min(deck, 0) - 80 * F).astype(int); x1, y1 = np.ceil(np.max(deck, 0) + 80 * F).astype(int)
    sub = np.asarray(rgb[y0:y1, x0:x1]).astype(np.float32)
    def sample(al, ac):
        X = c[0] + al * u[0] + ac * v[0] - x0; Y = c[1] + al * u[1] + ac * v[1] - y0
        return cv2.remap(sub, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR)
    acs = np.arange(-hw, hw, 0.5, dtype=np.float32)
    best = (None, 0)
    for side in (-1, 1):                                               # з якого боку чистіша дорога (менше розмітки переходів)
        al_ = side * (hl + np.arange(0, int(30 * F), dtype=np.float32))
        st = sample(al_[None, :].repeat(len(acs), 0), acs[:, None].repeat(len(al_), 1))
        var = st.std(1).mean()
        if best[0] is None or var < best[1]: best = (side, var, st)
    side = best[0]
    cen = best[2].mean(-1)
    row = cen[np.argmax(cen.std(1))]                                   # рядок із переривчастою лінією
    r = row - row.mean()
    per = max(range(int(6 * F), int(16 * F)), key=lambda q: np.corrcoef(r[:-q], r[q:])[0, 1])
    Pk = per * (int(30 * F) // per)
    sh = max(0.0, 0.0)
    shadow(rgb, deck, 5, 0.22)
    cv = Canvas(*(np.min(deck, 0) - 2), *(np.max(deck, 0) + 2))
    sel = cv.mask(deck).astype(bool)
    X, Y = cv.grid()
    al = (X - c[0]) * u[0] + (Y - c[1]) * u[1]
    ac = (X - c[0]) * v[0] + (Y - c[1]) * v[1]
    start = side * hl                                                  # відрізок-джерело: [hl, hl+Pk) з боку side
    d = (al - start) * side                                            # <0 — усередині прольоту
    al_src = start + side * np.mod(d, Pk)
    tex = sample(al_src, ac)
    cv.put(sel, tex[sel])
    par = sel & (np.abs(ac) > hw - m(0.45))                            # тонка бетонна огорожа в кольорі бордюру
    cv.put(par, np.clip(tex[par] * 1.12 + 10, 0, 255))
    cv.commit(rgb)
    print('  міст на розв\'язці: %.0f x %.0f м' % (hl * 2 * MPP, hw * 2 * MPP))
    return dict(c=[float(c[0]), float(c[1])], u=[float(u[0]), float(u[1])], hl=float(hl), hw=float(hw), rail=float(m(0.7)))


# ---------------------------------------------------------------- дороги вільні: паркани/лінії-бар'єри з деталей не перетинають проїжджу частину
def carve_roads(D, solid):
    n = 0
    for w in D['w']:
        t = w['t']
        if not CD.drivable(t) or t.get('access') in ('private', 'no') or t.get('layer', '0') != '0': continue
        P = np.round(np.array(w['pts']) * K).astype(np.int32)
        x0, y0 = P.min(0) - int(20 * F); x1, y1 = P.max(0) + int(20 * F)
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(solid.shape[1], x1), min(solid.shape[0], y1)
        if x1 <= x0 or y1 <= y0: continue
        sub = np.asarray(solid[y0:y1, x0:x1]).copy()
        mk = np.zeros(sub.shape, np.uint8)
        cv2.polylines(mk, [P - [x0, y0]], False, 1, max(3, int(CD.ROAD_W[t['highway']] / MPP * 0.7)))
        hit = sub & mk.astype(bool)
        if hit.any():
            n += int(hit.sum()); sub[mk.astype(bool)] = False; solid[y0:y1, x0:x1] = sub
    print('  дороги очищено від перешкод: %d пікс' % n)


# ---------------------------------------------------------------- речовий ринок: кіоски — тверді
def market_solid(D, rgb, solid):
    for w in D['w']:
        t = w['t']
        if t.get('amenity') != 'marketplace': continue
        P = np.array(w['pts']) * K
        if abs(P[:, 0].mean() - 3766 * F) > 400 * F or abs(P[:, 1].mean() - 9357 * F) > 400 * F: continue
        x0, y0 = np.floor(P.min(0)).astype(int); x1, y1 = np.ceil(P.max(0)).astype(int) + 1
        mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mk, [np.round(P - [x0, y0]).astype(np.int32)], 1)
        img = np.asarray(rgb[y0:y1, x0:x1]).astype(np.int32)
        # дахи кіосків: світло-сірі (ROOF у citydetail.markets, із затіненням) або кольорові акценти
        lum = img.mean(-1); sat_ = img.max(-1) - img.min(-1)
        roof = (lum > 118) & (sat_ < 26)
        acc = np.zeros_like(roof)
        for a in ((62, 104, 172), (166, 66, 54), (52, 122, 76), (214, 170, 52)):
            for f in (0.75, 0.85, 0.95, 1.05):
                acc |= np.abs(img - np.array(a) * f).sum(-1) < 36
        kiosk = mk.astype(bool) & (roof | acc)
        kiosk = cv2.morphologyEx(kiosk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        s = np.asarray(solid[y0:y1, x0:x1]).copy(); s |= kiosk
        solid[y0:y1, x0:x1] = s
        print('  ринок: тверді кіоски %.0f м²' % (kiosk.sum() * MPP * MPP))


def fresh_base():
    """латки кладуться на чисту карту: першого разу (або після нового citybuild) зберегти чисті копії, далі — відновлювати з них"""
    import shutil
    stamp = 'build/mappatch.json'
    for f in ('map_rgb', 'solid'):
        src, base = 'build/%s.npy' % f, 'build/%s_base.npy' % f
        new_city = not os.path.exists(stamp) or os.path.getmtime(src) > os.path.getmtime(stamp) + 1
        if not os.path.exists(base) or new_city:
            shutil.copyfile(src, base)
        else:
            shutil.copyfile(base, src)


def main():
    fresh_base()
    D = CD.data()
    rgb = np.load('build/map_rgb.npy', mmap_mode='r+')
    solid = np.load('build/solid.npy', mmap_mode='r+')
    sat = np.load('build/sat_map.npy', mmap_mode='r') if os.path.exists('build/sat_map.npy') else None
    landmark_roofs(D, rgb, sat)
    photo_roofs(D, rgb)
    monblan(D, rgb)
    chekman(D, rgb)
    mz = None                                                     # скульптури Мазура прибрано (25.10.2026: не схоже)
    br = interchange(D, rgb)
    carve_roads(D, solid)
    market_solid(D, rgb, solid)
    rgb.flush(); solid.flush()
    json.dump(dict(mazur=mz, bridges=[br]), open('build/mappatch.json', 'w'))


if __name__ == '__main__':
    main()
