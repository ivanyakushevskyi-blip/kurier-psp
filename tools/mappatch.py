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
MPP = CD.MPP                                     # 0,5 м на піксель карти
LV = np.array([-0.5, -0.5, 0.7]); LV /= np.linalg.norm(LV)     # світло з північного заходу (тіні на південний схід)
GOLD = (226, 182, 64)


def m(x): return x / MPP                          # метри -> пікселі карти


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
    edge = sel & (d2 > (R - 0.6) ** 2)
    c[edge] *= 0.6
    cv.put(sel, np.clip(c[sel], 0, 255))
    if cross:
        s = max(R * 0.55, 1.6)
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
    k = min(14.0, h_m * 0.22 * 1.6)
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
    f = 0.5 / 0.32
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
    mk = cv2.erode(mk, np.ones((5, 5), np.uint8))
    px = s[mk.astype(bool)]
    if len(px) < 10: return default
    return tuple(int(v) for v in np.median(px, 0))


def find_poly(D, x, y, src=('B', 'SAT')):
    best = None
    for s in src:
        for b in D[s]:
            Q = np.array(b['q'], np.float32) * K
            if abs(Q[:, 0].mean() - x) > 300 or abs(Q[:, 1].mean() - y) > 300: continue
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
    # id: точка, колір даху, колір бань, бані в частках прямокутника (u, v, радіус м)
    'pokrov':   ((5749, 8694), (226, 222, 210), (226, 180, 60), [(0, 0, 3.6), (-0.3, -0.28, 1.8), (-0.3, 0.28, 1.8), (0.3, -0.28, 1.8), (0.3, 0.28, 1.8), (-0.44, 0, 1.6)]),
    'rizdva':   ((6779, 7049), (196, 104, 66), (206, 120, 74), [(0, 0, 4.4), (-0.3, -0.3, 1.7), (-0.3, 0.3, 1.7), (0.3, -0.3, 1.7), (0.3, 0.3, 1.7)]),
    'ugcc':     ((10328, 5899), (222, 222, 216), (96, 112, 136), [(0, 0, 3.8)]),
    'mykol':    ((9600, 4840), (84, 112, 92), (224, 178, 58), [(0, 0, 3.4), (-0.3, -0.26, 1.6), (-0.3, 0.26, 1.6), (0.3, -0.26, 1.6), (0.3, 0.26, 1.6)]),
    'triitska': ((9205, 6362), (206, 104, 48), (224, 178, 58), [(0.05, 0, 2.2)]),
}
PLAIN = {      # будівлі без бань: колір зі знімка, скати за контуром
    'arkhiv': ((6349, 8535), 3.5), 'vokzal': ((10955, 9805), 5.0), 'oazis': ((7134, 6174), 2.0), 'monblan': ((6885, 5875), 2.0),
}


def landmark_roofs(D, rgb, sat):
    for k, (pt, roofc, domec, domes) in CHURCHES.items():
        P = find_poly(D, *pt, src=('B',))
        c, u, v, L, W = rect_frame(P)
        cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
        hip_roof(cv, P, roofc, pitch_m=min(6.0, W * MPP * 0.45), steep=1.2)
        for fu, fv, r in domes:
            x, y = c + u * fu * L + v * fv * W
            dome(cv, x, y, r, domec)
        cv.commit(rgb)
        print('  дах', k)
    # костел: хрестоподібний сірий дах + вежа з гострим шпилем на південному кінці
    P = find_poly(D, 8027, 3710, src=('B',))
    c, u, v, L, W = rect_frame(P)
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    hip_roof(cv, P, (150, 156, 166), pitch_m=6.0, steep=1.3)
    end = c + u * L * 0.4 * (1 if u[1] > 0 else -1)
    end = c + (end - c) * 0.9
    tower(cv, *end, 8.0, (196, 198, 204), (70, 76, 90), ang=math.atan2(u[1], u[0]))
    cv.commit(rgb); print('  дах kostel')
    for k, (pt, pitch) in PLAIN.items():
        P = find_poly(D, *pt, src=('B',))
        col = map_color(rgb, P, (180, 180, 184))
        cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
        hip_roof(cv, P, col, pitch_m=pitch, steep=1.0)
        cv.commit(rgb); print('  дах', k, col)
    # стара пожежна вежа (каланча): будівля + квадратна вежа над вузлом OSM з довгою тінню
    P = find_poly(D, 6773, 7489, src=('B',))
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    hip_roof(cv, P, map_color(rgb, P, (176, 170, 160)), pitch_m=4.0)
    cv.commit(rgb)
    tx, ty = 6772, 7489
    sq = [(tx - m(4), ty - m(4)), (tx + m(4), ty - m(4)), (tx + m(4), ty + m(4)), (tx - m(4), ty + m(4))]
    shadow(rgb, sq, 30)
    cv = Canvas(tx - 12, ty - 12, tx + 12, ty + 12)
    tower(cv, tx, ty, 8.0, (232, 196, 92), (120, 64, 50))
    cv.commit(rgb); print('  пожежна вежа')
    # трибуни стадіону «Поділля»: ряди сидінь уздовж поля + дах-козирок
    P = find_poly(D, 8742, 8930, src=('B',))
    c, u, v, L, W = rect_frame(P)
    cv = Canvas(*(P.min(0) - 3), *(P.max(0) + 3))
    sel = cv.mask(P).astype(bool)
    X, Y = cv.grid()
    fx, fy = 8742 - c[0], 8840 - c[1]                              # з боку поля — нижні ряди
    sgn = 1 if fx * v[0] + fy * v[1] > 0 else -1
    t = ((X - c[0]) * v[0] + (Y - c[1]) * v[1]) * sgn / (W / 2)      # -1 (далеко від поля) .. 1 (біля поля)
    row = np.floor((t + 1) * W * MPP / 0.9).astype(int)
    seats = np.where((row % 2)[..., None] == 0, np.array((176, 44, 44), np.float32), np.array((138, 32, 36), np.float32))
    cv.put(sel, seats[sel])
    canopy = sel & (t < -0.35)
    cv.put(canopy, np.array((214, 216, 220), np.float32) * (0.9 + 0.1 * ((X + Y) % 6 < 3))[canopy][..., None] if False else (212, 214, 218))
    edge = sel & ~cv2.erode(sel.astype(np.uint8), np.ones((SS * 2,) * 2, np.uint8)).astype(bool)
    cv.put(edge, (110, 112, 118))
    cv.commit(rgb); print('  трибуни')


# ---------------------------------------------------------------- парк Чекмана: білий міст, колесо огляду
def chekman(D, rgb):
    for w in D['w']:
        t = w['t']
        P = np.array(w['pts']) * K
        if t.get('man_made') == 'bridge' and t.get('bridge:structure') == 'humpback' and abs(P[:, 0].mean() - 5465) < 40:
            break
    else:
        print('  !! білий міст не знайдено'); return
    c, u, v, L, W = rect_frame(P)
    W2 = max(W, m(4.0)) / 2 + m(0.4); L2 = L / 2 + m(1.0)
    deck = [c - u * L2 - v * W2, c + u * L2 - v * W2, c + u * L2 + v * W2, c - u * L2 + v * W2]
    shadow(rgb, deck, 3, 0.45)
    cv = Canvas(*(np.min(deck, 0) - 3), *(np.max(deck, 0) + 3))
    sel = cv.mask(deck).astype(bool)
    X, Y = cv.grid()
    al = ((X - c[0]) * u[0] + (Y - c[1]) * u[1]) / L2                 # -1..1 уздовж
    ac = ((X - c[0]) * v[0] + (Y - c[1]) * v[1]) / W2                 # -1..1 поперек
    hump = 1 - al ** 2                                                  # горб: середина світліша
    col = np.array((236, 236, 232), np.float32)[None, None, :] * (0.88 + 0.12 * hump)[..., None]
    cv.put(sel, col[sel])
    rail = sel & (np.abs(ac) > 0.78)
    cv.put(rail, (250, 250, 248))
    rail_sh = sel & (np.abs(ac) > 0.7) & (np.abs(ac) <= 0.78)
    cv.put(rail_sh, (186, 188, 190))
    posts = rail & (np.mod((al + 1) * L2 / m(1.6), 1) < 0.22)
    cv.put(posts, (150, 152, 156))
    cv.commit(rgb); print('  білий міст')
    # колесо огляду: згори — вузький обід із синіми кабінками, опори, довга тінь на південний схід
    x, y = 4278, 6643
    Rw = m(16)
    ax = np.array([1.0, 0.0])
    rim = [(x - Rw, y - m(1.2)), (x + Rw, y - m(1.2)), (x + Rw, y + m(1.2)), (x - Rw, y + m(1.2))]
    shadow(rgb, rim, 34, 0.32)
    cv = Canvas(x - Rw - 6, y - 10, x + Rw + 6, y + 10)
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
            d = np.abs(P.mean(0) - [5659, 9951]).sum()
            if best is None or d < best[0]: best = (d, P, t)
    _, P, t = best
    A, B = P[0], P[-1]
    c = (A + B) / 2; L = np.linalg.norm(B - A); u = (B - A) / L; v = np.array([-u[1], u[0]])
    hl = L / 2 + m(1.0)
    hw = CD.ROAD_W[t['highway']] / MPP / 2 + m(0.8)
    deck = [c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw]
    shadow(rgb, deck, 6, 0.5)
    cv = Canvas(*(np.min(deck, 0) - 3), *(np.max(deck, 0) + 3))
    sel = cv.mask(deck).astype(bool)
    X, Y = cv.grid()
    al = (X - c[0]) * u[0] + (Y - c[1]) * u[1]
    ac = (X - c[0]) * v[0] + (Y - c[1]) * v[1]
    noise = (np.sin(X * 1.7) * np.cos(Y * 2.3) * 3).astype(np.float32)
    asph = np.dstack([88 + noise, 90 + noise, 96 + noise])
    cv.put(sel, asph[sel])
    lanes = int(t.get('lanes', 4))
    lw = (hw - m(0.8)) * 2 / lanes
    for k in range(1, lanes):
        off = -hw + m(0.8) + k * lw
        if k == lanes // 2:                                            # подвійна суцільна по осі
            ln = sel & ((np.abs(ac - off - m(0.15)) < m(0.08)) | (np.abs(ac - off + m(0.15)) < m(0.08)))
        else:                                                         # переривчасті між смугами
            ln = sel & (np.abs(ac - off) < m(0.08)) & (np.mod(al, m(9)) < m(3))
        cv.put(ln, (226, 226, 220))
    edge_line = sel & (np.abs(np.abs(ac) - (hw - m(1.1))) < m(0.08))
    cv.put(edge_line, (226, 226, 220))
    rail = sel & (np.abs(ac) > hw - m(0.7))
    lit = ac * (v[0] + v[1]) < 0
    cv.put(rail & lit, (214, 214, 208)); cv.put(rail & ~lit, (176, 176, 172))
    posts = rail & (np.mod(al, m(2.5)) < m(0.35))
    cv.put(posts, (140, 140, 138))
    cv.commit(rgb)
    print('  міст на розв\'язці: %.0f x %.0f м' % (hl * 2 * MPP, hw * 2 * MPP))
    return dict(c=[float(c[0]), float(c[1])], u=[float(u[0]), float(u[1])], hl=float(hl), hw=float(hw), rail=float(m(0.7)))


# ---------------------------------------------------------------- речовий ринок: кіоски — тверді
def market_solid(D, rgb, solid):
    for w in D['w']:
        t = w['t']
        if t.get('amenity') != 'marketplace': continue
        P = np.array(w['pts']) * K
        if abs(P[:, 0].mean() - 3766) > 400 or abs(P[:, 1].mean() - 9357) > 400: continue
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
    chekman(D, rgb)
    mz = mazur(D, rgb)
    br = interchange(D, rgb)
    market_solid(D, rgb, solid)
    rgb.flush(); solid.flush()
    json.dump(dict(mazur=mz, bridges=[br]), open('build/mappatch.json', 'w'))


if __name__ == '__main__':
    main()
