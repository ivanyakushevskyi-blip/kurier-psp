"""Запарковані машини (01.11.2026): на паркуваннях (amenity=parking) — рядами в смугах розмітки, на широких дорогах — біля
бордюру поза смугами руху, на вузьких — наполовину на тротуарі. Лише там, де вільно (не стіна, не дерево, не перехрестя).
build(D, g, W, H, play, solid, surf, trees) -> [(x, y, кут 0..255, модель)] у пікселях карти; bake.py пише масив у ресурси."""
import math, random
import numpy as np, cv2
import citydraw as CD

M = lambda m: m / CD.MPP                         # метри -> пікселі карти
CAR_L, CAR_W = 4.6, 1.9                          # габарит для перевірки місця (м)


def _free(x, y, a, solid, surf, play, tg, W, H, ok_surf):
    """усі точки кузова — у зоні, не тверді, на дозволеній поверхні; поруч немає стовбура"""
    c, s = math.cos(a), math.sin(a)
    for lx in (-CAR_L / 2, 0, CAR_L / 2):
        for ly in (-CAR_W / 2, 0, CAR_W / 2):
            px, py = int(x + M(lx) * c - M(ly) * s), int(y + M(lx) * s + M(ly) * c)
            if not (0 <= px < W and 0 <= py < H) or not play[py, px] or solid[py, px]: return False
            if (int(surf[py, px]) & 15) not in ok_surf: return False
    gx, gy = int(x // 64), int(y // 64)
    for k in ((gx + i, gy + j) for i in (-1, 0, 1) for j in (-1, 0, 1)):
        for tx, ty in tg.get(k, ()):
            if (tx - x) ** 2 + (ty - y) ** 2 < M(2.0) ** 2: return False
    return True


def lots(D, W, H):
    """контури відкритих паркувань у пікселях карти"""
    out = []
    for w in D['w']:
        t = w['t']
        if not w.get('closed') or t.get('amenity') != 'parking' or t.get('parking') in ('underground', 'multi-storey') or len(w['pts']) < 4: continue
        P = np.array(w['pts'], np.float32) * CD.K
        if P[:, 0].max() < 0 or P[:, 1].max() < 0 or P[:, 0].min() > W or P[:, 1].min() > H: continue
        out.append(P)
    return out


ROW, AISLE, STALL = 5.0, 6.5, 2.6               # ряд, проїзд, місце (м)


def lot_rows(P):
    """геометрія розкладки: центр, осі (уздовж довшої сторони, поперек), список (поперечна позиція центру ряду, напрям носа)"""
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(P); a = math.radians(ang)
    ux, uy = (math.cos(a), math.sin(a)) if rw >= rh else (-math.sin(a), math.cos(a))
    L, S = max(rw, rh), min(rw, rh)
    rows = []
    if S < M(ROW + AISLE * 0.6): return (cx, cy, ux, uy, L, S, rows)
    # пари рядів ніс до носа, між парами проїзд; якщо лишається місце — одиночний ряд біля проїзду
    period = M(2 * ROW + AISLE)
    n = int((S + M(AISLE) * 0.5) // period)
    used = n * period - M(AISLE)
    single = S - used - M(AISLE) >= M(ROW) * 0.9
    tot = used + (M(AISLE) + M(ROW) if single else 0)
    a0 = -tot / 2
    for k in range(n):
        b = a0 + k * period
        rows.append((b + M(ROW) / 2, +1)); rows.append((b + M(ROW) * 1.5, -1))
    if single: rows.append((a0 + n * period + M(ROW) / 2, -1))
    return (cx, cy, ux, uy, L, S, rows)


def lot_slots(P):
    """місця: (x, y, кут машини, чи в межах паркування)"""
    cx, cy, ux, uy, L, S, rows = lot_rows(P)
    x0, y0 = np.floor(P.min(0)).astype(int); x1, y1 = np.ceil(P.max(0)).astype(int)
    mk = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mk, [np.round(P - [x0, y0]).astype(np.int32)], 1)
    mk = cv2.erode(mk, np.ones((int(M(0.6)) * 2 + 1,) * 2, np.uint8))
    out = []
    n = int(L / M(STALL))
    for acr, nose in rows:
        for ks in range(n):
            al = -n * M(STALL) / 2 + (ks + 0.5) * M(STALL)
            x = cx + ux * al - uy * acr; y = cy + uy * al + ux * acr
            ok = True
            for da in (-M(2.3), M(2.3)):                                       # обидва кінці машини — у межах
                for dl in (-M(1.0), M(1.0)):
                    xx, yy = int(x + ux * dl - uy * da) - x0, int(y + uy * dl + ux * da) - y0
                    if not (0 <= xx < mk.shape[1] and 0 <= yy < mk.shape[0]) or not mk[yy, xx]: ok = False
            car_a = math.atan2(ux, -uy) if nose > 0 else math.atan2(-ux, uy)   # ніс поперек, до сусіднього ряду
            out.append((x, y, car_a, ok))
    return out


def build(D, g, W, H, play, solid, surf, trees, seed=11):
    rs = random.Random(seed)
    tg = {}
    for t in trees: tg.setdefault((int(t[0]) // 64, int(t[1]) // 64), []).append((t[0], t[1]))
    out = []
    taken = {}                                    # сітка 4 м — не ставити машини одна на одну
    def put(x, y, a):
        k = (int(x // M(4)), int(y // M(4)))
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for (qx, qy) in taken.get((k[0] + i, k[1] + j), ()):
                    if (qx - x) ** 2 + (qy - y) ** 2 < M(4.4) ** 2: return False
        taken.setdefault(k, []).append((x, y))
        out.append((int(round(x)), int(round(y)), int(round((a % (2 * math.pi)) / (2 * math.pi) * 256)) & 255, rs.randrange(8)))
        return True
    # ---- паркування: ряди «ніс до носа» з проїздами (lot_slots — та сама розкладка, що й розмітка в mappatch), ~50% місць
    npark = 0
    for P in lots(D, W, H):
        for x, y, car_a, ok_in in lot_slots(P):
            if not ok_in or rs.random() > 0.5: continue
            if _free(x, y, car_a, solid, surf, play, tg, W, H, (0, 1, 3, 4)) and put(x, y, car_a): npark += 1
    # ---- уздовж доріг: широкі — біля бордюру на проїжджій частині поза смугами; вузькі — наполовину на тротуарі
    nroad = ncurb = 0
    for e in g['edges']:
        if not e['traf'] or e['cls'] > 3: continue
        q = np.array(e['q']); seg = np.diff(q, axis=0); sl = np.hypot(seg[:, 0], seg[:, 1]); Ltot = sl.sum()
        if Ltot < M(40): continue
        wm = e['w'] * CD.MPP
        per = max(1, e['lanes'] if e['one'] else e['lanes'] // 2)
        if e['one']: lane_edge = min(3.4, wm / per) * per / 2                 # зовнішній край смуг від осі (м)
        else: lane_edge = min(3.4, wm / (2 * per)) * per + 0.35
        room = wm / 2 - lane_edge                                              # вільно між смугами й бордюром
        if room >= 2.3 and e['cls'] <= 2: off_m, kind, ok = wm / 2 - 1.15, 'road', (0, 3, 4)
        elif e['cls'] >= 2: off_m, kind, ok = wm / 2 + 0.75, 'curb', (0, 1, 3, 4)
        else: continue
        prob = 0.22 if kind == 'road' else 0.20
        sides = (1,) if e['one'] else (1, -1)
        acc = np.concatenate([[0], np.cumsum(sl)])
        s_ = M(16) + rs.random() * M(6)
        while s_ < Ltot - M(16):
            k = min(len(sl) - 1, int(np.searchsorted(acc, s_) - 1)); k = max(k, 0)
            tt = (s_ - acc[k]) / max(sl[k], 1e-3); bx, by = q[k] + seg[k] * tt
            ux, uy = seg[k] / max(sl[k], 1e-3)
            for sd in sides:
                if rs.random() > prob: continue
                x, y = bx - uy * M(off_m) * sd, by + ux * M(off_m) * sd
                a = math.atan2(uy, ux) + (0 if sd > 0 else math.pi)
                if _free(x, y, a, solid, surf, play, tg, W, H, ok) and put(x, y, a):
                    if kind == 'road': nroad += 1
                    else: ncurb += 1
            s_ += M(5.6) + (M(4) if rs.random() < 0.3 else 0)
    print('  запарковані машини: паркування %d, біля бордюру %d, на тротуарі %d' % (npark, nroad, ncurb))
    return out
