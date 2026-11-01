"""Нічне світло (01.11.2026): ліхтарі — знайдені на самій карті (стовп + світильник, які малює citydetail.lamps уздовж
головних вулиць), світло з вікон — плями біля будинків (більше в центрі й у багатоповерхівках, у приватному секторі — подекуди).
build(rgb, g, bl, dens, W, H, play) -> [(x, y, радіус пікс, вид)]; вид: 0 — ліхтар, 1 — тепле вікно, 2 — холодне вікно."""
import math, random
import numpy as np, cv2
import citydraw as CD

M = lambda m: m / CD.MPP


def lamps(rgb, g, W, H):
    """світильники ліхтарів: світла еліпса (226,226,214) поруч із темним корпусом (54,56,60), лише біля головних вулиць"""
    out = []
    corr = np.zeros(((H + 7) // 8, (W + 7) // 8), np.uint8)                 # коридор головних вулиць (сітка 8 пікс)
    for e in g['edges']:
        if e['cls'] > 2: continue
        q = np.round(np.array(e['q']) / 8).astype(np.int32)
        cv2.polylines(corr, [q], False, 1, max(1, int((e['w'] / 2 + M(4)) * 2 / 8)))
    for y0 in range(0, H, 1024):
        a = np.asarray(rgb[y0:y0 + 1024]).astype(np.int16)
        c = corr[y0 // 8:(y0 + 1024) // 8 + 1]
        cm = np.repeat(np.repeat(c, 8, 0), 8, 1)[:a.shape[0], :a.shape[1]].astype(bool)
        if not cm.any(): continue
        head = (np.abs(a - np.array([226, 226, 214], np.int16)).sum(-1) < 40) & cm
        body = np.abs(a - np.array([54, 56, 60], np.int16)).sum(-1) < 45
        head &= cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        n, lab, st, cen = cv2.connectedComponentsWithStats(head.astype(np.uint8))
        for i in range(1, n):
            if 1 <= st[i, 4] <= 14: out.append((cen[i][0], cen[i][1] + y0))
    # злиті дублікати (ближче 3 м)
    res = []
    grid = {}
    for x, y in out:
        k = (int(x // M(3)), int(y // M(3)))
        if any((qx - x) ** 2 + (qy - y) ** 2 < M(3) ** 2 for i in (-1, 0, 1) for j in (-1, 0, 1) for qx, qy in grid.get((k[0] + i, k[1] + j), ())): continue
        grid.setdefault(k, []).append((x, y)); res.append((x, y))
    return res


def build(rgb, g, bl, dens, W, H, play, seed=5):
    rs = random.Random(seed)
    out = [(int(x), int(y), int(M(11)), 0) for x, y in lamps(rgb, g, W, H)]
    nl = len(out)
    # вікна: плями світла на землі біля стін (зовні контуру на 1,5 м)
    for t in bl:
        cx, cy, h, q = t[0], t[1], t[2], t[3].astype(np.float32)
        area = cv2.contourArea(q) * CD.MPP ** 2
        if area < 25: continue
        f = dens(cx, cy)
        tall = h >= 9                                                     # багатоповерхівка
        n = int(min(10, area / (90 if tall else 160) + 0.5)) if tall else (1 if rs.random() < 0.25 + 0.35 * f else 0)
        if tall: n = max(1, int(n * (0.5 + 0.5 * f) + 0.5))
        per = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([q, q[:1]]), axis=0).T))]
        L = per[-1]
        if L < 1: continue
        for k in range(n):
            s = rs.random() * L
            i = min(len(q) - 1, int(np.searchsorted(per, s) - 1))
            a, b = q[i], q[(i + 1) % len(q)]
            seg = b - a; sl = np.hypot(*seg)
            if sl < 1: continue
            p = a + seg * ((s - per[i]) / sl)
            nrm = np.array([seg[1], -seg[0]]) / sl
            if cv2.pointPolygonTest(q, (float(p[0] + nrm[0] * 3), float(p[1] + nrm[1] * 3)), False) > 0: nrm = -nrm   # назовні
            x, y = p + nrm * M(2.6)
            if not (0 <= x < W and 0 <= y < H) or not play[int(y), int(x)]: continue
            out.append((int(x), int(y), int(M(4.0 + rs.random() * 2)), 1 if rs.random() < 0.8 else 2))
    print('  нічне світло: ліхтарів %d, вікон %d' % (nl, len(out) - nl))
    return out
