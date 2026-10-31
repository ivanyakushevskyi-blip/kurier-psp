"""Будинки для паралаксу дахів (bake.py): центр, висота (м), контур (≤32 вершини) з трикутниками —
для маски дахів у карті й темної плями під дахом; розширений на ~0,5 пікс контур з трикутниками —
для графічного чипа (плавний зсув зі згладженими краями).
"""
import numpy as np, cv2


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


def buildings(CD, md, W, H, NCX, CS, play):
    """-> список (cx, cy, h, контур, розширений контур, трикутники розшир., трикутники контуру), відсортований за блоком центру;
       початки блоків (NCX*NCY+1)"""
    D = CD.data()
    arch = [np.array(q, np.float32) for q in md.get('arches', [])]
    bl = []
    for b in D['B'] + D['SAT']:
        Q = np.array(b['q'], np.float32) * CD.K
        if len(Q) < 3: continue
        cx, cy = Q.mean(0)
        if not (0 <= cx < W and 0 <= cy < H) or not play[int(cy), int(cx)]: continue
        if cv2.contourArea(Q) < 3.0 / CD.MPP ** 2: continue
        if any(cv2.pointPolygonTest(A, (float(cx), float(cy)), False) >= 0 for A in arch): continue     # будинки над арками — окремо
        eps = 0.7
        while True:
            q = cv2.approxPolyDP(Q, eps, True)[:, 0, :]
            if len(q) <= 32: break
            eps *= 1.4
        if len(q) < 3: continue
        e = expand(Q)
        if e is None: continue
        tr = earclip(e)
        qi = np.round(q).astype(np.int16)
        ot = earclip(qi)
        if not tr or not ot: continue
        bl.append((int(cx), int(cy), int(round(b.get('h') or 6)), qi, e, np.array(tr, np.uint8), np.array(ot, np.uint8)))
    bl.sort(key=lambda t: (t[1] // CS) * NCX + t[0] // CS)
    NCY = H // CS
    bc = np.zeros(NCX * NCY + 1, np.uint32)
    for t in bl: bc[(t[1] // CS) * NCX + t[0] // CS + 1] += 1
    return bl, np.cumsum(bc).astype(np.uint32)
