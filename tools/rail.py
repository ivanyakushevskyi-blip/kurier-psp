"""Залізниця для поїздів (07.11.2026): головні колії (usage=main/branch, без service) з OSM -> безперервні лінії в
пікселях карти; переїзд на вул. Франка — точка на лінії й ребра графа доріг, що його перетинають.
lines(D, play) -> [np.array (N, 2)] (крок ~2 м); crossings(lines, g) -> [(лінія, дуга на лінії, [(ребро, напрям, s)])]"""
import math
import numpy as np
import citydraw as CD
from shapely.geometry import LineString, Point


def _ways(D):
    out = []
    for w in D['w']:
        t = w['t']
        if t.get('railway') == 'rail' and t.get('usage') in ('main', 'branch') and not t.get('service') and len(w['nds']) > 1:
            out.append(w)
    return out


def lines(D, play, min_in=600.0):
    K = CD.K; H, W = play.shape
    ws = _ways(D)
    ends = {}
    for i, w in enumerate(ws):
        for e in (w['nds'][0], w['nds'][-1]): ends.setdefault(e, []).append(i)
    used = set(); res = []
    def pts(i, rev=False):
        P = np.array(ws[i]['pts'], np.float64) * K
        return P[::-1] if rev else P
    def heading(P, at_end):
        a, b = (P[-2], P[-1]) if at_end else (P[1], P[0])
        return math.atan2(b[1] - a[1], b[0] - a[0])
    # кінці ланцюгів: вузли з одним шляхом
    starts = [(n, l[0]) for n, l in ends.items() if len(l) == 1]
    for n0, i0 in starts:
        if i0 in used: continue
        rev = ws[i0]['nds'][-1] == n0
        chain = [pts(i0, rev)]; used.add(i0); cur = ws[i0]['nds'][0] if rev else ws[i0]['nds'][-1]
        while True:
            h = heading(np.vstack(chain), True); best = None
            for j in ends.get(cur, []):
                if j in used: continue
                rj = ws[j]['nds'][-1] == cur
                P = pts(j, rj); hj = math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0])
                dd = abs((hj - h + math.pi) % (2 * math.pi) - math.pi)
                if dd < 0.5 and (best is None or dd < best[0]): best = (dd, j, rj)
            if best is None: break
            _, j, rj = best; used.add(j); P = pts(j, rj); chain.append(P[1:])
            cur = ws[j]['nds'][0] if rj else ws[j]['nds'][-1]
        P = np.vstack(chain)
        L = LineString(P)
        n = max(2, int(L.length / (2.0 / CD.MPP)))
        R = np.array([L.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])
        inside = sum(1 for x, y in R if 0 <= x < W and 0 <= y < H and play[int(y), int(x)])
        if inside * 2.0 >= min_in: res.append(R)
    res.sort(key=len, reverse=True)
    print('  залізниця: ліній %d, довжини %s м' % (len(res), [int(len(r) * 2) for r in res]))
    return res


def crossings(lines_, g, D, names=('Франка',)):
    """переїзди: перетини ліній з дорогами з цими назвами (решта перетинів — мости)"""
    out = []
    for li, R in enumerate(lines_):
        LR = LineString(R)
        for ei, e in enumerate(g['edges']):
            nm = e['t'].get('name', '')
            if not any(k in nm for k in names): continue
            LE = LineString(e['q'])
            if not LR.intersects(LE): continue
            p = LR.intersection(LE)
            p = p if p.geom_type == 'Point' else p.centroid
            arc = LR.project(p); s0 = LE.project(p)
            out.append((li, arc, ei, s0, e['L']))
    print('  переїзди: %d %s' % (len(out), [(o[0], int(o[1])) for o in out]))
    return out
