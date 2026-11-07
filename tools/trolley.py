"""Тролейбусні маршрути (07.11.2026): справжні маршрути з OSM (relation route=trolleybus) -> послідовність ребер
графа доріг гри (ребро, напрям) + зупинки (крок, відстань уздовж ребра). Тролейбус у грі їде лише по своєму списку.
build(g, D) -> [dict(ref, steps=[(e, dir)], stops=[(k, s_px)])]"""
import glob, math
import xml.etree.ElementTree as ET
import numpy as np
import citydraw as CD, citygen as CG
from shapely.geometry import LineString, Point

TOL = 9.0                       # м: ребро вважається частиною маршруту, якщо всі його точки ближче


def relations(src='data/osm_v3'):
    rels = {}
    for f in sorted(glob.glob(src + '/*.osm')):
        for r in ET.parse(f).getroot().iter('relation'):
            t = {x.get('k'): x.get('v') for x in r.iter('tag')}
            if t.get('route') != 'trolleybus' or r.get('id') in rels: continue
            mem = [(m.get('type'), m.get('ref'), m.get('role') or '') for m in r.iter('member')]
            rels[r.get('id')] = dict(ref=t.get('ref', '?'), name=t.get('name', ''), mem=mem)
    return rels


def route_line(rel, wid, D):
    """упорядкована ламана маршруту в пікселях карти (зшиті члени-шляхи з правильною орієнтацією)"""
    K = CD.K
    pts = []
    for typ, ref, role in rel['mem']:
        if typ != 'way' or ref not in wid or role in ('platform', 'stop'): continue
        w = wid[ref]
        P = [tuple(np.array(CG.px(*D['nodes'][n])) * K) for n in w['nds'] if n in D['nodes']]
        if len(P) < 2: continue
        if not pts: pts = P; continue
        e = pts[-1]
        d0 = math.dist(e, P[0]); d1 = math.dist(e, P[-1])
        if len(pts) and d1 < d0: P = P[::-1]
        if not any(pts) or min(d0, d1) > 60 / CD.MPP:           # розрив — перший шлях могли орієнтувати навпаки
            if len(pts) and math.dist(pts[0], P[0]) < min(d0, d1): pts = pts[::-1]
        pts.extend(P[1:] if math.dist(pts[-1], P[0]) < 1 else P)
    return pts


def build(g, D):
    wid = {w['id']: w for w in D['w']}
    nodes, edges = g['nodes'], g['edges']
    out_adj = {}
    for i, e in enumerate(edges):
        out_adj.setdefault(e['a'], []).append((i, 0))
        if not e['one']: out_adj.setdefault(e['b'], []).append((i, 1))
    def epts(i, d):
        q = edges[i]['q']; return q if d == 0 else q[::-1]
    tol = TOL / CD.MPP
    res = []
    for rid, rel in relations().items():
        P = route_line(rel, wid, D)
        if len(P) < 2: continue
        R = LineString(P)
        # вузли графа біля маршруту, упорядковані вздовж нього (старти шматків: маршрут може виходити за межу зони)
        near = []
        for n_, p in enumerate(nodes):
            if n_ not in out_adj: continue
            pt = Point(p)
            if R.distance(pt) < tol: near.append((R.project(pt), n_))
        near.sort()
        pieces, arc0 = [], -1.0
        while True:
            st = next(((a_, n_) for a_, n_ in near if a_ > arc0 + 1), None)
            if st is None: break
            arc, n = st; steps = []; seen = set()
            while True:
                cand = []
                for i, d in out_adj.get(n, []):
                    if (i, d) in seen: continue
                    q = epts(i, d); L = LineString(q)
                    samp = [L.interpolate(t, normalized=True) for t in np.linspace(0, 1, 6)]
                    dm = max(R.distance(s_) for s_ in samp)
                    a1 = R.project(samp[-1])
                    if dm < tol and a1 > arc + 2: cand.append((dm, -a1, i, d, a1))
                if not cand: break
                cand.sort(); dm, _, i, d, a1 = cand[0]
                steps.append((i, d)); seen.add((i, d)); arc = a1
                n = edges[i]['b'] if d == 0 else edges[i]['a']
            arc0 = max(arc, st[0])
            if len(steps) >= 4: pieces.append(steps)
        for steps in pieces:
            # зупинки: вузли-члени зі stop/platform, прив'язані до кроку й відстані вздовж ребра
            stops = []
            for typ, ref, role in rel['mem']:
                if typ != 'node' or not role.startswith(('stop', 'platform')) or ref not in D['nodes']: continue
                sp = Point(*(np.array(CG.px(*D['nodes'][ref])) * CD.K))
                bk = None
                for k, (i, d) in enumerate(steps):
                    L = LineString(epts(i, d)); dd = L.distance(sp)
                    if dd < 25 / CD.MPP and (bk is None or dd < bk[0]): bk = (dd, k, L.project(sp))
                if bk and not any(s[0] == bk[1] and abs(s[1] - bk[2]) < 30 / CD.MPP for s in stops): stops.append((bk[1], bk[2]))
            stops.sort()
            res.append(dict(ref=rel['ref'], name=rel['name'], steps=steps, stops=stops))
    print('  тролейбусні маршрути: %d (%s), кроків %d, зупинок %d' % (len(res), ', '.join(sorted({r['ref'] for r in res})),
          sum(len(r['steps']) for r in res), sum(len(r['stops']) for r in res)))
    return res
