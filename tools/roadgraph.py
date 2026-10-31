"""Дорожній граф з OSM для гри: перехрестя (вузли) і ділянки між ними (ребра-ламані) з класом дороги,
кількістю смуг, односторонністю, шириною, світлофорами на кінцях. Потрібен помічнику смуги й трафіку.
build(D, play, W, H) -> dict з масивами; bake.py пише їх у assets.h.
"""
import math
import numpy as np, cv2
import citydraw as CD

CLS = {'trunk': 0, 'primary': 0, 'secondary': 1, 'tertiary': 2, 'primary_link': 2, 'secondary_link': 2, 'tertiary_link': 2,
       'residential': 3, 'unclassified': 3, 'living_street': 4, 'service': 4}
MAIN = 2                       # класи 0..2 — основні дороги (трафік)


def build(D, play, W, H):
    K = CD.K
    ways = []
    for w in D['w']:
        t = w['t']; h = t.get('highway')
        if h not in CLS or not CD.drivable(t) or t.get('layer', '0') not in ('0',): continue
        if t.get('access') in ('private', 'no') or t.get('service') in ('parking_aisle', 'driveway'): continue
        ways.append(w)
    use = {}
    for w in ways:
        for k, n in enumerate(w['nds']):
            use[n] = use.get(n, 0) + (2 if 0 < k < len(w['nds']) - 1 else 1)
    jn = {n for n, c in use.items() if c != 2}                         # кінці й перехрестя
    sig_nodes = {nid for nid, t in D['n'].items() if t.get('highway') == 'traffic_signals'}
    node_id, nodes = {}, []
    def nid_of(n, p):
        if n not in node_id: node_id[n] = len(nodes); nodes.append(p)
        return node_id[n]
    edges, pts = [], []
    for w in ways:
        t = w['t']; h = t['highway']
        ow = t.get('oneway')
        one = 1 if (ow in ('yes', '1', 'true') or t.get('junction') == 'roundabout' or h.endswith('_link') and ow != 'no' and False) else (-1 if ow == '-1' else 0)
        nl = 1 if one else 2
        try: nl = max(nl, int(str(t.get('lanes', nl)).split(';')[0]))
        except ValueError: pass
        P = np.array(w['pts'], np.float64) * K
        nds = w['nds']
        cut = [0] + [k for k in range(1, len(nds) - 1) if nds[k] in jn] + [len(nds) - 1]
        for a, b in zip(cut[:-1], cut[1:]):
            seg = P[a:b + 1]
            if one == -1: seg = seg[::-1]; na, nb = nds[b], nds[a]
            else: na, nb = nds[a], nds[b]
            if not any(0 <= x < W and 0 <= y < H and play[int(y), int(x)] for x, y in seg[:: max(1, len(seg) // 6)]): continue
            q = cv2.approxPolyDP(seg.astype(np.float32).reshape(-1, 1, 2), 0.8, False)[:, 0, :] if len(seg) > 2 else seg.astype(np.float32)
            L = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
            if L < 2: continue
            ia, ib = nid_of(na, seg[0]), nid_of(nb, seg[-1])
            sig = 0
            for k_, s_ in ((0, 1), (len(nds) - 1, 2)):
                pass
            # світлофор біля кінця ребра: вузол-світлофор на цьому ребрі в межах 30 пікс від кінця
            idx = list(range(a, b + 1)) if one != -1 else list(range(b, a - 1, -1))
            acc = 0.0; arc = [0.0]
            for k in range(1, len(idx)): acc += math.hypot(*(P[idx[k]] - P[idx[k - 1]])); arc.append(acc)
            for k, ii in enumerate(idx):
                if nds[ii] in sig_nodes:
                    if arc[k] < 15 / CD.MPP: sig |= 1
                    if L - arc[k] < 15 / CD.MPP: sig |= 2
            edges.append(dict(a=ia, b=ib, q=[list(map(float, v)) for v in q], cls=CLS[h], lanes=min(nl, 6), one=1 if one else 0,
                              w=CD.ROAD_W[h] / CD.MPP, L=L, sig=sig, t=t))
    edges, nodes = contract(edges, nodes)
    traf_flags(D, edges, W, H)
    def inside(x, y): return 0 <= int(y) < H and 0 <= int(x) < W and bool(play[int(y), int(x)])
    deg = [0] * len(nodes)
    for e in edges: deg[e['a']] += 1; deg[e['b']] += 1
    R = 30 / CD.MPP
    exits = []
    for k, (x, y) in enumerate(nodes):                                   # кінці за межею зони або обрізані біля неї
        ex = not inside(x, y)
        if not ex and deg[k] == 1:
            ex = any(not inside(x + R * math.cos(t), y + R * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False))
        exits.append(1 if ex else 0)
    print('  виїздів за межу зони: %d' % sum(exits))
    dead_ends(edges, len(nodes), exits)
    nrad = [0.0] * len(nodes)                                          # радіус перехрестя: половина найширшої дороги + 1,5 м
    for e in edges:
        for nd in (e['a'], e['b']): nrad[nd] = max(nrad[nd], e['w'] / 2 + 1.5 / CD.MPP)
    for e in edges:
        e['p0'] = len(pts); e['n'] = len(e['q'])
        pts.extend(np.round(np.array(e['q'])).astype(int).tolist())
    # суміжність: з вузла — ребра, якими можна виїхати (dir 0 — від a до b, 1 — від b до a, лише двосторонні)
    adj = [[] for _ in nodes]
    for i, e in enumerate(edges):
        adj[e['a']].append(i * 2)
        if not e['one']: adj[e['b']].append(i * 2 + 1)
    # сітка ребер 128 пікс для пошуку
    G = 128; gw, gh = (W + G - 1) // G, (H + G - 1) // G
    cells = [[] for _ in range(gw * gh)]
    for i, e in enumerate(edges):
        q = np.array(pts[e['p0']:e['p0'] + e['n']])
        m = int(e['w'] / 2 + 8)
        x0, y0 = (q.min(0) - m) // G; x1, y1 = (q.max(0) + m) // G
        for cy in range(max(0, y0), min(gh, y1 + 1)):
            for cx in range(max(0, x0), min(gw, x1 + 1)): cells[cy * gw + cx].append(i)
    print('граф доріг: вузлів %d, ребер %d, точок %d, основних ребер %d, з трафіком %d (%.0f км), ребер зі світлофором %d' % (
        len(nodes), len(edges), len(pts), sum(e['cls'] <= MAIN for e in edges), sum(e['traf'] for e in edges),
        sum(e['L'] for e in edges if e['traf']) * CD.MPP / 1000, sum(e['sig'] > 0 for e in edges)))
    return dict(nodes=nodes, edges=edges, pts=pts, adj=adj, G=G, gw=gw, gh=gh, cells=cells, nrad=nrad, exits=exits)


def contract(edges, nodes):
    """Складні перехрестя з OSM: короткі ділянки (< 12 м) між двома перехрестями — злити в один вузол (центр),
    кінці сусідніх ребер — у цей центр. Трафік не перескакує купу дрібних ребер (ривки), повертає одним рухом."""
    deg = [0] * len(nodes)
    for e in edges: deg[e['a']] += 1; deg[e['b']] += 1
    par = list(range(len(nodes)))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    short = set()
    for i, e in enumerate(edges):
        if e['L'] * CD.MPP < 12 and e['a'] != e['b'] and deg[e['a']] >= 3 and deg[e['b']] >= 3 and e['cls'] <= 3:
            short.add(i); par[f(e['a'])] = f(e['b'])
    groups = {}
    for n in range(len(nodes)): groups.setdefault(f(n), []).append(n)
    new_id, new_nodes = {}, []
    for r, mem in groups.items():
        k = len(new_nodes); new_nodes.append(tuple(np.mean([nodes[m] for m in mem], 0)))
        for m in mem: new_id[m] = k
    out = []
    for i, e in enumerate(edges):
        if i in short: continue
        a, b = new_id[e['a']], new_id[e['b']]
        q = e['q']
        if a == b and e['L'] * CD.MPP < 30: continue
        if tuple(nodes[e['a']]) != new_nodes[a]: q = [list(new_nodes[a])] + q
        if tuple(nodes[e['b']]) != new_nodes[b]: q = q + [list(new_nodes[b])]
        e = dict(e, a=a, b=b, q=q)
        e['L'] = float(np.sum(np.hypot(*np.diff(np.array(q), axis=0).T)))
        out.append(e)
    print('  злито коротких ділянок на перехрестях: %d (вузлів %d -> %d)' % (len(short), len(nodes), len(new_nodes)))
    return out, new_nodes


def traf_flags(D, edges, W, H):
    """Трафік: основні дороги й житлові вулиці з твердим покриттям; не в дачних масивах, гаражах, садах, полях."""
    K = CD.K; S = 4.0 / CD.MPP                                         # растр 4 м
    w_, h_ = int(W / S) + 1, int(H / S) + 1
    bad = np.zeros((h_, w_), np.uint8)
    for w in D['w']:
        if w['t'].get('landuse') in ('allotments', 'garages', 'orchard', 'farmland') and len(w['pts']) >= 3:
            cv2.fillPoly(bad, [np.round(np.array(w['pts']) * K / S).astype(np.int32)], 1)
    UNPAVED = {'ground', 'unpaved', 'gravel', 'dirt', 'compacted', 'pebblestone', 'fine_gravel', 'sand', 'grass', 'mud'}
    for e in edges:
        t = e['t']
        ok = e['cls'] <= MAIN or (e['cls'] == 3 and t.get('surface') not in UNPAVED and t.get('tracktype') is None)
        if ok and e['cls'] == 3:
            q = np.array(e['q']) / S
            inside = [bad[int(y), int(x)] for x, y in q if 0 <= x < w_ and 0 <= y < h_]
            if inside and np.mean(inside) > 0.5: ok = False
        e['traf'] = 1 if ok else 0


def dead_ends(edges, nn, exits):
    """Тупики для трафіку: ребро-напрям «мертве», якщо далі (без розвороту) лише мертві або нічого. Біт 0 — dir 0, біт 1 — dir 1."""
    out = [[] for _ in range(nn)]                                     # вузол -> (ребро, напрям) виїзду трафіком
    for i, e in enumerate(edges):
        if not e['traf']: continue
        out[e['a']].append((i, 0))
        if not e['one']: out[e['b']].append((i, 1))
    dead = {}
    for i, e in enumerate(edges):
        for d in ((0,) if e['one'] else (0, 1)):
            if e['traf']: dead[(i, d)] = False
    changed = True
    while changed:
        changed = False
        for (i, d), v in dead.items():
            if v: continue
            e = edges[i]; end = e['b'] if d == 0 else e['a']
            if exits[end]: continue                                   # виїзд за межу зони — не тупик (машина «їде далі» в темряву)
            onward = [k for k in out[end] if k[0] != i]
            if not onward or all(dead[k] for k in onward):
                dead[(i, d)] = True; changed = True
    for i, e in enumerate(edges):
        e['dead'] = (1 if dead.get((i, 0)) else 0) | (2 if dead.get((i, 1)) else 0)
    print('  трафік: тупикових напрямів %d з %d' % (sum(dead.values()), len(dead)))


def write_c(f, g):
    nodes, edges, pts, adj, cells = g['nodes'], g['edges'], g['pts'], g['adj'], g['cells']
    assert len(edges) < 32768 and len(pts) < 2 ** 31
    f.write('/* дорожній граф (tools/roadgraph.py) */\n')
    f.write('typedef struct { uint16_t a, b; uint32_t p0; uint16_t n; uint8_t cls, lanes, one, sig; uint8_t w2; uint16_t len; uint8_t traf, dead; } REdge;\n')
    f.write('#define RG_MAIN %d\n#define RG_G %d\n#define RG_GW %d\n#define RG_GH %d\n#define RG_NN %d\n#define RG_NE %d\n' % (MAIN, g['G'], g['gw'], g['gh'], len(nodes), len(edges)))
    f.write('static const int16_t RG_NODE[][2] = {%s};\n' % (','.join('{%d,%d}' % (int(round(x)), int(round(y))) for x, y in nodes) or '{0,0}'))
    f.write('static const REdge RG_EDGE[] = {%s};\n' % (','.join('{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}' % (
        e['a'], e['b'], e['p0'], e['n'], e['cls'], e['lanes'], e['one'], e['sig'], min(255, int(round(e['w']))), min(65535, int(round(e['L']))),
        e['traf'], e['dead']) for e in edges) or '{0}'))
    f.write('static const uint8_t RG_NODE_R[] = {%s};\n' % (','.join(str(min(255, int(round(r)))) for r in g['nrad']) or '0'))
    f.write('static const uint8_t RG_NODE_X[] = {%s};\n' % (','.join(map(str, g['exits'])) or '0'))
    f.write('static const int16_t RG_PT[][2] = {%s};\n' % (','.join('{%d,%d}' % (x, y) for x, y in pts) or '{0,0}'))
    st = [0]; flat = []
    for a in adj: flat.extend(a); st.append(len(flat))
    f.write('static const uint32_t RG_ADJ0[] = {%s};\nstatic const uint16_t RG_ADJ[] = {%s};\n' % (','.join(map(str, st)), ','.join(map(str, flat)) or '0'))
    st = [0]; flat = []
    for c in cells: flat.extend(c); st.append(len(flat))
    f.write('static const uint32_t RG_CELL0[] = {%s};\nstatic const uint16_t RG_CELL[] = {%s};\n' % (','.join(map(str, st)), ','.join(map(str, flat)) or '0'))
