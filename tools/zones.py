"""Зони гри (08.11.2026, узгоджено з Іваном — docs/design.md «Прогрес і зони місій»).
- етапи відкриття районів 1..4 (растр, крок 16 пікс ≈ 4,4 м): райони — Вороного за place=suburb/neighbourhood з OSM,
  центр і південь розділяє залізниця (усе над колією, зв'язане з Центром, — етап 3, під колією — 4);
- зона місій: ≤150 м від головних вулиць і не приватний сектор, або ≤40 м від головної, або центр/Виставка повністю;
- проріджування закладів у центрі (≈ вдвічі): мережі й заклади з повнішими даними OSM, ≥90 м між закладами одного типу.
build(D, md, W, H, play) -> dict(shops, addrs, zone, G) — shops/addrs з полем st (етап), зона для трафіку."""
import numpy as np, cv2
import citydraw as CD, citygen as CG
import roadgraph as RG

F = 16
STAGE = {
    1: ['Озерна', 'Приозерний ЖК', 'Перший парковий', 'Дендропарковий', 'Відрадне', 'Дивокрай', 'Обрій'],
    2: ['Заріччя', 'Голлівуд', 'Грузевиця-2', 'Виставка'],
    3: ['Гречани', 'Дальні Гречани', 'Садиби'],
}
CENTRE = ['Центр', 'Старе місто', 'Поділ', 'Завалля']            # 3 над колією, 4 під колією (як і південь)
FULL = CENTRE + ['Виставка']                                      # усі адреси — без обмеження «до 150 м від головної»
STAGE_NAME = ['', 'ОЗЕРНА', 'ЗАРІЧЧЯ І ВИСТАВКА', 'ГРЕЧАНИ І ЦЕНТР', 'ПІВДЕНЬ: ДУБОВЕ, КАВКАЗ, ПЛАН']


def stages(D, W, H, play):
    K = CD.K
    ow, oh = W // F + 1, H // F + 1
    pl = np.zeros((oh, ow), bool); p_ = np.asarray(play[::F, ::F]); pl[:p_.shape[0], :p_.shape[1]] = p_
    P = []
    for nid, t in D['n'].items():
        if t.get('place') in ('suburb', 'neighbourhood') and nid in D['nodes']:
            x, y = CG.px(*D['nodes'][nid]); x, y = x * K / F, y * K / F
            if 0 <= x < ow and 0 <= y < oh and pl[int(y), int(x)]: P.append((t['name'], x, y))
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    lab = np.argmin(np.stack([(xx - x) ** 2 + (yy - y) ** 2 for _, x, y in P]), 0)
    nm = np.array([n for n, _, _ in P], object)[lab]
    st = np.full((oh, ow), 4, np.uint8)
    for s, ns in STAGE.items(): st[np.isin(nm, ns)] = s
    # центр і південь: межа — залізниця
    import rail
    bar = np.zeros((oh, ow), np.uint8)
    for R in rail.lines(D, play): cv2.polylines(bar, [np.round(R / F).astype(np.int32)], False, 1, 3)
    cs = (st == 4) & (bar == 0)
    n_, cc = cv2.connectedComponents(cs.astype(np.uint8))
    cx, cy = next((int(x), int(y)) for n, x, y in P if n == 'Центр')
    north = cc == cc[cy, cx]
    st[north] = 3
    full = np.isin(nm, ['Виставка']) | (north & np.isin(nm, CENTRE))
    return st, full, pl


def zone(D, W, H, full):
    """зона місій на тій самій сітці"""
    K = CD.K; ow, oh = full.shape[1], full.shape[0]
    main = np.zeros((oh, ow), np.uint8)
    for w in D['w']:
        h = w['t'].get('highway')
        if h not in RG.CLS or RG.CLS[h] > RG.MAIN or not CD.drivable(w['t']) or len(w['pts']) < 2: continue
        cv2.polylines(main, [np.round(np.array(w['pts']) * K / F).astype(np.int32)], False, 1, 1)
    house = np.zeros((oh, ow), np.float32); allb = np.zeros((oh, ow), np.float32)
    for b in D['B'] + D.get('SAT', []):
        q = np.array(b['q']) * K / F; cx, cy = q.mean(0); a = b.get('area', 100)
        if not (0 <= cx < ow and 0 <= cy < oh): continue
        bt = b['t'].get('building', 'yes')
        small = bt in ('house', 'detached', 'semidetached_house', 'bungalow', 'shed', 'garage', 'garages', 'hut', 'cabin') \
            or (b['t'].get('src') == 'ms' and a < 220) or (bt == 'yes' and a < 160)
        allb[int(cy), int(cx)] += a; house[int(cy), int(cx)] += a * small
    k = int(round(66 / (F * CD.MPP))) | 1                                           # вікно ≈ 66 м
    hs = cv2.boxFilter(house, -1, (k, k), normalize=False); al = cv2.boxFilter(allb, -1, (k, k), normalize=False)
    priv = ((hs / np.maximum(al, 1)) > 0.6) & (al > 0)
    priv = cv2.morphologyEx(priv.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    dm = cv2.distanceTransform(1 - main, cv2.DIST_L2, 5) * F * CD.MPP             # м до головної
    return ((dm < 150) & ~priv) | (dm < 40) | full


def _shop_scores(D):
    """популярність за даними OSM: мережа (назва трапляється ≥2 разів або brand), сайт, години, телефон"""
    import collections
    tags = collections.defaultdict(int); cnt = collections.Counter()
    items = [t for t in D['n'].values()] + [w['t'] for w in D['w'] if w.get('closed')]
    for t in items:
        n = t.get('name')
        if not n or not (t.get('shop') or t.get('amenity')): continue
        n = n.strip().upper(); cnt[n] += 1
        s = 3 * ('brand' in t or 'brand:wikidata' in t) + ('website' in t or 'contact:website' in t) + ('opening_hours' in t) \
            + ('phone' in t or 'contact:phone' in t) + ('contact:facebook' in t or 'contact:instagram' in t)
        tags[n] = max(tags[n], s)
    return {n: tags[n] + (3 if cnt[n] >= 2 else 0) for n in tags}


def build(D, md, W, H, play):
    st, full, pl = stages(D, W, H, play)
    z = zone(D, W, H, full)
    oh, ow = st.shape
    def cell(p): return min(oh - 1, max(0, int(p['y']) // F)), min(ow - 1, max(0, int(p['x']) // F))
    addrs = []
    for a in md['addrs']:
        c = cell(a)
        if z[c]: addrs.append(dict(a, st=int(st[c])))
    sc = _shop_scores(D)
    shops, cen = [], []
    for s in md['shops']:
        c = cell(s)
        if not z[c]: continue
        s = dict(s, st=int(st[c]))
        (cen if full[c] and s['st'] == 3 else shops).append(s)
    # центр: лишити ≈ половину — спершу популярніші, не ближче 90 м до закладу того самого типу
    rs = np.random.RandomState(7)
    cen.sort(key=lambda s: (-sc.get(s['name'].strip().upper(), 0), rs.rand()))
    keep, lim, md_ = [], (len(cen) + 1) // 2, 90 / CD.MPP
    for s in cen:
        if len(keep) >= lim: break
        if any(k['kind'] == s['kind'] and abs(k['x'] - s['x']) + abs(k['y'] - s['y']) < md_ * 1.3 and
               np.hypot(k['x'] - s['x'], k['y'] - s['y']) < md_ for k in keep): continue
        keep.append(s)
    shops += keep
    for k in range(1, 5):
        print('  етап %d %s: закладів %d, адрес %d' % (k, STAGE_NAME[k], sum(s['st'] == k for s in shops), sum(a['st'] == k for a in addrs)))
    print('  зона місій: адрес %d з %d, закладів %d з %d (центр %d -> %d)' % (len(addrs), len(md['addrs']), len(shops), len(md['shops']), len(cen), len(keep)))
    # старт етапу 1 — заклад Озерної, найближчий до точки району
    K = CD.K; ozn = next(((CG.px(*D['nodes'][i])) for i, t in D['n'].items() if t.get('name') == 'Озерна' and t.get('place') and i in D['nodes']), None)
    sp1 = md['spawn']
    if ozn:
        ox, oy = ozn[0] * K, ozn[1] * K
        s1 = [s for s in shops if s['st'] == 1] or [a for a in addrs if a['st'] == 1]
        if s1: b = min(s1, key=lambda s: (s['x'] - ox) ** 2 + (s['y'] - oy) ** 2); sp1 = [b['x'], b['y']]
    return dict(shops=shops, addrs=addrs, zone=z, st=st, spawn1=sp1)
