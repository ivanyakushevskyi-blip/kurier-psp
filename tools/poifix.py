"""Після citybuild: точки замовлень/адрес/СТО/АЗС/старту — лише на проїжджій частині головної мережі доріг
(не в парках і кущах), і контури будинків над арками (до 64 вершин). Переписує build/mapdata.json.
python3 -I tools/poifix.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2


def main():
    md = json.load(open('build/mapdata.json'))
    solid = np.load('build/solid.npy', mmap_mode='r'); play = np.load('build/play.npy', mmap_mode='r')
    H, W = solid.shape
    # проїжджа частина: осі доріг з шириною смуг
    road = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1, hw, ow in md['lanes']:
        cv2.line(road, (x0, y0), (x1, y1), 1, max(10, int(2 * hw * (1 if ow else 2))))
    free = np.zeros((H, W), np.uint8)
    for y in range(0, H, 2048):
        free[y:y + 2048] = (road[y:y + 2048] > 0) & ~np.asarray(solid[y:y + 2048]) & np.asarray(play[y:y + 2048])
    del road
    free = cv2.erode(free, np.ones((5, 5), np.uint8))                       # не впритул до стін
    # головна мережа: найбільша зв'язна компонента
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ok = lab == big
    del lab, free
    print('доріг для точок: %.2f км²' % (ok.sum() * md['M'] ** 2 / 1e6))

    def snap(x, y, R=160):
        x, y = int(x), int(y)
        xa, ya = max(0, x - R), max(0, y - R)
        ys, xs = np.nonzero(ok[ya:y + R + 1, xa:x + R + 1])
        if not len(xs): return None
        d = (xs + xa - x) ** 2 + (ys + ya - y) ** 2; k = int(d.argmin())
        return int(xs[k] + xa), int(ys[k] + ya), float(np.sqrt(d[k]))

    for key in ('shops', 'addrs', 'sto', 'fuel'):
        out, moved, lost = [], 0, 0
        for e in md[key]:
            s = snap(e['x'], e['y'], 400 if key in ('sto', 'fuel') else 160)
            if s is None: lost += 1; continue
            if s[2] > 0.5: moved += 1
            e['x'], e['y'] = s[0], s[1]; out.append(e)
        md[key] = out
        print('%-6s %5d   зсунуто %d, без дороги поруч (прибрано) %d' % (key, len(out), moved, lost))
    s = snap(*md['spawn'], R=400)
    if s: md['spawn'] = [s[0], s[1]]
    # арки: контур будинку повністю (до 64 вершин)
    import citydraw as CD, citybuild as CB
    D = CD.data(); K = CD.K
    ar = []
    for w in D['w']:
        t = w['t']
        if t.get('highway') in CD.ROAD_W and (t.get('tunnel') == 'building_passage' or t.get('covered') == 'yes'):
            P = np.round(np.array(w['pts']) * K).astype(np.int32)
            for b in D['B'] + D['SAT']:
                Q = np.array(b['q'], np.float32) * K
                if any(cv2.pointPolygonTest(Q, (float(x), float(y)), False) >= 0 for x, y in P.tolist()):
                    ar.append(CB.arch_poly(Q))
    md['arches'] = ar
    print('арки', len(ar), 'вершин max', max((len(q) for q in ar), default=0))
    json.dump(md, open('build/mapdata.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
