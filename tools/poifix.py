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
    F = 0.5 / md['M']                                                      # старі пікселі 0,5 м -> пікселі карти
    # проїжджа частина: осі доріг з шириною смуг; усе — у сітці 2×2 пікс (пам'ять: карта 1,8× велика)
    Q = 2; h2, w2 = H // Q, W // Q
    road = np.zeros((h2, w2), np.uint8)
    for x0, y0, x1, y1, hw, ow in md['lanes']:
        cv2.line(road, (x0 // Q, y0 // Q), (x1 // Q, y1 // Q), 1, max(1, max(int(10 * F), int(2 * hw * (1 if ow else 2))) // Q))
    free = np.zeros((h2, w2), np.uint8)
    for y in range(0, h2 * Q, 2048):
        ye = min(h2 * Q, y + 2048)
        so = np.asarray(solid[y:ye, :w2 * Q]).reshape((ye - y) // Q, Q, w2, Q).any((1, 3))
        pl = np.asarray(play[y:ye, :w2 * Q]).reshape((ye - y) // Q, Q, w2, Q).all((1, 3))
        free[y // Q:ye // Q] = (road[y // Q:ye // Q] > 0) & ~so & pl
    del road
    free = cv2.erode(free, np.ones((max(3, int(5 * F / Q) | 1),) * 2, np.uint8))     # не впритул до стін
    # головна мережа: найбільша зв'язна компонента
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ok = lab == big
    del lab, free
    print('доріг для точок: %.2f км²' % (ok.sum() * (Q * md['M']) ** 2 / 1e6))

    def snap(x, y, R=160):
        x, y, R = int(x) // Q, int(y) // Q, int(R * F) // Q
        xa, ya = max(0, x - R), max(0, y - R)
        ys, xs = np.nonzero(ok[ya:y + R + 1, xa:x + R + 1])
        if not len(xs): return None
        d = (xs + xa - x) ** 2 + (ys + ya - y) ** 2; k = int(d.argmin())
        return int(xs[k] + xa) * Q + Q // 2, int(ys[k] + ya) * Q + Q // 2, float(np.sqrt(d[k])) * Q

    for key in ('shops', 'addrs', 'sto', 'fuel'):
        out, moved, lost = [], 0, 0
        for e in md[key]:
            s = snap(e['x'], e['y'], 400 if key in ('sto', 'fuel') else 160)
            if s is None: lost += 1; continue
            if s[2] > 2: moved += 1
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
