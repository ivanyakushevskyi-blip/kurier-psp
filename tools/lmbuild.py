"""Пам'ятки: data/landmarks.json + OSM -> build/landmarks.json (позиції в сітці гри, радіуси відкриття, колір бань).
Порядок записів у data/landmarks.json = номер біта в збереженні: нові пам'ятки лише ДОПИСУВАТИ в кінець.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2


def dome_of(rgb):
    if rgb is None: return 'gold'
    r, g, b = rgb
    if b > r + 15 and b > g: return 'blue'
    if g > r + 8 and g > b: return 'green'
    return 'gold'


def main():
    import citydraw as CD, citygen as CG
    D = CD.data()
    K = CD.K
    solid = np.load('build/solid.npy', mmap_mode='r'); play = np.load('build/play.npy', mmap_mode='r')
    H, W = solid.shape
    L = json.load(open('data/landmarks.json'))
    ids = [e['id'] for e in L]; assert len(ids) == len(set(ids))
    out = []
    for e in L:
        cands = []
        for nid, t in D['n'].items():
            if t.get('name') == e['osm'] and nid in D['nodes']:
                cands.append((t, CG.px(*D['nodes'][nid])))
        for w in D['w']:
            if w['t'].get('name') == e['osm']:
                cands.append((w['t'], tuple(np.array(w['pts']).mean(0))))
        if e.get('osm_tag'):
            k, v = e['osm_tag'].split('=')
            cands = [c for c in cands if c[0].get(k) == v]
        cands = [(t, (x * K, y * K)) for t, (x, y) in cands if 0 <= x * K < W and 0 <= y * K < H and play[int(y * K), int(x * K)]]
        if not cands:
            print('  !! не знайдено в зоні:', e['id'], e['osm']); continue
        t, (x, y) = cands[0]
        x, y = int(round(x)), int(round(y))
        # найближча точка, куди може доїхати машина (не стіна, в зоні)
        R = 400
        xa, ya = max(0, x - R), max(0, y - R)
        sub = ~np.asarray(solid[ya:y + R, xa:x + R]) & np.asarray(play[ya:y + R, xa:x + R])
        sub = cv2.erode(sub.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)      # місце, де вміщається машина
        ys, xs = np.nonzero(sub)
        d0 = float(np.sqrt(((xs + xa - x) ** 2 + (ys + ya - y) ** 2).min())) if len(xs) else 999.0
        r = int(max(90, d0 + 50))
        # колір бань — за дахом будинку під позначкою
        roof = None
        for b in D['B'] + D['SAT']:
            P = np.array(b['q'], np.float32) * K
            if cv2.pointPolygonTest(P, (float(x), float(y)), True) > -6:
                roof = b.get('roof'); break
        name = e['name'].upper().replace('«', '"').replace('»', '"').replace('’', "'").replace('ʼ', "'")
        out.append(dict(e, x=x, y=y, r=r, d0=round(d0), dome=dome_of(roof), label=name, bit=ids.index(e['id'])))
        if d0 > 200: print('  ! далеко від проїзду: %s %.0f м' % (e['id'], d0 * CD.MPP))
    json.dump(out, open('build/landmarks.json', 'w'), ensure_ascii=False, indent=0)
    print('landmarks', len(out), '/', len(L))


if __name__ == '__main__':
    main()
