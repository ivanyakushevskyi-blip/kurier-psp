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
    mp = np.load('build/map.npy', mmap_mode='r')
    WATER = json.load(open('build/water_idx.json'))
    # головна мережа проїзду: сітка B×B пікс (2 м), клітинка вільна, якщо вільна вся; найбільша зв'язна компонента
    B = 4
    hb, wb = H // B, W // B
    free = np.zeros((hb, wb), bool)
    for y0 in range(0, hb * B, 1024):
        y1 = min(hb * B, y0 + 1024)
        f = ~np.asarray(solid[y0:y1, :wb * B]) & np.asarray(play[y0:y1, :wb * B]) & ~np.isin(np.asarray(mp[y0:y1, :wb * B]), WATER)
        free[y0 // B:y1 // B] = f.reshape((y1 - y0) // B, B, wb, B).all((1, 3))
    n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    NET = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    del lab
    print('мережа проїзду: %.1f км²' % (NET.sum() * (B * CD.MPP) ** 2 / 1e6))
    L = json.load(open('data/landmarks.json'))
    ids = [e['id'] for e in L]; assert len(ids) == len(set(ids))
    out = []
    for e in L:
        cands = []
        if e.get('pos'):                                       # точка без назви в OSM (водонапірна вежа)
            cands.append(({}, CG.px(*e['pos'])))
        for nid, t in D['n'].items() if e['osm'] else ():
            if t.get('name') == e['osm'] and nid in D['nodes']:
                cands.append((t, CG.px(*D['nodes'][nid])))
        for w in (D['w'] if e['osm'] else ()):
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
        R = 600
        xa, ya = max(0, x - R), max(0, y - R)
        sub = ~np.asarray(solid[ya:y + R, xa:x + R]) & np.asarray(play[ya:y + R, xa:x + R])
        sub &= ~np.isin(np.asarray(mp[ya:y + R, xa:x + R]), WATER)                         # вода — теж не проїзд
        sub = cv2.erode(sub.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)      # місце, де вміщається машина
        # лише суша, зв'язана з головною мережею проїзду, — не острови й замкнені двори
        sub &= np.repeat(np.repeat(NET[ya // B:(y + R) // B + 1, xa // B:(x + R) // B + 1], B, 0), B, 1)[(ya % B):(ya % B) + sub.shape[0], (xa % B):(xa % B) + sub.shape[1]]
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
