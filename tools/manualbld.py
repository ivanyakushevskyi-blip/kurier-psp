"""Будівлі, яких немає в OSM і які автопошук не бере: задані вручну за знімком (data/manual_buildings.json).
Кожна — прямокутник-підказка в сітці 0,32 м; контур уточнюється GrabCut по знімку, або задається полігоном.
Пише build/manual_buildings.json у форматі buildings.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

SRC = 0.32


def outline(rect, sat, pad=30):
    x0, y0, x1, y1 = rect
    X0, Y0 = x0 - pad, y0 - pad
    a = np.ascontiguousarray(sat[Y0:y1 + pad, X0:x1 + pad])
    m = np.zeros(a.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(a, m, (pad, pad, x1 - x0, y1 - y0), bg, fg, 6, cv2.GC_INIT_WITH_RECT)
    f = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None, a
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 3.0, True)[:, 0, :].astype(float) + [X0, Y0]
    return poly, a


def build(src='data/manual_buildings.json', out='build/manual_buildings.json', debug=None):
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    items = json.load(open(src))
    res = []
    off = np.array(json.load(open('build/sat_map.json')).get('off', [0, 0]))
    for it in items:
        o = off if it.get('grid', 'v1') == 'v1' else np.zeros(2, int)       # координати у старій сітці -> нова
        if 'poly' in it: P = np.array(it['poly'], float) + o
        else:
            P, _ = outline([it['rect'][0] + o[0], it['rect'][1] + o[1], it['rect'][2] + o[0], it['rect'][3] + o[1]], sat)
            if P is None: print('  no outline', it.get('name')); continue
        Pi = np.round(P).astype(np.int32)
        mk = np.zeros((int(Pi[:, 1].max()) - int(Pi[:, 1].min()) + 3, int(Pi[:, 0].max()) - int(Pi[:, 0].min()) + 3), np.uint8)
        cv2.fillPoly(mk, [Pi - Pi.min(0) + 1], 1)
        ys, xs = np.nonzero(mk)
        px = np.asarray(sat[ys + Pi[:, 1].min() - 1, xs + Pi[:, 0].min() - 1])[..., ::-1].astype(float)
        lit = px[px.max(1) > 70]
        roof = it.get('roof') or (np.median(lit if len(lit) > 10 else px, 0).astype(int).tolist())
        q = [tuple(p) for p in P.tolist()] + [tuple(P[0].tolist())]
        A = cv2.contourArea(P.astype(np.float32)) * SRC * SRC
        res.append(dict(q=q, t={'building': it.get('building', 'yes'), 'name': it.get('name', ''), 'src': 'manual'},
                        area=round(A, 1), roof=roof, h=it.get('h', 12.0), h_src='manual', detail=True))
        print('  %-30s %.0f m²' % (it.get('name', '?'), A))
    json.dump(res, open(out, 'w'), ensure_ascii=False)
    if debug:
        for k, r in enumerate(res):
            P = np.array(r['q']); c = P.mean(0).astype(int)
            x0, y0 = c[0] - 260, c[1] - 260
            a = np.ascontiguousarray(sat[y0:y0 + 520, x0:x0 + 520]).copy()
            cv2.polylines(a, [np.round(P - [x0, y0]).astype(np.int32)], True, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(debug, 'man%02d.jpg' % k), a)
    return res


if __name__ == '__main__':
    build(debug=sys.argv[1] if len(sys.argv) > 1 else None)
