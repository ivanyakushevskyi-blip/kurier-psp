"""Пошук дахів на знімку там, де OSM будинків не має (приватний сектор).
Ознаки даху: однорідна пляма не-зеленого кольору, майже прямокутна, 12–400 м²,
і поруч з нею з північного боку — тінь (будинок має висоту, плитка/асфальт — ні).
Повертає список будинків у форматі buildings.json (q — у сітці 0,32 м)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

SRC = 0.32


def detect(sx0, sy0, sw, sh, osm_mask=None, road_mask=None, debug=None):
    """sx0..: вікно в сітці 0,32 м. osm_mask/road_mask — того ж розміру (bool)."""
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    a = np.ascontiguousarray(sat[sy0:sy0 + sh, sx0:sx0 + sw])
    seg = cv2.pyrMeanShiftFiltering(a, 4, 16)
    rgb = seg[..., ::-1].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); exg = 2 * g - r - b
    lum = rgb.mean(-1)
    dark = cv2.GaussianBlur(a, (0, 0), 1.5).max(-1) < 64
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)), (0, 0), 2)
    ok = (exg < 10) & ~dark & (tex < 14)
    if osm_mask is not None: ok &= ~cv2.dilate(osm_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if road_mask is not None: ok &= ~road_mask
    # регіони однакового кольору: квантування + зв'язні компоненти
    q = (np.round(rgb / 14)).astype(np.int32)
    key = q[..., 0] * 10000 + q[..., 1] * 100 + q[..., 2]
    key[~ok] = -1
    out = []
    used = np.zeros(key.shape, bool)
    for kv in np.unique(key):
        if kv < 0: continue
        m = (key == kv).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 4)
        for i in range(1, n):
            A = st[i, cv2.CC_STAT_AREA] * SRC * SRC
            if A < 12 or A > 6000: continue
            ys, xs = np.nonzero(lab == i)
            pts = np.c_[xs, ys].astype(np.float32)
            rect = cv2.minAreaRect(pts); (rw, rh) = rect[1]
            if min(rw, rh) < 7: continue                                  # < 2,2 м
            sol = len(xs) / max(rw * rh, 1)
            asp = max(rw, rh) / max(min(rw, rh), 1)
            big = A > 400
            if sol < (0.8 if big else 0.72) or asp > (6 if big else 4): continue
            c = rgb[ys, xs].mean(0)
            warm = c[0] > c[2] + 18
            # тінь з північного боку
            x0, x1 = xs.min(), xs.max(); y0 = ys.min()
            band = dark[max(0, y0 - 14):max(0, y0 - 2), x0:x1 + 1]
            shf = band.mean() if band.size else 0
            if big:
                if shf < 0.35: continue
            elif not (shf > 0.3 or (warm and sol > 0.82 and shf > 0.1)): continue
            if used[ys, xs].mean() > 0.3: continue
            used[ys, xs] = True
            box = cv2.boxPoints(rect)
            q_ = [(float(x + sx0), float(y + sy0)) for x, y in box] + [(float(box[0][0] + sx0), float(box[0][1] + sy0))]
            out.append(dict(q=q_, t={'building': 'yes' if big else 'house', 'src': 'sat'}, area=round(A, 1),
                            roof=[int(c[0]), int(c[1]), int(c[2])], h=12.0 if big else 5.0, h_src='sat', detail=True))
    if debug is not None:
        v = a.copy()
        for o in out:
            P = np.array(o['q'], np.float32) - [sx0, sy0]
            cv2.polylines(v, [np.round(P).astype(np.int32)], True, (255, 0, 255), 1)
        cv2.imwrite(debug, v)
    return out




def detect_all(out='build/sat_houses.json', B=1600, M=60):
    import citygen as CG
    from PIL import Image, ImageDraw
    meta = json.load(open('build/sat_map.json')); W, H = meta['W'], meta['H']
    Bl = json.load(open('build/buildings.json'))['buildings']
    _, _, ways = CG.load('data/osm_v2')
    res = []
    for y0 in range(0, H, B):
        for x0 in range(0, W, B):
            sx0, sy0 = max(0, x0 - M), max(0, y0 - M)
            sw, sh = min(W, x0 + B + M) - sx0, min(H, y0 + B + M) - sy0
            om = Image.new('L', (sw, sh), 0); d = ImageDraw.Draw(om)
            for b in Bl:
                P = [(x - sx0, y - sy0) for x, y in b['q']]
                if max(p[0] for p in P) < 0 or max(p[1] for p in P) < 0 or min(p[0] for p in P) > sw or min(p[1] for p in P) > sh: continue
                d.polygon(P, fill=1)
            rm = Image.new('L', (sw, sh), 0); dr = ImageDraw.Draw(rm)
            for w in ways:
                h = w['t'].get('highway')
                if not h or w['t'].get('area') == 'yes': continue
                P = [(x - sx0, y - sy0) for x, y in w['pts']]
                if max(p[0] for p in P) < 0 or max(p[1] for p in P) < 0 or min(p[0] for p in P) > sw or min(p[1] for p in P) > sh: continue
                wd = {'primary': 16, 'secondary': 14, 'tertiary': 12, 'residential': 9, 'unclassified': 8, 'service': 6}.get(h, 3)
                dr.line(P, fill=1, width=int(wd / SRC))
            r = detect(sx0, sy0, sw, sh, np.array(om, bool), np.array(rm, bool))
            for o in r:
                c = np.array(o['q']).mean(0)
                if x0 <= c[0] < x0 + B and y0 <= c[1] < y0 + B: res.append(o)
            print('  ', x0, y0, len(res), flush=True)
    json.dump(res, open(out, 'w'))
    print('sat houses', len(res))


if __name__ == '__main__':
    if sys.argv[1:] and sys.argv[1] == 'all': detect_all()
    else:
        sx0, sy0, sw, sh = map(int, sys.argv[1:5])
        print(len(detect(sx0, sy0, sw, sh, debug=sys.argv[5])), 'дахів')


def detect_big(sx0, sy0, sw, sh, osm_mask, road_mask, debug=None):
    """Великі будівлі, яких немає в OSM: «забудована» маска (не зелень, не тінь, гладка), без OSM і доріг,
    зв'язні області 300–8000 м² з тінню з північного боку. Контур — спрощений полігон."""
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    a = np.ascontiguousarray(sat[sy0:sy0 + sh, sx0:sx0 + sw])
    sm = cv2.bilateralFilter(a, 7, 30, 5)
    rgb = sm[..., ::-1].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b
    dark = cv2.GaussianBlur(a, (0, 0), 1.5).max(-1) < 64
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)), (0, 0), 3)
    built = (exg < 12) & ~dark & (tex < 11)
    built &= ~cv2.dilate(osm_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    built &= ~cv2.dilate(road_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    m = cv2.morphologyEx(built.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 4)
    out = []
    for i in range(1, n):
        A = st[i, cv2.CC_STAT_AREA] * SRC * SRC
        if A < 300 or A > 8000: continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cs, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        sol = cv2.contourArea(cnt) / max(cv2.contourArea(hull), 1)
        rect = cv2.minAreaRect(cnt); rectf = cv2.contourArea(cnt) / max(rect[1][0] * rect[1][1], 1)
        if sol < 0.82 or rectf < 0.6: continue
        # тінь з північного боку: смуга над верхнім краєм
        ys, xs = np.nonzero(comp)
        top = {}
        for x, y in zip(xs, ys):
            if x not in top or y < top[x]: top[x] = y
        sh_ = [dark[max(0, y - 12):max(0, y - 2), x].mean() if y > 3 else 0 for x, y in top.items()]
        if np.mean(sh_) < 0.4: continue
        poly = cv2.approxPolyDP(cnt, 2.5, True)[:, 0, :].astype(float)
        c = rgb[ys, xs].mean(0)
        q_ = [(float(x + sx0), float(y + sy0)) for x, y in poly]; q_.append(q_[0])
        out.append(dict(q=q_, t={'building': 'yes', 'src': 'sat_big'}, area=round(A, 1),
                        roof=[int(c[0]), int(c[1]), int(c[2])], h=12.0, h_src='sat', detail=True))
    if debug is not None:
        v = a.copy()
        for o in out:
            P = np.array(o['q'], np.float32) - [sx0, sy0]
            cv2.polylines(v, [np.round(P).astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(debug, v)
    return out
