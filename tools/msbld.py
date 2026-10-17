"""Будівлі з Microsoft Global ML Building Footprints (ODbL), яких немає в OSM.
data/ms/*.csv.gz -> build/ms_buildings.json (формат buildings.json, сітка 0,32 м)."""
import sys, os, json, gzip, glob, math
import numpy as np, cv2

SRC = 0.32
LAT0, LON0 = 49.422, 26.995
MLON = 111320.0 * math.cos(math.radians(LAT0))


def main():
    meta = json.load(open('build/sat_map.json')); W, H = meta['W'], meta['H']
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    osm = json.load(open('build/buildings.json'))['buildings']
    extra = json.load(open('build/manual_buildings.json')) if os.path.exists('build/manual_buildings.json') else []
    om = np.zeros((H, W), np.uint8)
    for b in osm + extra:
        cv2.fillPoly(om, [np.round(np.array(b['q'])).astype(np.int32)], 1)
    res = []
    for f in sorted(glob.glob('data/ms/*.csv.gz')):
        for line in gzip.open(f, 'rt'):
            c = json.loads(line)['geometry']['coordinates'][0]
            q = np.array([(((lo - LON0) * MLON - meta['x0']) / SRC, (-(la - LAT0) * 111320 - meta['y0']) / SRC) for lo, la in c])
            if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() >= W or q[:, 1].min() >= H: continue
            x0, y0 = np.floor(q.min(0)).astype(int); x1, y1 = np.ceil(q.max(0)).astype(int) + 1
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
            if x1 - x0 < 3 or y1 - y0 < 3: continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.round(q - [x0, y0]).astype(np.int32)], 1)
            m = m.astype(bool)
            if m.sum() < 30: continue
            if om[y0:y1, x0:x1][m].mean() > 0.25: continue          # уже є в OSM
            A = m.sum() * SRC * SRC
            px = np.asarray(sat[y0:y1, x0:x1])[cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) if m.sum() > 80 else m][:, ::-1].astype(float)
            lit = px[px.max(1) > 70]
            use = lit if len(lit) >= max(6, 0.15 * len(px)) else px
            ch = use.max(1) - use.min(1)
            top = use[ch >= np.percentile(ch, 60)] if len(use) else use
            roof = np.median(top, 0) if len(top) >= 6 and np.median(top.max(1) - top.min(1)) > 28 else np.median(use, 0)
            h = 4.0 if A < 60 else (6.0 if A < 250 else 12.0)
            res.append(dict(q=[tuple(p) for p in q.tolist()], t={'building': 'house' if A < 250 else 'yes', 'src': 'ms'},
                            area=round(A, 1), roof=[int(v) for v in roof], h=h, h_src='ms', detail=True))
    json.dump(res, open('build/ms_buildings.json', 'w'))
    print('MS buildings added:', len(res))


if __name__ == '__main__':
    main()
