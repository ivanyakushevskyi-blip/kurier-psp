"""Межі v4 (31.10.2026, Іван): Озерна розширюється до лісу — по Лісогринівецькій до Старокостянтинівського шосе;
на схід від шосе над Автовокзалом №1 — лабіринти гаражів/дач прибрано (межа — шосе, сам автовокзал лишається).
Решта — як v3. Пише data/bounds_v4.json ([lon, lat]); дороги поблизу межі до зони додає bounds2 (snap_roads).
python3 -I tools/bounds4.py"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import satmosaic as S

LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
G = 2.0
X0, Y0, X1, Y1 = -6000.0, -6000.0, 6000.0, 6000.0


def m2px(p): return ((p[0] - X0) / G, (p[1] - Y0) / G)
def ll2m(lo, la): return ((lo - LON0) * MLON, -(la - LAT0) * MLAT)
def m2ll(x, y): return (LON0 + x / MLON, LAT0 - y / MLAT)


def way_pts(ways, name, hw=None):
    out = []
    for p, t in ways:
        if t.get('name') == name and (hw is None or t.get('highway') in hw) and p: out.append(np.array(p))
    return out


def chain(segs, start):
    """з'єднати відрізки в одну ламану, починаючи з найближчого до start"""
    segs = [s for s in segs]
    cur = min(segs, key=lambda s: min(np.linalg.norm(s[0] - start), np.linalg.norm(s[-1] - start)))
    segs = [x for x in segs if x is not cur]
    if np.linalg.norm(cur[-1] - start) < np.linalg.norm(cur[0] - start): cur = cur[::-1]
    line = list(cur)
    while segs:
        end = line[-1]
        nx = min(segs, key=lambda s: min(np.linalg.norm(s[0] - end), np.linalg.norm(s[-1] - end)))
        if min(np.linalg.norm(nx[0] - end), np.linalg.norm(nx[-1] - end)) > 30: break
        segs = [x for x in segs if x is not nx]
        if np.linalg.norm(nx[-1] - end) < np.linalg.norm(nx[0] - end): nx = nx[::-1]
        line.extend(nx[1:])
    return np.array(line)


def main():
    ways = S.load_osm('data/osm_v3')
    W, H = int((X1 - X0) / G), int((Y1 - Y0) / G)
    old = [ll2m(lo, la) for lo, la in json.load(open('data/bounds_v3.json'))]
    play = np.zeros((H, W), np.uint8)
    cv2.fillPoly(play, [np.round(np.array([m2px(p) for p in old]) * 8).astype(np.int32)], 1, shift=3)
    # ---- північ: Бандери -> Лісогринівецька -> Старокостянтинівське шосе -> Січових стрільців -> стара межа
    lis = chain(way_pts(ways, 'Лісогринівецька вулиця'), np.array([1101.0, -4297.0]))
    sk = [s for s in way_pts(ways, 'Старокостянтинівське шосе') if s[:, 1].min() > -3420 and s[:, 1].max() < -1400]
    skl = chain(sk, np.array([1830.0, -3385.0]))                       # з півночі на південь
    north = [(640, -3982), (760, -4140), (860, -4297)] + [tuple(p) for p in lis] + [(1830, -3385)] + \
            [tuple(p) for p in skl if p[1] < -3180] + [(1166, -3193), (970, -3597)]
    cv2.fillPoly(play, [np.round(np.array([m2px(p) for p in north]) * 8).astype(np.int32)], 1, shift=3)
    # ---- схід: усе на схід від шосе між Січових стрільців і Автовокзалом №1 — геть (сам автовокзал і пр. Миру лишаються)
    east = [tuple(np.array(p) + [35, 0]) for p in skl if p[1] < -1450] + [(1700, -1300), (2120, -1300), (2120, -1250),
            (3200, -1250), (3200, -3700), (1860, -3700)]
    cv2.fillPoly(play, [np.round(np.array([m2px(p) for p in east]) * 8).astype(np.int32)], 0, shift=3)
    north_strip = [(1830, -3385)] + [tuple(np.array(p) + [35, 0]) for p in skl if p[1] < -3180] + [(1866, -3180), (2000, -3700), (1860, -3700)]
    cv2.fillPoly(play, [np.round(np.array([m2px(p) for p in north_strip]) * 8).astype(np.int32)], 0, shift=3)
    play = cv2.morphologyEx(play, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(play)
    play = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cs, _ = cv2.findContours(play, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 2.0, True)[:, 0, :].astype(np.float64)
    pts = [m2ll(x * G + X0, y * G + Y0) for x, y in c]
    json.dump([[round(lo, 7), round(la, 7)] for lo, la in pts], open('data/bounds_v4.json', 'w'))
    print('межа v4: %d вершин, %.2f км² (v3: %.2f км²)' % (len(pts), cv2.contourArea(c.astype(np.float32)) * G * G / 1e6,
          cv2.contourArea(np.array([m2px(p) for p in old], np.float32)) * G * G / 1e6))


if __name__ == '__main__':
    main()
