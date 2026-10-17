"""Аналіз будинків: колір даху зі знімка, висота (OSM або за довжиною тіні на знімку).
Пише build/buildings.json."""
import glob, json, math, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
sys.path.insert(0, __import__('os').path.dirname(__file__))
import satmosaic as S


def levels_of(t):
    for k in ('building:levels', 'levels'):
        try:
            v = float(t.get(k, '').replace(',', '.'))
            if v > 0: return v
        except ValueError: pass
    try:
        h = float(t.get('height', '').replace('m', '').strip())
        if h > 0: return h / 3.0
    except ValueError: pass
    return None


def main():
    meta = json.load(open('build/sat_map.json')); sat = np.load('build/sat_map.npy', mmap_mode='r')
    x0, y0, mpp = meta['x0'], meta['y0'], meta['mpp']; H, W = sat.shape[:2]
    det = np.load('build/sat_detail_mask.npy', mmap_mode='r')
    ways = S.load_osm('data/osm_v2')
    blds = []
    for p, t in ways:
        if 'building' not in t or len(p) < 4: continue
        q = np.array([((a - x0) / mpp, (b - y0) / mpp) for a, b in p])
        if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() >= W or q[:, 1].min() >= H: continue
        area = 0.5 * abs(np.dot(q[:-1, 0], q[1:, 1]) - np.dot(q[1:, 0], q[:-1, 1])) * mpp * mpp
        if area < 8: continue
        blds.append(dict(q=q, t=t, area=area))
    print('buildings', len(blds))
    # 1) кольори даху та маски
    for b in blds:
        q = b['q']; bx0, by0 = np.floor(q.min(0)).astype(int) - 2; bx1, by1 = np.ceil(q.max(0)).astype(int) + 3
        bx0, by0 = max(bx0, 0), max(by0, 0); bx1, by1 = min(bx1, W), min(by1, H)
        b['box'] = [int(bx0), int(by0), int(bx1), int(by1)]
        if bx1 - bx0 < 3 or by1 - by0 < 3: b['roof'] = None; continue
        im = Image.new('1', (bx1 - bx0, by1 - by0), 0)
        ImageDraw.Draw(im).polygon([(x - bx0, y - by0) for x, y in q], fill=1)
        m = np.array(im)
        er = ndi.binary_erosion(m, iterations=3)
        if er.sum() < 12: er = ndi.binary_erosion(m, iterations=1)
        if er.sum() < 6: er = m
        px = np.asarray(sat[by0:by1, bx0:bx1])[er].astype(np.float32)          # BGR
        b['detail'] = bool(np.asarray(det[by0:by1, bx0:bx1])[er].mean() > 0.5) if er.any() else False
        if len(px) == 0: b['roof'] = None; continue
        v = px.max(1)
        lit = px[v > 70]                                                       # без тіней
        use = lit if len(lit) >= max(6, 0.15 * len(px)) else px
        med = np.median(use, 0)
        b['roof'] = med[::-1].round().astype(int).tolist()                     # RGB
        b['roof_lit_frac'] = float(len(lit) / len(px))
    # 2) напрям тіні: калібрування на будинках з поверховістю
    gray = None
    def dark_at(xs, ys):
        xs = np.clip(xs.astype(int), 0, W - 1); ys = np.clip(ys.astype(int), 0, H - 1)
        return np.asarray(sat[ys, xs]).max(-1) < 55
    calib = [b for b in blds if levels_of(b['t']) and b['area'] > 150]
    print('calibration buildings with levels:', len(calib))
    def shadow_len(b, ang, maxm=120):
        q = b['q']; dx, dy = math.cos(ang), math.sin(ang)
        # точки вздовж контуру
        pts = []
        for i in range(len(q) - 1):
            L = np.hypot(*(q[i + 1] - q[i])); n = max(1, int(L / 6))
            for k in range(n): pts.append(q[i] + (q[i + 1] - q[i]) * (k + 0.5) / n)
        pts = np.array(pts)
        if len(pts) == 0: return 0.0
        im_poly = q
        # точки, що лежать із «тіньового» боку (нормаль зовні дивиться в напрямку тіні)
        c = q.mean(0); side = ((pts - c) @ [dx, dy]) > 0
        pts = pts[side]
        if len(pts) == 0: return 0.0
        steps = np.arange(2, int(maxm / mpp), 2)
        X = pts[:, 0:1] + dx * steps[None]; Y = pts[:, 1:2] + dy * steps[None]
        D = dark_at(X, Y)
        # довжина безперервної темряви від контуру
        run = np.argmin(np.c_[D, np.zeros(len(D), bool)], 1)
        lens = steps[np.clip(run - 1, 0, len(steps) - 1)] * (run > 0)
        return float(np.percentile(lens, 70) * mpp)
    best = None
    for deg in range(0, 360, 10):
        ang = math.radians(deg)
        sc = np.mean([shadow_len(b, ang, 60) for b in calib[:150]])
        if best is None or sc > best[1]: best = (deg, sc)
    for deg in np.arange(best[0] - 9, best[0] + 9.1, 3):
        ang = math.radians(deg); sc = np.mean([shadow_len(b, ang, 60) for b in calib[:150]])
        if sc > best[1]: best = (deg, sc)
    sang = math.radians(best[0])
    print('shadow direction %.0f deg (image coords, 0=east, 90=south)' % best[0])
    # 3) коефіцієнт: висота = k * довжина тіні
    hs, ls = [], []
    for b in calib:
        L = shadow_len(b, sang)
        if L > 3: hs.append(levels_of(b['t']) * 3.0 + 1.5); ls.append(L)
    hs, ls = np.array(hs), np.array(ls)
    k = float(np.median(hs / ls))
    print('k = %.3f (height/shadow), n=%d, spread %.2f' % (k, len(hs), np.std(hs / ls)))
    for b in blds:
        lv = levels_of(b['t'])
        L = shadow_len(b, sang) if b.get('detail') else 0.0
        hsh = k * L
        b['h_shadow'] = hsh
        if lv: b['h'] = lv * 3.0 + 1.5; b['h_src'] = 'osm'
        elif hsh > 4: b['h'] = float(min(hsh, 90)); b['h_src'] = 'shadow'
        else:
            bt = b['t'].get('building')
            b['h'] = 3.5 if bt in ('garage', 'garages', 'shed', 'kiosk', 'roof') else (6.0 if b['area'] < 200 else 12.0); b['h_src'] = 'guess'
    out = [dict(q=b['q'].round(2).tolist(), id=None, t=b['t'], area=round(b['area'], 1), roof=b.get('roof'), h=round(b['h'], 1),
                h_src=b['h_src'], detail=b.get('detail', False)) for b in blds]
    json.dump(dict(buildings=out, shadow_deg=best[0], k=k), open('build/buildings.json', 'w'), ensure_ascii=False)
    from collections import Counter
    print(Counter(b['h_src'] for b in blds))


if __name__ == '__main__':
    main()
