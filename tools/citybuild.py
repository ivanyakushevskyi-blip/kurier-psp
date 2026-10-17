"""Уся карта, намальована citydraw, у масштабі гри 0,5 м/пікс.
Вихід: build/map.npy (індекси палітри palgame), build/solid.npy, build/play.npy, build/mapdata.json,
       build/map_colors.json, build/preview_full.jpg
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import citydraw as CD

META = json.load(open('build/sat_map.json'))
K = CD.K
CS = 64
W = int(META['W'] * K) // CS * CS
H = int(META['H'] * K) // CS * CS
B, M = 1024, 64


def play_mask():
    b = np.load('build/bounds.npz')
    play = b['play']; G = float(b['G'])
    lon0, lat1, mlon, mlat = float(b['lon0']), float(b['lat1']), float(b['mlon']), float(b['mlat'])
    LAT0, LON0 = 49.422, 26.995
    MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
    xs = META['x0'] + (np.arange(W) + 0.5) * CD.MPP; ys = META['y0'] + (np.arange(H) + 0.5) * CD.MPP
    lon = LON0 + xs / MLON; lat = LAT0 - ys / MLAT
    jx = np.clip(((lon - lon0) * mlon / G).astype(int), 0, play.shape[1] - 1)
    iy = np.clip(((lat1 - lat) * mlat / G).astype(int), 0, play.shape[0] - 1)
    return play[iy[:, None], jx[None, :]]


def main():
    print('map', W, H)
    rgb = np.zeros((H, W, 3), np.uint8)
    cls = np.zeros((H, W), np.uint8)
    for by in range(0, H, B):
        for bx in range(0, W, B):
            w, h = min(B, W - bx), min(B, H - by)
            img, c = CD.render(bx - M, by - M, w + 2 * M, h + 2 * M)
            rgb[by:by + h, bx:bx + w] = img[M:M + h, M:M + w]
            cls[by:by + h, bx:bx + w] = c[M:M + h, M:M + w]
            print('  block', bx, by, flush=True)
    # ---- зіткнення: будинки OSM + знімкові, вода; поза ігровою зоною — стіна
    D = CD.data()
    solid = np.zeros((H, W), np.uint8)
    for b in D['B'] + D.get('SAT', []):
        P = np.array(b['q']) * K
        cv2.fillPoly(solid, [np.round(P).astype(np.int32)], 1)
    solid = solid.astype(bool) | (cls == 3)              # G_WATER
    play = play_mask()
    solid |= ~play
    # ---- палітра карти
    import palgame
    flat = rgb.reshape(-1, 3)
    cols, inv, cnt = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    print('unique colors', len(cols))
    if len(cols) > palgame.MAP_MAX:
        samp = flat[np.random.RandomState(1).choice(len(flat), 400000, replace=False)].astype(np.float32)
        _, _, cen = cv2.kmeans(samp, palgame.MAP_MAX, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
        # точні кольори, що займають багато пікселів, лишаємо як є
        top = cols[np.argsort(-cnt)[:palgame.MAP_MAX // 2]].astype(np.float32)
        cen = np.vstack([top, cen])[:palgame.MAP_MAX]
        cen = np.unique(np.round(cen).astype(np.int32), axis=0).astype(np.float32)
        lut = np.concatenate([((cols[i:i + 20000, None].astype(np.float32) - cen[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(cols), 20000)])
        cols = np.round(cen).astype(np.uint8)
        idx = lut[inv.ravel()]
    else:
        idx = inv.ravel()
    json.dump([list(map(int, c)) for c in cols], open('build/map_colors.json', 'w'))
    mp = (idx.reshape(H, W) + palgame.MAP_START).astype(np.uint8)
    np.save('build/map.npy', mp); np.save('build/solid.npy', solid); np.save('build/play.npy', play)
    # ---- дані гри: зі старого mapdata (сітка 0,32) у сітку 0,5
    md = json.load(open('build/mapdata_032.json'))
    def sp(x, y, r=40):
        x, y = int(x * K), int(y * K)
        if not (0 <= x < W and 0 <= y < H): return None
        if not solid[y, x]: return (x, y)
        ys_, xs_ = np.nonzero(~solid[max(0, y - r):y + r, max(0, x - r):x + r])
        if len(xs_) == 0: return None
        k = ((xs_ - r) ** 2 + (ys_ - r) ** 2).argmin(); return (int(xs_[k] + max(0, x - r)), int(ys_[k] + max(0, y - r)))
    shops = [dict(s, x=p[0], y=p[1]) for s in md['shops'] for p in [sp(s['x'], s['y'])] if p]
    addrs = [dict(a, x=p[0], y=p[1]) for a in md['addrs'] for p in [sp(a['x'], a['y'])] if p]
    segs = [[int(a * K), int(b * K), int(c * K), int(d * K), n] for a, b, c, d, n in md['segs']]
    spawn = sp(*md['spawn'], r=80) or (W // 2, H // 2)
    json.dump(dict(shops=shops, addrs=addrs, names=md['names'], segs=segs, spawn=list(spawn), W=W, H=H, M=CD.MPP),
              open('build/mapdata.json', 'w'), ensure_ascii=False)
    pv = rgb[::3, ::3].copy(); pv[~play[::3, ::3]] = (pv[~play[::3, ::3]] * 0.5).astype(np.uint8)
    cv2.imwrite('build/preview_full.jpg', pv[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('shops', len(shops), 'addrs', len(addrs), 'colors', len(cols), 'spawn', spawn)


if __name__ == '__main__':
    main()
