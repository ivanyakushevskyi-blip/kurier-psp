"""Сирі знімки Івана (ref/khn, ref/khn_south) у сітці гри: найдетальніший знімок, що покриває точку,
перетворений у пікселі карти (з надвибіркою). Для малювання пам'яток за зразком — не для вставки в гру."""
import os, json
import numpy as np, cv2

_IM = {}
META = None


def _load():
    global META
    if META is None:
        META = json.load(open('build/sat_map.json'))
        for fn, d in (('data/khn_init.json', 'ref/khn'), ('data/khn_south_init.json', 'ref/khn_south')):
            for k, (n, M) in json.load(open(fn)).items():
                p = os.path.join(d, k + '.jpg')
                if os.path.exists(p): _IM[k] = [p, np.array(M), None]


def crop(gx0, gy0, gx1, gy1, ss=1):
    """RGB float (H, W, 3) для прямокутника карти [gx0, gx1) x [gy0, gy1), ss пікс на піксель карти; None — нема знімка"""
    _load()
    W, H = int((gx1 - gx0) * ss), int((gy1 - gy0) * ss)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    gx = gx0 + (xx + 0.5) / ss; gy = gy0 + (yy + 0.5) / ss
    import citydraw as CD
    wx = META['x0'] + gx * CD.MPP; wy = META['y0'] + gy * CD.MPP
    best = None
    for k, (p, M, img) in _IM.items():
        Mi = np.linalg.inv(M)
        cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        q = Mi @ [META['x0'] + cx * CD.MPP, META['y0'] + cy * CD.MPP, 1]
        res = abs(M[0, 0])                                        # м на піксель знімка
        if img is None:
            import PIL.Image; PIL.Image.MAX_IMAGE_PIXELS = None
            sz = PIL.Image.open(p).size
        else:
            sz = (img.shape[1], img.shape[0])
        if 20 < q[0] < sz[0] - 20 and 20 < q[1] < sz[1] - 20 and (best is None or res < best[0]):
            best = (res, k)
    if best is None: return None
    k = best[1]
    if _IM[k][2] is None: _IM[k][2] = cv2.imread(_IM[k][0])[..., ::-1].copy()
    p, M, img = _IM[k]
    Mi = np.linalg.inv(M)
    u = Mi[0, 0] * wx + Mi[0, 1] * wy + Mi[0, 2]; v = Mi[1, 0] * wx + Mi[1, 1] * wy + Mi[1, 2]
    return cv2.remap(img.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
