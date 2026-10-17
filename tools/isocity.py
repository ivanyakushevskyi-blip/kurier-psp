"""Варіант 2 карти: основа — стилізований супутниковий знімок (піксель-арт),
будинки з OSM-контурів «піднімаються» з домальованими фасадами (вид згори під кутом).
Зразок: python3 -I tools/isocity.py sample <cx> <cy> <w> <h> <out.png>
"""
import json, math, random, sys
import numpy as np, cv2
from PIL import Image, ImageDraw

META = json.load(open('build/sat_map.json'))
MPP = META['mpp']
WALL_K = 1.9            # пікселів стіни на метр висоти (більше — «ізометричніше»)
WALL_MAX = 80


# ---------------------------------------------------------------- 1. стилізація знімка
def lift_shadows(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    Lb = cv2.GaussianBlur(L, (0, 0), 6)
    m = np.clip((105 - Lb) / 45, 0, 1) ** 0.8
    lab[..., 0] = np.clip(L * (1 + 1.5 * m) + 16 * m, 0, 235)
    lab[..., 1] += 3 * m
    lab[..., 2] += 8 * m
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # глибока тінь без інформації — домалювати з околиць
    core = (bgr.max(-1) < 26).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if core.any():
        out = cv2.inpaint(out, core, 6, cv2.INPAINT_TELEA)
    return out


def stylize(bgr):
    a = lift_shadows(bgr)
    a = cv2.pyrMeanShiftFiltering(a, 3, 20)
    a = cv2.convertScaleAbs(a, alpha=1.1, beta=-6)
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.2
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- 2. будинки
def load_buildings():
    return json.load(open('build/buildings.json'))['buildings']


def poly_mask(pts, w, h, dx=0.0, dy=0.0):
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).polygon([(x + dx, y + dy) for x, y in pts], fill=1)
    return np.array(im, bool)


def roof_shift(gray_grad, pts, w, h, R):
    """Зсув, на якому контур будинку найкраще лягає на дах зі знімка (знімок має нахил висоток)."""
    base = poly_mask(pts, w, h)
    edge = base & ~cv2.erode(base.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(edge)
    if len(xs) < 8: return 0, 0
    best = (-1, 0, 0)
    step = 2 if R > 10 else 1
    for dy in range(-R, R // 3 + 1, step):
        for dx in range(-R, R + 1, step):
            X = xs + dx; Y = ys + dy
            ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
            if ok.mean() < 0.7: continue
            sc = gray_grad[Y[ok], X[ok]].mean() - 0.02 * (dx * dx + dy * dy) ** 0.5
            if sc > best[0]: best = (sc, dx, dy)
    return best[1], best[2]


FACADES = {   # (стіна світла, стіна тінь, низ, вікно, рамка)
    'plaster_w': ((214, 208, 196), (168, 162, 152), (120, 112, 104), (52, 70, 96), (240, 236, 228)),
    'plaster_y': ((222, 196, 140), (176, 152, 106), (122, 100, 70), (52, 70, 96), (244, 228, 190)),
    'brick':     ((168, 88, 64), (128, 64, 48), (86, 46, 36), (44, 58, 80), (214, 190, 170)),
    'panel':     ((196, 200, 204), (150, 154, 160), (104, 108, 114), (48, 64, 90), (226, 228, 232)),
    'glass':     ((96, 132, 170), (66, 96, 130), (52, 64, 84), (150, 190, 220), (190, 210, 228)),
    'shed':      ((176, 178, 180), (136, 138, 140), (96, 98, 100), (96, 98, 100), (196, 198, 200)),
}


def facade_kind(t, hm, area, rs):
    b = t.get('building', 'yes')
    if b in ('garage', 'garages', 'shed', 'roof', 'kiosk', 'service', 'industrial', 'warehouse'): return 'shed'
    if t.get('building:material') == 'brick' or b in ('church', 'cathedral'): return 'brick'
    if b in ('apartments', 'residential', 'dormitory') and hm > 13: return rs.choice(['panel', 'panel', 'plaster_w', 'brick'])
    if b in ('retail', 'commercial', 'office') and hm > 10: return rs.choice(['glass', 'plaster_w', 'panel'])
    return rs.choice(['plaster_w', 'plaster_y', 'brick', 'plaster_w'])


def draw_building(canvas, ghost, sty, grad, b, ox, oy, seed):
    """canvas: RGB (H,W,3) uint8 у координатах вікна; pts — у координатах карти."""
    H, W = canvas.shape[:2]
    pts = [(x - ox, y - oy) for x, y in b['q']]
    hm = b['h']; t = b['t']
    if b['h_src'] == 'shadow' and b['area'] < 150: hm = min(hm, 9)
    if t.get('building') in ('garage', 'garages', 'shed', 'kiosk', 'roof'): hm = min(hm, 3.5)
    wp = int(round(min(WALL_MAX, max(4, WALL_K * hm))))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx0 = int(min(xs)) - 50; by0 = int(min(ys)) - wp - 50
    bx1 = int(max(xs)) + 50; by1 = int(max(ys)) + 50
    if bx1 < 0 or by1 < 0 or bx0 >= W or by0 >= H: return
    cx0, cy0 = max(bx0, 0), max(by0, 0); cx1, cy1 = min(bx1, W), min(by1, H)
    w, h = cx1 - cx0, cy1 - cy0
    lp = [(x - cx0, y - cy0) for x, y in pts]
    F = poly_mask(lp, w, h)
    if F.sum() < 6: return
    rs = random.Random(seed)
    # зсув даху на знімку (нахил висотки)
    R = int(min(40, 4 + hm * 0.9))
    dx, dy = roof_shift(grad[cy0:cy1, cx0:cx1], lp, w, h, R) if hm > 5 else (0, 0)
    # «привид» будинку на знімку: від контуру до даху
    G = np.zeros_like(F)
    for k in np.linspace(0, 1, 6):
        G |= poly_mask(lp, w, h, dx * k, dy * k)
    ghost[cy0:cy1, cx0:cx1] |= cv2.dilate(G.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # дах: текстура зі знімка в зсунутому контурі, ставимо на висоту стіни
    roof = np.zeros_like(F); roof[:-wp] = F[wp:] if wp < h else roof[:-wp]
    sub = sty[cy0:cy1, cx0:cx1]
    Ys, Xs = np.nonzero(roof)
    sy = np.clip(Ys + wp + dy, 0, h - 1); sx = np.clip(Xs + dx, 0, w - 1)
    rtex = sub[sy, sx].astype(np.int16)
    # стіни: витяг контуру вгору
    S = F.copy()
    for d in range(1, wp + 1): S[:-d] |= F[d:]
    wall = S & ~roof
    fk = FACADES[facade_kind(t, hm, b['area'], rs)]
    out = canvas[cy0:cy1, cx0:cx1]
    base = np.full(w, -1)
    anyc = F.any(0)
    base[anyc] = (h - 1 - np.argmax(F[::-1], 0))[anyc]
    # освітлення стіни за нахилом нижнього краю (південь — світло, захід/схід — тінь)
    slope = np.zeros(w); bb = base.astype(float)
    for x in range(1, w - 1):
        if base[x - 1] >= 0 and base[x + 1] >= 0: slope[x] = (bb[x + 1] - bb[x - 1]) / 2
    yy, xx = np.nonzero(wall)
    hy = base[xx] - yy                                  # висота над землею в пікселях
    lit = np.clip(0.75 - 0.35 * np.clip(slope[xx], -2, 2), 0, 1)
    c_l, c_d, c_low, c_win, c_fr = [np.array(c[::-1], np.float32) for c in fk]   # RGB -> BGR
    col = c_d + (c_l - c_d) * lit[:, None]
    col = np.where((hy < 3)[:, None], c_low, col)
    floors = max(1, int(round(hm / 3.0)))
    if fk is not FACADES['shed'] and wp >= 7:
        fl = max(3.0, (wp - 3) / floors)
        fpos = (hy - 2) % fl
        win = (hy >= 4) & (hy <= wp - 2) & (fpos >= fl * 0.3) & (fpos < fl * 0.3 + max(2.0, fl * 0.5)) & ((xx % 6) < 3)
        col[win] = c_win * (0.8 + 0.4 * lit[win, None])
        cor = (hy >= wp - 1)
        col[cor] = c_fr
    out[yy, xx] = np.clip(col, 0, 255).astype(np.uint8)
    # дах
    rt = rtex.astype(np.float32)
    out[Ys, Xs] = np.clip(rt * 1.05 + 4, 0, 255).astype(np.uint8)
    # край даху світлий, лінія між дахом і стіною темна, контур
    rU = roof & ~np.vstack([np.zeros((1, w), bool), roof[:-1]])
    rL = roof & ~np.hstack([np.zeros((h, 1), bool), roof[:, :-1]])
    hi = rU | rL
    out[hi] = np.clip(out[hi].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    rb = wall & np.vstack([np.zeros((1, w), bool), roof[:-1]])
    out[rb] = (out[rb] * 0.55).astype(np.uint8)
    ol = cv2.dilate(S.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~S
    out[ol] = (out[ol] * 0.35).astype(np.uint8)
    # тінь на землю (на схід)
    return S, (cx0, cy0)


def render_window(cx, cy, w, h):
    x0, y0 = cx - w // 2, cy - h // 2
    M = 80                                                  # запас для стилізації
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    raw = np.ascontiguousarray(sat[y0 - M:y0 + h + M, x0 - M:x0 + w + M])
    sty = stylize(raw)
    gray = cv2.cvtColor(cv2.GaussianBlur(raw, (3, 3), 0), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    canvas = sty.copy()
    ghost = np.zeros(canvas.shape[:2], bool)
    blds = load_buildings()
    sel = []
    for i, b in enumerate(blds):
        q = np.array(b['q'])
        if q[:, 0].max() < x0 - M - 60 or q[:, 0].min() > x0 + w + M + 60 or q[:, 1].max() < y0 - M - 60 or q[:, 1].min() > y0 + h + M + 200: continue
        sel.append((q[:, 1].max(), i, b))
    sel.sort(key=lambda s: s[0])
    # 1) прибрати «привидів» висоток зі знімка
    canvas_tmp = canvas.copy(); masks = []
    for _, i, b in sel:
        r = draw_building(canvas_tmp, ghost, sty, grad, b, x0 - M, y0 - M, i)
        if r: masks.append(r)
    covered = np.zeros_like(ghost)
    for S, (ax, ay) in masks:
        covered[ay:ay + S.shape[0], ax:ax + S.shape[1]] |= S
    hole = (ghost & ~covered).astype(np.uint8)
    ground = cv2.inpaint(sty, hole, 5, cv2.INPAINT_TELEA) if hole.any() else sty
    # 2) тіні будинків на землю (на схід-північний схід) і самі будинки
    canvas = ground.copy()
    shadow = np.zeros_like(ghost)
    for S, (ax, ay) in masks:
        k = 10
        sh = np.zeros((S.shape[0] + 4, S.shape[1] + k), bool)
        for d in range(1, k): sh[:S.shape[0], d:d + S.shape[1]] |= S
        hh, ww = min(sh.shape[0], shadow.shape[0] - ay), min(sh.shape[1], shadow.shape[1] - ax)
        if hh > 0 and ww > 0: shadow[ay:ay + hh, ax:ax + ww] |= sh[:hh, :ww]
    shadow &= ~covered
    canvas[shadow] = (canvas[shadow] * 0.72).astype(np.uint8)
    ghost[:] = False
    for _, i, b in sel:
        draw_building(canvas, ghost, sty, grad, b, x0 - M, y0 - M, i)
    return canvas[M:M + h, M:M + w], raw[M:M + h, M:M + w]


if __name__ == '__main__':
    if sys.argv[1] == 'sample':
        cx, cy, w, h = map(int, sys.argv[2:6])
        img, raw = render_window(cx, cy, w, h)
        # як буде на PSP: 8-бітна палітра
        Z = img.reshape(-1, 3).astype(np.float32)
        _, _, cen = cv2.kmeans(Z[::5], 160, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        q = np.concatenate([((Z[i:i + 200000, None, :] - cen[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(Z), 200000)])
        img = cen[q].reshape(img.shape).astype(np.uint8)
        cv2.imwrite(sys.argv[6], img)
        cv2.imwrite(sys.argv[6].replace('.png', '_raw.png'), raw)
