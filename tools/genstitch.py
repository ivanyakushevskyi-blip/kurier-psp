"""Зшивання генерованих частин у сітці карти без «подвійних» країв.
Кроки для кожної нової частини відносно вже зшитого:
 1) геометрія: точки SIFT між частинами (той самий стиль — збігів багато) -> плавне поле зсувів (RBF), щоб частина лягла піксель у піксель;
 2) колір: лінійна корекція кольорів за спільною зоною;
 3) шов: розріз по шляху найменшої різниці (по дорогах/дахах, де картинки однакові), розмиття лише 2–3 пікселі.
python3 -I tools/genstitch.py <out.png> name1 name2 ...   (ref/gen/<name>.jpg|png + build/gen/<name>.npz)
"""
import sys, os, glob
import numpy as np, cv2
from scipy.interpolate import RBFInterpolator


def load(name):
    p = [f for f in glob.glob('ref/gen/%s.*' % name) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))][0]
    return cv2.imread(p), np.load('build/gen/%s.npz' % name)['H']


def warp(img, H, box):
    x0, y0, W, Hh = box
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    w = cv2.warpPerspective(img, T @ H, (W, Hh), flags=cv2.INTER_NEAREST)
    m = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T @ H, (W, Hh), flags=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0                    # без розмитих країв картинки
    return w, m


def refine_geometry(base, bm, img, im):
    """Підтягнути img до base у спільній зоні плавним полем зсувів."""
    ov = bm & im
    if ov.sum() < 2000: return img, im
    s = cv2.SIFT_create(20000)
    ka, da = s.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.dilate(ov.astype(np.uint8), np.ones((41, 41))) * 255)
    kb, db = s.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.dilate(ov.astype(np.uint8), np.ones((41, 41))) * 255)
    if da is None or db is None: return img, im
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    g = [x for x, y in m if x.distance < 0.75 * y.distance]
    P = np.float32([ka[x.queryIdx].pt for x in g]); Q = np.float32([kb[x.trainIdx].pt for x in g])
    d = Q - P; ok = np.linalg.norm(d, axis=1) < 25
    P, d = P[ok], d[ok]
    if len(P) < 12: return img, im
    # відкинути викиди відносно сусідів
    med = np.median(d, 0); ok = np.linalg.norm(d - med, axis=1) < 10
    P, d = P[ok], d[ok]
    print('    шов: точок %d, зсув медіана %.1f px, макс %.1f px' % (len(P), np.median(np.linalg.norm(d, axis=1)), np.linalg.norm(d, axis=1).max()))
    # поле зсувів на грубій сітці, далі збільшити; за межами спільної зони — плавно до медіанного зсуву
    H, W = im.shape; step = 16
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    rbf = RBFInterpolator(P, d, kernel='thin_plate_spline', smoothing=200.0, degree=1)
    F = rbf(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape + (2,)).astype(np.float32)
    F = np.clip(F, -25, 25)
    Fx = cv2.resize(F[..., 0], (W, H), interpolation=cv2.INTER_LINEAR)
    Fy = cv2.resize(F[..., 1], (W, H), interpolation=cv2.INTER_LINEAR)
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # img(p) має опинитися в p+d  =>  нове(q) = img(q - d)
    img2 = cv2.remap(img, xx - Fx, yy - Fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    im2 = cv2.remap(im.astype(np.uint8), xx - Fx, yy - Fy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return img2, im2


def lab_stats(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return L.mean(0), L.std(0)


def normalize_colors(img, target):
    """Вся частина цілком під спільну «температуру» міста (середнє і розкид по LAB).
    Порівнювати лише спільну смугу не можна: там генератор малює ті самі об'єкти по-різному,
    і корекція то «сіріє», то перенасичує всю частину."""
    m, s = lab_stats(img); tm, ts = target
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = (L - m) * np.clip(ts / (s + 1e-3), 0.85, 1.2) + tm
    return cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def seam_merge(base, bm, img, im):
    """Розріз по шляху найменшої різниці в спільній зоні."""
    ov = bm & im
    out = base.copy(); out[im & ~bm] = img[im & ~bm]
    if not ov.any(): return out, bm | im
    diff = np.abs(base.astype(np.int16) - img.astype(np.int16)).sum(-1).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.5)
    # хто «з якого боку»: відстань до власного краю
    db = cv2.distanceTransform(bm.astype(np.uint8), cv2.DIST_L2, 5)
    di = cv2.distanceTransform(im.astype(np.uint8), cv2.DIST_L2, 5)
    # мінімальний розріз: заливка з двох боків по зростанню різниці (watershed на карті різниці)
    markers = np.zeros(bm.shape, np.int32)
    markers[bm & ~im] = 1; markers[im & ~bm] = 2
    markers[ov & (db > di * 3 + 20)] = 1; markers[ov & (di > db * 3 + 20)] = 2
    markers[~(bm | im)] = 0
    grad = np.clip(diff * 2, 0, 255).astype(np.uint8)
    lab = cv2.watershed(cv2.merge([grad, grad, grad]), markers.copy())
    take_img = (lab == 2) & im
    take_img = cv2.dilate(take_img.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    take_img &= im
    wgt = cv2.GaussianBlur(take_img.astype(np.float32), (0, 0), 1.2)
    wgt[~bm] = 1.0; wgt[~im] = 0.0
    out = (base * (1 - wgt[..., None]) + img * wgt[..., None]).astype(np.uint8)
    return out, bm | im


def stitch(names):
    items = [load(n) for n in names]
    st = [lab_stats(i) for i, _ in items]
    target = (np.mean([m for m, _ in st], 0), np.mean([v for _, v in st], 0))
    items = [(normalize_colors(i, target), H) for i, H in items]
    cs = []
    for img, H in items:
        h, w = img.shape[:2]
        cs.append(cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0])
    allc = np.vstack(cs)
    x0, y0 = np.floor(allc.min(0)).astype(int); x1, y1 = np.ceil(allc.max(0)).astype(int)
    box = (x0, y0, x1 - x0, y1 - y0)
    base, bm = warp(*items[0], box)
    for n, (img, H) in zip(names[1:], items[1:]):
        print('  +', n)
        w, m = warp(img, H, box)
        base, bm = seam_merge(base, bm, w, m)
    return base, bm, (x0, y0)


if __name__ == '__main__':
    out = sys.argv[1]
    img, m, (x0, y0) = stitch(sys.argv[2:])
    img[~m] = 0
    cv2.imwrite(out, img)
    np.save(os.path.splitext(out)[0] + '_xy.npy', np.array([x0, y0]))
    print('saved', out, img.shape, 'at map', x0, y0)
