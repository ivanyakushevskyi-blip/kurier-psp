"""Прив'язка перемальованої генератором частини знімка до сітки карти.
python3 -I tools/genreg.py <gen.png> <quarter.jpg>  -> друкує зсуви, пише build/gen/<name>.npz (A: gen px -> map px)"""
import sys, os, json
import numpy as np, cv2
from scipy.spatial import cKDTree

SAT = 'build/sat_map.npy'


def sift_pairs(a, b, ratio=0.75, n=8000):
    s = cv2.SIFT_create(n)
    ka, da = s.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = s.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    g = [x for x, y in m if x.distance < ratio * y.distance]
    return np.float32([ka[x.queryIdx].pt for x in g]), np.float32([kb[x.trainIdx].pt for x in g]), (ka, da, kb, db)


def quarter_to_map(q):
    """фото-частина -> карта (фото↔фото збігається надійно)."""
    sat = np.load(SAT, mmap_mode='r')
    H, W = sat.shape[:2]
    # грубо: зменшена карта, пошук по всій
    f = 0.25
    small = cv2.resize(np.ascontiguousarray(sat[::2, ::2]), None, fx=f * 2, fy=f * 2, interpolation=cv2.INTER_AREA)
    qs = cv2.resize(q, None, fx=0.32 / 0.32 * f * 1.0, fy=f, interpolation=cv2.INTER_AREA)
    P, Q, _ = sift_pairs(qs, small, 0.8, 20000)
    Hm, inl = cv2.findHomography(P, Q, cv2.RANSAC, 4)
    Hc = np.diag([1 / f, 1 / f, 1]) @ Hm @ np.diag([f, f, 1])          # q px -> map px (грубо)
    # уточнення на повній роздільності у вікні
    c = cv2.perspectiveTransform(np.float32([[[0, 0], [q.shape[1], 0], [q.shape[1], q.shape[0]], [0, q.shape[0]]]]), Hc)[0]
    x0, y0 = int(max(0, c[:, 0].min() - 100)), int(max(0, c[:, 1].min() - 100))
    x1, y1 = int(min(W, c[:, 0].max() + 100)), int(min(H, c[:, 1].max() + 100))
    crop = np.ascontiguousarray(sat[y0:y1, x0:x1])
    s2 = 0.5
    P, Q, _ = sift_pairs(cv2.resize(q, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA),
                         cv2.resize(crop, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA), 0.75, 20000)
    P /= s2; Q = Q / s2 + [x0, y0]
    pred = cv2.perspectiveTransform(P[None], Hc)[0]
    ok = np.linalg.norm(pred - Q, axis=1) < 60
    Hf, inl = cv2.findHomography(P[ok], Q[ok], cv2.RANSAC, 3)
    print('  photo->map inliers', int(inl.sum()))
    return Hf


def register(gen_path, q_path, out_dir='build/gen'):
    g = cv2.imread(gen_path); q = cv2.imread(q_path)
    Hq = quarter_to_map(q)
    S = np.diag([q.shape[1] / g.shape[1], q.shape[0] / g.shape[0], 1])
    H0 = Hq @ S                                                        # припущення: кадр збережено
    sat = np.load(SAT, mmap_mode='r')
    c = cv2.perspectiveTransform(np.float32([[[0, 0], [g.shape[1], 0], [g.shape[1], g.shape[0]], [0, g.shape[0]]]]), H0)[0]
    x0, y0 = int(c[:, 0].min() - 120), int(c[:, 1].min() - 120); x1, y1 = int(c[:, 0].max() + 120), int(c[:, 1].max() + 120)
    crop = np.ascontiguousarray(sat[y0:y1, x0:x1])
    sc = g.shape[1] / (c[:, 0].max() - c[:, 0].min())
    cs = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    s = cv2.SIFT_create(20000)
    ka, da = s.detectAndCompute(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), None)
    kb, db = s.detectAndCompute(cv2.cvtColor(cs, cv2.COLOR_BGR2GRAY), None)
    P = np.float32([k.pt for k in ka]); Q = np.float32([k.pt for k in kb]) / sc + [x0, y0]
    pred = cv2.perspectiveTransform(P[None], H0)[0]
    tr = cKDTree(Q); pairs = []
    for i, p in enumerate(pred):
        idx = tr.query_ball_point(p, 50)
        if not idx: continue
        d = np.linalg.norm(db[idx] - da[i], axis=1); o = np.argsort(d)
        if len(o) == 1 or d[o[0]] < 0.9 * d[o[1]]: pairs.append((i, idx[o[0]]))
    P2 = P[[a for a, b in pairs]]; Q2 = Q[[b for a, b in pairs]]
    Hg, inl = cv2.findHomography(P2, Q2, cv2.RANSAC, 10); inl = inl.ravel().astype(bool)
    r0 = np.linalg.norm(cv2.perspectiveTransform(P2[inl][None], H0)[0] - Q2[inl], axis=1)
    r1 = np.linalg.norm(cv2.perspectiveTransform(P2[inl][None], Hg)[0] - Q2[inl], axis=1)
    mpp = 0.32 * np.sqrt(abs(np.linalg.det(Hg[:2, :2])))
    name = os.path.splitext(os.path.basename(gen_path))[0]
    print('%s: inliers %d | frame kept: err med %.1f m | after fit: med %.1f m p90 %.1f m | %.2f m/px' % (
        name, inl.sum(), np.median(r0) * 0.32, np.median(r1) * 0.32, np.percentile(r1, 90) * 0.32, mpp))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, name + '.npz'), H=Hg, P=P2[inl], Q=Q2[inl])
    return Hg


if __name__ == '__main__':
    register(sys.argv[1], sys.argv[2])
