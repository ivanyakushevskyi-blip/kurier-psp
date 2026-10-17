"""Класифікація супутникової підкладки на сітці ~1 м (крок 3 пікселі карти).
Класи: 0 тінь, 1 трава, 2 дерево, 3 ґрунт/суха трава, 4 світле покриття (бетон/плитка), 5 асфальт, 6 інше.
Пише build/satcls.npz: cls (h,w uint8), rgb (h,w,3 uint8 — середній колір без тіні), tex."""
import json, numpy as np
from scipy import ndimage as ndi

F = 3


def features(sat):
    H, W = sat.shape[:2]; h, w = H // F, W // F
    out_rgb = np.zeros((h, w, 3), np.float32); tex = np.zeros((h, w), np.float32)
    B = 330  # рядків сітки за раз
    for r0 in range(0, h, B):
        r1 = min(h, r0 + B)
        a = np.asarray(sat[r0 * F:r1 * F, :w * F]).astype(np.float32)[..., ::-1]   # RGB
        blk = a.reshape(r1 - r0, F, w, F, 3)
        out_rgb[r0:r1] = blk.mean((1, 3))
        L = a.mean(-1)
        gx = np.abs(np.diff(L, axis=1, append=L[:, -1:])); gy = np.abs(np.diff(L, axis=0, append=L[-1:]))
        tex[r0:r1] = (gx + gy).reshape(r1 - r0, F, w, F).mean((1, 3))
    return out_rgb, tex


def classify(rgb, tex):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); ch = mx - mn; lum = rgb.mean(-1)
    t = ndi.uniform_filter(tex, 3)
    exg = 2 * g - r - b
    cls = np.full(r.shape, 6, np.uint8)
    paved = ch < 22
    cls[paved & (lum >= 118)] = 4
    cls[paved & (lum < 118)] = 5
    soil = (r > b + 14) & (ch >= 22) & (exg < 14)
    cls[soil] = 3
    green = exg >= 14
    cls[green] = 1
    # дерева: текстура + кольоровість (зелені, жовті, помаранчеві крони)
    autumn = (r > b + 40) & (ch > 45) & (r >= g - 5)
    tree = (t > 9.5) & (green | autumn) | (green & (lum < 70) & (t > 6))
    cls[tree] = 2
    cls[mx < 52] = 0
    return cls


def main():
    meta = json.load(open('build/sat_map.json'))
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    rgb, tex = features(sat)
    cls = classify(rgb, tex)
    # чистка: дерева — відкриття/закриття, решта — мажоритарний фільтр
    tr = ndi.binary_closing(cls == 2, iterations=1); tr = ndi.binary_opening(tr, iterations=1)
    nt = cls.copy(); nt[cls == 2] = 6
    # мажоритарний 5x5 по класах 1,3,4,5
    cnt = np.stack([ndi.uniform_filter((nt == k).astype(np.float32), 5) for k in range(7)])
    maj = cnt.argmax(0).astype(np.uint8)
    maj[tr] = 2
    # тінь заповнити найближчим не-тіньовим класом / кольором
    sh = maj == 0
    _, (iy, ix) = ndi.distance_transform_edt(sh, return_indices=True)
    maj = maj[iy, ix]
    rgbf = rgb[iy, ix]
    np.savez_compressed('build/satcls.npz', cls=maj, rgb=rgbf.clip(0, 255).astype(np.uint8), tex=tex.astype(np.float16),
                        shadow=sh, F=F)
    print('classes', np.bincount(maj.ravel(), minlength=7) / maj.size)


if __name__ == '__main__':
    main()
