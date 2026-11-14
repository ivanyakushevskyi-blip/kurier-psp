"""Перевірка карти перед здачею (15.11.2026, рішення Івана: «нічого не має поїхати»).

  python3 tools/mapcheck.py NEW.BIN [OLD.BIN] [out_dir]

1) Порівняння з попередньою картою: які чанки 64×64 змінились (теплокарта changes.png, червоне — змінено),
   місця змін згруповані в плями; для найбільших — пари «було | стало» 1:1 (diff_NN.png) і список у changes.txt.
2) Шви: по межах блоків генератора (кожні 512 пікс) шукає відрізки, де стрибок кольору через межу набагато
   більший, ніж поруч (ознака склейки різних рендерів). Вирізки seam_NN.png, список у seams.txt.
"""
import sys, os, struct, zlib
import numpy as np, cv2
from scipy import ndimage as ndi


class Map:
    def __init__(self, fn):
        self.f = open(fn, 'rb')
        assert self.f.read(4) == b'KHM3'
        self.NCX, self.NCY, self.CS = struct.unpack('<3I', self.f.read(12))
        n = self.NCX * self.NCY
        self.off = struct.unpack('<%dI' % (n + 1), self.f.read(4 * (n + 1)))

    def raw(self, cx, cy):
        i = cy * self.NCX + cx
        self.f.seek(self.off[i]); return self.f.read(self.off[i + 1] - self.off[i])

    def rgb(self, cx, cy):
        CS = self.CS
        p = np.frombuffer(zlib.decompress(self.raw(cx, cy))[:CS * CS * 2], '<u2').reshape(CS, CS)
        return np.stack([(p & 31) << 3, ((p >> 5) & 31) << 3, ((p >> 10) & 31) << 3], -1).astype(np.uint8)

    def crop(self, x0, y0, w, h):
        CS = self.CS; out = np.zeros((h, w, 3), np.uint8)
        for cy in range(max(0, y0 // CS), min(self.NCY, (y0 + h - 1) // CS + 1)):
            for cx in range(max(0, x0 // CS), min(self.NCX, (x0 + w - 1) // CS + 1)):
                c = self.rgb(cx, cy); X, Y = cx * CS - x0, cy * CS - y0
                xa, ya = max(0, X), max(0, Y); xb, yb = min(w, X + CS), min(h, Y + CS)
                out[ya:yb, xa:xb] = c[ya - Y:yb - Y, xa - X:xb - X]
        return out


def diff(new, old, out):
    NCX, NCY, CS = new.NCX, new.NCY, new.CS
    ch = np.zeros((NCY, NCX), np.float32)
    for cy in range(NCY):
        for cx in range(NCX):
            if new.raw(cx, cy) == old.raw(cx, cy): continue
            a, b = new.rgb(cx, cy), old.rgb(cx, cy)
            ch[cy, cx] = (np.abs(a.astype(np.int16) - b).max(-1) > 16).mean()
    np.save(os.path.join(out, 'changes.npy'), ch)
    hm = np.zeros((NCY, NCX, 3), np.uint8); hm[...] = 40
    hm[ch > 0] = (90, 90, 200); hm[ch > 0.02] = (0, 0, 255)
    cv2.imwrite(os.path.join(out, 'changes.png'), cv2.resize(hm, (NCX * 3, NCY * 3), interpolation=cv2.INTER_NEAREST))
    lab, n = ndi.label(ch > 0.02, np.ones((3, 3)))
    spots = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        m = lab[sl] == i + 1
        spots.append((float(ch[sl][m].sum()), sl))
    spots.sort(key=lambda s: -s[0])
    with open(os.path.join(out, 'changes.txt'), 'w') as f:
        f.write('змінених чанків %d з %d (%.1f%%), плям %d\n' % ((ch > 0.02).sum(), ch.size, 100 * (ch > 0.02).mean(), n))
        for k, (s, sl) in enumerate(spots[:40]):
            y0, x0 = sl[0].start * CS, sl[1].start * CS; h, w = (sl[0].stop - sl[0].start) * CS, (sl[1].stop - sl[1].start) * CS
            f.write('%2d  x %d..%d  y %d..%d  вага %.1f\n' % (k, x0, x0 + w, y0, y0 + h, s))
            if k < 12:
                cxp, cyp = x0 + w // 2, y0 + h // 2; W_, H_ = min(max(w, 320), 640), min(max(h, 240), 480)
                a = old.crop(cxp - W_ // 2, cyp - H_ // 2, W_, H_); b = new.crop(cxp - W_ // 2, cyp - H_ // 2, W_, H_)
                cv2.imwrite(os.path.join(out, 'diff_%02d.png' % k), np.hstack([a, np.full((H_, 6, 3), 255, np.uint8), b])[..., ::-1])
    print(open(os.path.join(out, 'changes.txt')).readline().strip())


def seams(m, out, B=512):
    CS = m.CS; W, H = m.NCX * CS, m.NCY * CS
    found = []
    def scan(vertical):
        for k in range(1, (W if vertical else H) // B + 1):
            p = k * B
            for s0 in range(0, H if vertical else W, CS):
                if vertical: a = m.crop(p - 3, s0, 6, CS).astype(np.int16).transpose(1, 0, 2)
                else: a = m.crop(s0, p - 3, CS, 6).astype(np.int16)
                if not a.any(): continue
                d = [np.abs(a[i + 1] - a[i]).sum(-1) for i in range(5)]      # стрибки між сусідніми рядами; межа — між 2 і 3
                across = d[2]; ref = np.maximum(np.maximum(d[0], d[1]), np.maximum(d[3], d[4]))
                bad = (across > ref * 2.5 + 30)
                run = max((len(r) for r in ''.join('1' if b else '0' for b in bad).split('0')), default=0)
                if run >= 24: found.append((run, vertical, p, s0))
    scan(True); scan(False)
    found.sort(reverse=True)
    with open(os.path.join(out, 'seams.txt'), 'w') as f:
        f.write('підозрілих відрізків %d\n' % len(found))
        for k, (run, v, p, s0) in enumerate(found[:40]):
            x, y = (p, s0 + CS // 2) if v else (s0 + CS // 2, p)
            f.write('%2d  %s x %d y %d  довжина %d\n' % (k, 'верт' if v else 'гориз', x, y, run))
            if k < 8: cv2.imwrite(os.path.join(out, 'seam_%02d.png' % k), cv2.resize(m.crop(x - 80, y - 60, 160, 120), (320, 240), interpolation=cv2.INTER_NEAREST)[..., ::-1])
    print('шви: підозрілих відрізків %d' % len(found))


if __name__ == '__main__':
    out = sys.argv[3] if len(sys.argv) > 3 else 'build/check'
    os.makedirs(out, exist_ok=True)
    new = Map(sys.argv[1])
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]): diff(new, Map(sys.argv[2]), out)
    seams(new, out)
