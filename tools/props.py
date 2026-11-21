"""Об'єкти гри замість намальованих у карті (15.11.2026, рішення Івана): лавки, урни, контейнери, дитмайданчики, шлагбауми.

Спрайти малюються ТИМ САМИМ кодом citydetail (вигляд не змінюється), на прозорому полотні:
два проходи (чорне й біле тло) дають колір і прозорість. Текстура RGBA 8888, 2 тексели на піксель карти,
розмір — степінь двійки, центр об'єкта — центр текстури, вісь u — вправо (кут 0)."""
import math
import numpy as np, cv2
import citydetail as CDt
import citydraw as CD

TPX = 2                     # текселів на піксель карти
SS = 4                      # надвибірка при малюванні
KINDS = ['bench', 'urn', 'bins', 'pg_slide', 'pg_swing', 'pg_sand', 'pg_climb', 'pg_carousel', 'pg_yard', 'barrier', 'lamp', 'stop']
# 21.11.2026: перешкоди — півдовжина й півширина в метрах уздовж осі об'єкта (0 — не заважає): машина й пішохід не проходять
COLL = {'bench': (0.9, 0.3), 'urn': (0.35, 0.35), 'bins': (1.6, 0.7), 'pg_slide': (1.4, 0.5), 'pg_swing': (1.2, 0.3),
        'pg_climb': (1.0, 1.0), 'pg_carousel': (1.0, 1.0), 'pg_sand': (0, 0), 'pg_yard': (0, 0), 'barrier': (0, 0),
        'lamp': (0.25, 0.25), 'stop': (1.8, 0.7)}


class _RS:
    """замінник random.Random з наперед заданими значеннями (варіант спрайта)"""
    def __init__(self, vals): self.v = list(vals); self.i = 0
    def random(self):
        r = self.v[self.i % len(self.v)]; self.i += 1; return r
    def randrange(self, n):
        r = self.i % n; self.i += 1; return r


def _barrier(img, c, x, y, L):
    """бетонні блоки поперек дороги: червоні й білі по черзі, з тінню"""
    n = max(2, int(round(L / 1.6)))
    for k in range(n):
        ox = (k - (n - 1) / 2) * (L / n)
        col = (200, 40, 36) if k % 2 == 0 else (236, 236, 230)
        CDt._rect(img, x + c.m(ox) + c.m(0.3), y + c.m(0.4), c.m(L / n * 0.86), c.m(0.9), 1, 0, (70, 70, 70))
        CDt._rect(img, x + c.m(ox), y, c.m(L / n * 0.86), c.m(0.9), 1, 0, col)
        CDt._rect(img, x + c.m(ox), y - c.m(0.25), c.m(L / n * 0.86), c.m(0.2), 1, 0, tuple(min(255, int(v * 1.12)) for v in col))


def _draw(kind, var, bg):
    N = 1024
    img = np.zeros((N, N, 3), np.uint8); img[:] = bg
    c = CDt.Ctx(img=img, cls=None, S=SS, MF=CD.MPP / (TPX * SS), x0=0, y0=0, props=None)
    x = y = N / 2
    if kind == 'bench':
        CDt._rect(img, x + c.m(0.3), y + c.m(0.4), c.m(1.8), c.m(0.6), 1, 0, (90, 96, 84))
        CDt._rect(img, x, y, c.m(1.8), c.m(0.6), 1, 0, (146, 102, 64))
        CDt._rect(img, x, y + c.m(0.22), c.m(1.8), c.m(0.15), 1, 0, (112, 76, 46))
    elif kind == 'urn': CDt.bin_(c, x, y)
    elif kind == 'bins':
        c.bins_var = var; CDt._bins(img, c, x, y, 1.0, 0.0, _RS([0]))
    elif kind == 'pg_yard':
        CDt._playground(img, c, x, y, 1.0, 0.0, _RS([0.1 if var < 2 else 0.9, 0.1 if var % 2 == 0 else 0.9]))
    elif kind.startswith('pg_'): CDt._pg_item(img, c, kind[3:], x, y, 1.0, 0.0)
    elif kind == 'barrier': _barrier(img, c, x, y, min(var / 4.0, 30.0))
    elif kind == 'lamp':                                                    # стовп у центрі, кронштейн і світильник уздовж +x
        ax, ay = x + c.m(2.0), y
        cv2.line(img, CDt._ip((x + c.m(0.3), y + c.m(0.4))), CDt._ip((ax + c.m(0.3), ay + c.m(0.4))), (96, 96, 96), max(1, int(round(c.m(0.22)))))
        cv2.line(img, CDt._ip((x, y)), CDt._ip((ax, ay)), (70, 72, 78), max(1, int(round(c.m(0.22)))))
        cv2.circle(img, CDt._ip((x, y)), max(1, int(round(c.m(0.32)))), (58, 60, 64), -1)
        cv2.ellipse(img, CDt._ip((ax, ay)), (max(1, int(c.m(0.6))), max(1, int(c.m(0.3)))), 0, 0, 360, (54, 56, 60), -1)
        cv2.ellipse(img, CDt._ip((ax, ay)), (max(1, int(c.m(0.45))), max(1, int(c.m(0.18)))), 0, 0, 360, (226, 226, 214), -1)
    elif kind == 'stop':                                                    # павільйон зупинки: каркас, скляний дах, тінь
        L, Wd = c.m(3.6), c.m(1.4)
        CDt._rect(img, x + c.m(0.25), y + c.m(0.35), L, Wd, 1, 0, (90, 92, 94))
        CDt._rect(img, x, y, L, Wd, 1, 0, (44, 58, 70))
        CDt._rect(img, x, y, L * 0.86, Wd * 0.6, 1, 0, (104, 150, 176))
        cv2.line(img, CDt._ip((x - L * 0.43, y - Wd * 0.3)), CDt._ip((x + L * 0.43, y - Wd * 0.3)), (176, 210, 226), max(1, int(c.m(0.1))))
    return img


def sprite(kind, var=0):
    """-> (RGBA bytes, tw, th, hx, hy) ; hx, hy — половини розміру в пікселях карти"""
    a = _draw(kind, var, (0, 0, 0)).astype(np.float32); b = _draw(kind, var, (255, 255, 255)).astype(np.float32)
    al = 1 - (b - a).mean(-1) / 255.0                                       # прозорість: різниця на чорному й білому тлі
    col = np.where(al[..., None] > 1e-3, a / np.maximum(al[..., None], 1e-3), 0)
    rgba = np.dstack([np.clip(col, 0, 255) * al[..., None], al * 255]).astype(np.float32)   # премножене -> зменшення без облямівки
    N = rgba.shape[0]
    ys, xs = np.nonzero(rgba[..., 3] > 2)
    r = int(max(abs(xs - N / 2).max(), abs(ys - N / 2).max())) + SS * 2 if len(xs) else SS * 4
    rx = int(max(abs(xs - N / 2).max(), 1)) + SS * 2 if len(xs) else r
    ry = int(max(abs(ys - N / 2).max(), 1)) + SS * 2 if len(xs) else r
    rx, ry = min(rx, N // 2), min(ry, N // 2)
    crop = rgba[N // 2 - ry:N // 2 + ry, N // 2 - rx:N // 2 + rx]
    small = cv2.resize(crop, (crop.shape[1] // SS, crop.shape[0] // SS), interpolation=cv2.INTER_AREA)
    p2 = lambda n: 1 << max(3, (n - 1).bit_length())
    tw, th = p2(small.shape[1]), p2(small.shape[0])
    out = np.zeros((th, tw, 4), np.float32)
    oy, ox = (th - small.shape[0]) // 2, (tw - small.shape[1]) // 2
    out[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    A = out[..., 3:4] / 255.0
    out[..., :3] = np.where(A > 1e-3, out[..., :3] / np.maximum(A, 1e-3), 0)       # назад з премноженого
    import sprites
    t = sprites._bleed(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    return t.tobytes(), tw, th, tw / TPX / 2, th / TPX / 2


if __name__ == '__main__':                                                  # огляд усіх спрайтів: build/props_sheet.png
    import sys
    tiles = []
    for k in KINDS:
        for v in ((0, 1, 2, 3) if k in ('bins', 'pg_yard') else ((32,) if k == 'barrier' else (0,))):
            b, tw, th, _, _ = sprite(k, v)
            t = np.frombuffer(b, np.uint8).reshape(th, tw, 4).astype(np.float32)
            bg = np.full((th, tw, 3), 150, np.float32)
            tiles.append(np.clip(bg * (1 - t[..., 3:] / 255) + t[..., :3] * t[..., 3:] / 255, 0, 255).astype(np.uint8))
    H = max(t.shape[0] for t in tiles); sheet = np.hstack([np.pad(t, ((0, H - t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in tiles])
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else 'build/props_sheet.png', cv2.resize(sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)[..., ::-1])
