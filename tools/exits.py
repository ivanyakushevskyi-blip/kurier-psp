"""Шлагбауми на всіх великих виїздах (15.11.2026, Іван: «та додай ти вже ті шлагбауми»).

Великі дороги (MAINX у citybuild) за межею зони — стіна (бічні вулиці — ні, вони йдуть у затемнення).
Тут кожна велика дорога OSM проходиться вздовж; де вона виходить з зони (build/play.npy) — ряд червоно-білих
бетонних блоків поперек дороги впритул до стіни. Пише записи 'barrier' у build/props.json (старі прибирає).
Карту не чіпає: після нього досить REUSE_MAP=1 bake.py (лише EBOOT).
  python3 tools/exits.py
"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import citydraw as CD

MAINX = ('primary', 'trunk', 'secondary', 'tertiary', 'primary_link', 'secondary_link', 'tertiary_link', 'trunk_link')


def main():
    play = np.load('build/play.npy', mmap_mode='r'); H, W = play.shape
    D = CD.data(); K = CD.K; F = 1.0 / CD.MPP
    inside = lambda x, y: 0 <= int(x) < W and 0 <= int(y) < H and bool(play[int(y), int(x)])
    found = []
    for w in D['w']:
        h = w['t'].get('highway')
        if h not in MAINX or w['t'].get('tunnel') == 'yes' or w['t'].get('area') == 'yes': continue
        P = [(p[0] * K, p[1] * K) for p in w['pts']]
        width = CD.ROAD_W.get(h, CD.ROAD_W.get('primary', 13)) if h != 'trunk' else 14
        # уся дорога точками через 1 м: координата, напрямок, «у зоні»
        S_ = []
        for i in range(len(P) - 1):
            (xa, ya), (xb, yb) = P[i], P[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < 1: continue
            ux, uy = (xb - xa) / L, (yb - ya) / L
            for s in np.arange(0.0, L, 1.0): S_.append((xa + ux * s, ya + uy * s, ux, uy))
        if len(S_) < 3: continue
        ins = np.array([inside(x, y) for x, y, _, _ in S_])
        # серії однакових значень; вихід — перехід між серією «в зоні» (≥20 м або від початку) і серією «поза» (≥60 м або до кінця, ≥15 м)
        ch = np.nonzero(ins[1:] != ins[:-1])[0] + 1
        b = np.concatenate([[0], ch, [len(ins)]])
        runs = [(b[k], b[k + 1], bool(ins[b[k]])) for k in range(len(b) - 1)]
        M = lambda m: m * F
        for k in range(len(runs) - 1):
            (a0, a1, va), (c0, c1, vc) = runs[k], runs[k + 1]
            if va == vc: continue
            inr, outr = (runs[k], runs[k + 1]) if va else (runs[k + 1], runs[k])
            in_ok = (inr[1] - inr[0]) >= M(20) or inr[0] == 0 or inr[1] == len(ins)
            out_len = outr[1] - outr[0]
            out_ok = out_len >= M(60) or outr[0] == 0 or outr[1] == len(ins)          # до кінця відрізка OSM — далі інша лінія
            if not (in_ok and out_ok): continue
            x, y, ux, uy = S_[a1]
            ox, oy = (ux, uy) if va else (-ux, -uy)                       # назовні
            cx, cy = x - ux * 0.5 + ox * 0.7 * F, y - uy * 0.5 + oy * 0.7 * F
            found.append((cx, cy, math.atan2(oy, ox) + math.pi / 2, width + 1.5))
    # злити близькі (дві смуги однієї дороги, сусідні відрізки): у межах 30 м — один, довжиною на обидва
    out = []
    for f in sorted(found, key=lambda f: -f[3]):
        for g in out:
            if math.hypot(f[0] - g[0], f[1] - g[1]) < 30 * F: break
        else: out.append(f)
    try: props = json.load(open('build/props.json'))
    except FileNotFoundError: props = []
    props = [p for p in props if p[3] != 'barrier']
    for x, y, a, Lm in out:
        props.append([round(x, 1), round(y, 1), round(a, 4), 'barrier', int(round(min(Lm, 30) * 4))])
    json.dump(props, open(os.environ.get('PROPS_OUT', 'build/props.json'), 'w'))
    print('шлагбаумів на виїздах: %d' % len(out))
    for x, y, a, Lm in out: print('  %6d %6d  %.0f м' % (x, y, Lm))


if __name__ == '__main__':
    main()
