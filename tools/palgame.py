"""Палітра гри (256): кольори інтерфейсу/спрайтів з pal.py + кольори карти з build/map_colors.json."""
import os, json
import numpy as np
import pal as _old

UI = ['out', 'ink', 'panel', 'white', 'yellow', 'green', 'red', 'box', 'blue', 'skin', 'tire', 'steel']
PAL = [(0, 0, 0)]
R = {}
for n in UI:
    s = len(PAL); PAL.extend(_old.PAL[i] for i in _old.R[n]); R[n] = list(range(s, s + len(_old.R[n])))
# сріблястий кузов Golf
s = len(PAL); PAL.extend([(92, 96, 104), (140, 144, 150), (182, 186, 192), (214, 218, 222), (240, 242, 244)]); R['silver'] = list(range(s, s + 5))
s = len(PAL); PAL.extend([(28, 34, 44), (54, 66, 84), (96, 116, 140)]); R['glass'] = list(range(s, s + 3))
MAP_START = len(PAL)
MAP_MAX = 256 - MAP_START
_mc = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'map_colors.json')
MAP_COLORS = [tuple(c) for c in json.load(open(_mc))] if os.path.exists(_mc) else []
PAL.extend(MAP_COLORS)
assert len(PAL) <= 256, len(PAL)


def dark_lut():
    lut = list(range(256))
    for ids in R.values():
        for k, i in enumerate(ids): lut[i] = ids[max(k - 1, 0)]
    if MAP_COLORS:
        mc = np.array(MAP_COLORS, float)
        for k, c in enumerate(mc):
            d = ((mc - c * 0.72) ** 2).sum(1); lut[MAP_START + k] = MAP_START + int(d.argmin())
    return lut


def offroad():
    off = [0] * 256
    for k, (r, g, b) in enumerate(MAP_COLORS):
        if (g > r + 8 and g > b + 8) or (r > b + 25 and r < 175 and g > b + 10 and r - g < 40 and r > g): off[MAP_START + k] = 1
    return off
