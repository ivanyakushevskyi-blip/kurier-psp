import sys, glob
from PIL import Image, ImageDraw
sys.path.insert(0, 'tools')
from osm_load import load
ws = load(sorted(glob.glob('data/osm/t*.osm')))
xs = [p[0] for _, pts, _, _ in ws for p in pts]; ys = [p[1] for _, pts, _, _ in ws for p in pts]
x0, y0 = -950, -900; S = 0.5
W, H = int(1900 * S), int(1800 * S)
im = Image.new('RGB', (W, H), (235, 235, 225)); d = ImageDraw.Draw(im)
def P(pts): return [((x - x0) * S, (y - y0) * S) for x, y in pts]
for _, pts, t, cl in ws:
    if cl and ('landuse' in t or 'leisure' in t or 'natural' in t) and len(pts) > 2:
        c = (190, 225, 170) if t.get('landuse') in ('grass',) or 'leisure' in t or t.get('natural') in ('wood', 'grassland', 'scrub') else (225, 220, 210)
        if t.get('natural') == 'water': c = (150, 190, 230)
        d.polygon(P(pts), fill=c)
for _, pts, t, cl in ws:
    if 'building' in t and len(pts) > 2: d.polygon(P(pts), fill=(170, 160, 150))
W_ = {'primary': 9, 'secondary': 8, 'tertiary': 7, 'residential': 5, 'unclassified': 5, 'living_street': 4, 'service': 3, 'pedestrian': 4, 'footway': 1, 'path': 1}
for _, pts, t, cl in ws:
    h = t.get('highway')
    if h in W_: d.line(P(pts), fill=(250, 250, 250) if h not in ('footway', 'path') else (200, 120, 120), width=W_[h])
    if 'railway' in t and t['railway'] == 'rail': d.line(P(pts), fill=(90, 90, 90), width=2)
    if 'waterway' in t: d.line(P(pts), fill=(100, 150, 230), width=2)
for _, pts, t, cl in ws:
    if t.get('highway') in ('primary', 'secondary', 'tertiary') and t.get('name'):
        m = pts[len(pts) // 2]; d.text(P([m])[0], t['name'][:18], fill=(0, 0, 0))
for gx in range(0, W, 100): d.line([(gx, 0), (gx, H)], fill=(255, 0, 0))
for gy in range(0, H, 100): d.line([(0, gy), (W, gy)], fill=(255, 0, 0))
im.save('build/overview.png')
print(min(xs), max(xs), min(ys), max(ys))
