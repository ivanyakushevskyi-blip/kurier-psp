"""Огляд меж ігрової зони на схемі OSM (для погодження з Іваном).
python3 -I tools/boundsview.py out.png [bounds_new.json] [x0 y0 x1 y1 (м)] [м/пікс]
Сіра — поточна межа (data/bounds_v3.json), жовта — нова."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import satmosaic as S

LAT0 = 49.422; MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0)); LON0 = 26.995
COL = {'primary': (250, 160, 60), 'trunk': (250, 160, 60), 'secondary': (246, 210, 90), 'tertiary': (240, 240, 200),
       'residential': (200, 200, 200), 'unclassified': (190, 190, 190), 'living_street': (170, 170, 170), 'service': (120, 120, 120)}
WID = {'primary': 4, 'trunk': 4, 'secondary': 3, 'tertiary': 3, 'residential': 2, 'unclassified': 2, 'living_street': 1, 'service': 1}


def ll2m(lo, la): return ((lo - LON0) * MLON, -(la - LAT0) * MLAT)


def main():
    out = sys.argv[1]
    new = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    x0, y0, x1, y1 = map(float, sys.argv[3:7]) if len(sys.argv) > 6 else (-4400, -4700, 4600, 4200)
    s = float(sys.argv[7]) if len(sys.argv) > 7 else 4.0
    W, H = int((x1 - x0) / s), int((y1 - y0) / s)
    P = lambda p: ((p[0] - x0) / s, (p[1] - y0) / s)
    im = Image.new('RGB', (W, H), (34, 38, 34)); d = ImageDraw.Draw(im)
    ways = S.load_osm('data/osm_v3')
    for p, t in ways:                                                   # ліс, вода, забудова
        if len(p) < 3: continue
        c = None
        if t.get('natural') == 'wood' or t.get('landuse') == 'forest': c = (40, 80, 44)
        elif t.get('natural') == 'water' or t.get('water'): c = (50, 80, 140)
        elif t.get('landuse') in ('residential',): c = (52, 54, 56)
        elif t.get('building'): c = (90, 86, 82)
        if c: d.polygon([P(q) for q in p], fill=c)
    for p, t in ways:
        h = t.get('highway')
        if h in COL and len(p) >= 2: d.line([P(q) for q in p], fill=COL[h], width=WID[h])
        if t.get('railway') == 'rail' and len(p) >= 2: d.line([P(q) for q in p], fill=(150, 90, 90), width=2)
    old = [P(ll2m(lo, la)) for lo, la in json.load(open('data/bounds_v3.json'))]
    mk = Image.new('L', (W, H), 0); ImageDraw.Draw(mk).polygon(old, fill=255)
    dim = Image.eval(im, lambda v: v // 3)
    im.paste(dim, (0, 0), Image.eval(mk, lambda v: 255 - v)); d = ImageDraw.Draw(im)
    d.line(old + [old[0]], fill=(255, 60, 60), width=3)
    if new:
        nw = [P(ll2m(lo, la)) for lo, la in json.load(open(new))]
        d.line(nw + [nw[0]], fill=(255, 220, 0), width=4)
    try:
        fc = ImageFont.truetype('tools/fonts/rubik-cyrillic-700.ttf', 14)
    except Exception: fc = None
    done = set()
    for p, t in ways:                                                   # підписи головних вулиць
        n = t.get('name'); h = t.get('highway')
        if not n or h not in ('primary', 'secondary', 'tertiary', 'trunk') or n in done or len(p) < 2: continue
        q = P(p[len(p) // 2])
        if 0 < q[0] < W and 0 < q[1] < H:
            done.add(n); d.text(q, n.replace('вулиця ', '').replace(' вулиця', ''), fill=(255, 255, 255), font=fc)
    im.save(out)


if __name__ == '__main__':
    main()
