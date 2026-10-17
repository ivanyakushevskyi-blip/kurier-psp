"""Ігрова зона: межі за вулицями (Зарічанська, Старокостянтинівське ш., Кам'янецька + Бандери, залізниця).
Записує build/bounds.npz: play (де можна їздити) і margin (що малюємо з запасом), сітка 2 м."""
import glob, math, sys, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

LAT0 = 49.422; MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
LON0, LON1, LAT_0, LAT1 = 26.958, 27.022, 49.405, 49.442          # область завантажених даних
G = 2.0
MARGIN_M = 180                                                     # скільки міста малюємо за межею


def grid():
    W = int((LON1 - LON0) * MLON / G); H = int((LAT1 - LAT_0) * MLAT / G)
    return W, H


def P(lat, lon):
    return ((lon - LON0) * MLON / G, (LAT1 - lat) * MLAT / G)


def is_boundary(t, ll):
    nm = t.get('name', ''); h = t.get('highway')
    if not h: return False
    if 'Зарічанська' in nm or 'Старокостянтинівське' in nm: return True
    if "Кам'янецька" in nm or 'Камʼянецька' in nm: return True
    if nm == 'вулиця Степана Бандери' and max(q[0] for q in ll) < 49.4375: return True
    return False


def main(osm_dir, out):
    W, H = grid()
    nodes, ways = {}, {}
    for f in sorted(glob.glob(osm_dir + '/t*.osm')):
        r = ET.parse(f).getroot()
        for n in r.iter('node'): nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
        for w in r.iter('way'): ways[w.get('id')] = ([nd.get('ref') for nd in w.iter('nd')], {x.get('k'): x.get('v') for x in w.iter('tag')})
    wall = Image.new('L', (W, H), 0); dw = ImageDraw.Draw(wall)
    rail = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(rail)
    sk_south = None
    for nds, t in ways.values():
        ll = [nodes[n] for n in nds if n in nodes]
        if len(ll) < 2: continue
        p = [P(*q) for q in ll]
        if is_boundary(t, ll):
            dw.line(p, fill=1, width=9)
            if 'Старокостянтинівське' in t.get('name', ''):
                s = min(ll, key=lambda q: q[0])
                if sk_south is None or s[0] < sk_south[0]: sk_south = s
        if t.get('railway') == 'rail' and t.get('service') not in ('yard', 'siding', 'spur'):
            dw.line(p, fill=1, width=7); dr.line(p, fill=1, width=3)
    # Старокостянтинівське закінчується за кілька метрів до колії — дотягуємо до найближчої рейки
    ra = np.array(rail) > 0
    ys, xs = np.nonzero(ra)
    sx, sy = P(*sk_south)
    k = np.argmin((xs - sx) ** 2 + (ys - sy) ** 2)
    dw.line([(sx, sy), (xs[k], ys[k])], fill=1, width=9)
    wa = np.array(wall) > 0
    cx, cy = P(49.422, 26.995)                                     # центр (Проскурівська)
    lab, _ = ndi.label(~wa)
    region = lab == lab[int(cy), int(cx)]
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        sys.exit('межа не замкнена: зона витікає до краю даних')
    play = ndi.binary_dilation(region, iterations=6)               # разом із самими межовими вулицями
    margin = ndi.distance_transform_edt(~play) * G <= MARGIN_M
    np.savez_compressed(out, play=play, margin=margin, G=G, lon0=LON0, lat1=LAT1, mlon=MLON, mlat=MLAT)
    print('ігрова зона %.2f км², з запасом %.2f км²' % (play.sum() * G * G / 1e6, margin.sum() * G * G / 1e6))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
