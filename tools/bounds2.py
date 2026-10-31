"""Ігрова зона v2 (18.10.2026): полігон меж, погоджений з Іваном (data/bounds_v2.json, [lon, lat]):
Захід — Західно-Окружна, Південь — залізниця, Схід — вул. Крамара (Трудова) з Автовокзалом і Дендропарком,
Північ — Озерна до Панаса Мирного / Січових стрільців.
Записує build/bounds.npz у тому ж форматі, що й bounds.py: play, margin (сітка 2 м)."""
import json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy import ndimage as ndi

LAT0 = 49.422; MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
LON0, LON1, LAT_0, LAT1 = 26.925, 27.045, 49.395, 49.470          # data/osm_v3
G = 2.0
GROW_M = 22            # межові вулиці входять у зону цілком
MARGIN_M = 180


ROAD_FULL = {'primary': 19, 'secondary': 17, 'tertiary': 15, 'residential': 11, 'unclassified': 10, 'living_street': 8,
             'primary_link': 11, 'secondary_link': 10, 'tertiary_link': 10, 'trunk': 19}
NEAR_M = 80            # дороги, що йдуть уздовж межі ззовні (до 80 м), входять у зону повністю
GAP_M = 40             # смуга між полігоном і такою дорогою теж входить


def snap_roads(play, d, W, H):
    """Полігон намальований вручну і місцями зрізає межову вулицю (Бандери, Трудова).
    Додаємо до зони дороги поблизу межі на всю ширину (з тротуарами), заповнюємо щілини між ними й полігоном,
    але не заходимо на колію."""
    import satmosaic as S
    from PIL import Image, ImageDraw
    ways = S.load_osm('data/osm_v3')
    ox, oy = (26.995 - LON0) * MLON, (LAT1 - 49.422) * MLAT
    gp = lambda pts: [((x + ox) / G, (y + oy) / G) for x, y in pts]
    rd = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(rd)
    rl = Image.new('L', (W, H), 0); dl = ImageDraw.Draw(rl)
    for p, t in ways:
        if len(p) < 2: continue
        h = t.get('highway')
        if h in ROAD_FULL and t.get('area') != 'yes':
            q = gp(p); w = ROAD_FULL[h] / G + 4
            dr.line(q, fill=1, width=int(round(w)))
            for x, y in q: dr.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=1)
        if t.get('railway') in ('rail',) and t.get('service') not in ('yard', 'siding', 'spur'):
            dl.line(gp(p), fill=1, width=int(round(16 / G)))
    road = np.array(rd, bool); rail = np.array(rl, bool)
    near = road & (d <= NEAR_M)
    k = int(GAP_M / G / 2)
    se = np.hypot(*np.mgrid[-k:k + 1, -k:k + 1]) <= k
    grown = ndi.binary_closing(np.pad(play | near, k), se)[k:-k, k:-k]
    grown = ndi.binary_fill_holes(grown)
    extra = grown & ~play & ~rail
    lab, _ = ndi.label(play | extra)
    keep = np.unique(lab[play]); keep = keep[keep > 0]
    out = np.isin(lab, keep)
    print('  межа: +%.2f км² вздовж доріг' % ((out & ~play).sum() * G * G / 1e6))
    return out


def main(src='data/bounds_v4.json', out='build/bounds.npz'):
    W = int((LON1 - LON0) * MLON / G); H = int((LAT1 - LAT_0) * MLAT / G)
    poly = np.array([((lo - LON0) * MLON / G, (LAT1 - la) * MLAT / G) for lo, la in json.load(open(src))], np.float32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly * 8).astype(np.int32)], 1, shift=3)
    d = ndi.distance_transform_edt(m == 0) * G
    play = d <= GROW_M
    play = snap_roads(play, d, W, H)
    margin = ndi.distance_transform_edt(~play) * G <= MARGIN_M
    np.savez_compressed(out, play=play, margin=margin, G=G, lon0=LON0, lat1=LAT1, mlon=MLON, mlat=MLAT)
    print('ігрова зона %.2f км², з запасом %.2f км²' % (play.sum() * G * G / 1e6, margin.sum() * G * G / 1e6))


if __name__ == '__main__':
    main(*sys.argv[1:])
