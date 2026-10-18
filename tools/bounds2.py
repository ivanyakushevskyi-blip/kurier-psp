"""Ігрова зона v2 (18.10.2026): полігон меж, погоджений з Іваном (data/bounds_v2.json, [lon, lat]):
Захід — Західно-Окружна, Південь — залізниця, Схід — вул. Крамара (Трудова) з Автовокзалом і Дендропарком,
Північ — Озерна до Панаса Мирного / Січових стрільців.
Записує build/bounds.npz у тому ж форматі, що й bounds.py: play, margin (сітка 2 м)."""
import json, math, sys
import numpy as np, cv2
from scipy import ndimage as ndi

LAT0 = 49.422; MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
LON0, LON1, LAT_0, LAT1 = 26.925, 27.045, 49.395, 49.470          # data/osm_v3
G = 2.0
GROW_M = 22            # межові вулиці входять у зону цілком
MARGIN_M = 180


def main(src='data/bounds_v2.json', out='build/bounds.npz'):
    W = int((LON1 - LON0) * MLON / G); H = int((LAT1 - LAT_0) * MLAT / G)
    poly = np.array([((lo - LON0) * MLON / G, (LAT1 - la) * MLAT / G) for lo, la in json.load(open(src))], np.float32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly * 8).astype(np.int32)], 1, shift=3)
    d = ndi.distance_transform_edt(m == 0) * G
    play = d <= GROW_M
    margin = ndi.distance_transform_edt(~play) * G <= MARGIN_M
    np.savez_compressed(out, play=play, margin=margin, G=G, lon0=LON0, lat1=LAT1, mlon=MLON, mlat=MLAT)
    print('ігрова зона %.2f км², з запасом %.2f км²' % (play.sum() * G * G / 1e6, margin.sum() * G * G / 1e6))


if __name__ == '__main__':
    main(*sys.argv[1:])
