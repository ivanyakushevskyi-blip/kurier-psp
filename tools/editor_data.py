"""Дані для веб-редактора карти (23.11.2026): плитки карти 0,5× (1024 пікс), огляд 1/8, контури будівель з прогнозом типу.
Запуск: python3 tools/editor_data.py OUTDIR"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import citydraw as CD
OUT = sys.argv[1]; os.makedirs(os.path.join(OUT, 't'), exist_ok=True)
m = np.load('build/map_rgb.npy', mmap_mode='r'); H, W = m.shape[:2]
T = 2048                                                                 # плитка: 2048 пікс карти -> 1024 пікс
nx, ny = (W + T - 1) // T, (H + T - 1) // T
ov = np.zeros(((H + 7) // 8, (W + 7) // 8, 3), np.uint8)
for ty in (range(ny) if not os.environ.get('ONLYB') else []):
    for tx in range(nx):
        a = np.asarray(m[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T])
        if a.shape[0] < T or a.shape[1] < T:
            b = np.full((T, T, 3), 200, np.uint8); b[:a.shape[0], :a.shape[1]] = a; a = b
        s = cv2.resize(a, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT, 't', '%d_%d.jpg' % (tx, ty)), cv2.cvtColor(s, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
        o = cv2.resize(a, (T // 8, T // 8), interpolation=cv2.INTER_AREA)
        hh, ww = min(T // 8, ov.shape[0] - ty * T // 8), min(T // 8, ov.shape[1] - tx * T // 8)
        ov[ty * T // 8:ty * T // 8 + hh, tx * T // 8:tx * T // 8 + ww] = o[:hh, :ww]
if not os.environ.get('ONLYB'): cv2.imwrite(os.path.join(OUT, 'overview.jpg'), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
# будівлі: id, центр, контур (спрощений), висота, прогноз типу генератора, підпис
D = CD.data(); K = CD.K; MPP = CD.MPP
def btype(bt, floors, a):
    if bt in ('roof', 'carport'): return 'canopy'
    if bt in ('garage', 'garages', 'shed', 'kiosk', 'service') or a < 25: return 'kiosk' if a < 30 else 'shed'
    if bt in ('retail', 'commercial', 'supermarket', 'mall') or a > 3000: return 'store'
    if bt in ('house', 'detached', 'semidetached_house', 'terrace') or (floors <= 2 and a < 220): return 'house'
    if floors >= 9: return 'panel'
    if floors >= 4: return 'brick'
    return 'old'
out = []
for b in D['B']:
    q = np.array(b['q'], np.float32) * K
    if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > W or q[:, 1].min() > H: continue
    ap = cv2.approxPolyDP(q.reshape(-1, 1, 2), 1.0, True).reshape(-1, 2)
    if len(ap) < 3: ap = q
    a = abs(cv2.contourArea(q)) * MPP * MPP
    h = float(b.get('h') or 6); fl = max(1, int(round(h / 3)))
    t = b.get('t', {})
    lab = t.get('name') or ((t.get('addr:street', '') + ', ' + t.get('addr:housenumber', '')) if t.get('addr:housenumber') else '')
    cx_, cy_ = int(q[:, 0].mean()), int(q[:, 1].mean())
    bid = str(b['id']) if b.get('id') is not None else 'c%d_%d' % (cx_, cy_)   # контури MS без id — ключ за центром
    out.append([bid, cx_, cy_, np.round(ap).astype(int).flatten().tolist(), fl,
                btype(t.get('building', 'yes'), fl, a), lab, int(a)])
with open(os.path.join(OUT, 'buildings.json'), 'w', encoding='utf-8') as f:
    f.write('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in out) + '\n]')
json.dump({'W': W, 'H': H, 'tile': T, 'nx': nx, 'ny': ny, 'mpp': MPP}, open(os.path.join(OUT, 'meta.json'), 'w'))
print('плиток', nx * ny, 'будівель', len(out))
