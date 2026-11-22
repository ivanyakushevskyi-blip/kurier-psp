"""Прототип (22.11.2026): деталізація карти для пішохода (камера 2×).

Три панелі однієї ділянки, як на екрані PSP (480×272):
  A — зараз: карта 1× (0,278 м/пікс), збільшена 2× піксель у піксель;
  B — «детальні текстури»: колір з карти 1× + маска поверхонь 2× + дрібний візерунок поверхні (збирається на льоту);
  C — еталон: уся карта вдвічі детальніша (0,139 м/пікс), як окремий великий файл.
Плюс смуга «як це працює»: колір 1×, маска 2×, візерунки, результат.
Запуск: python3 tools/proto_detail.py X Y  (центр у пікселях карти 1×). Файли — у build/proto/.
"""
import os, sys, inspect, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import citydraw as CD

cx, cy = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (13040, 14660)
W, H, M = 240, 136, 96
x0, y0 = cx - W // 2, cy - H // 2
OUT = 'build/proto'; os.makedirs(OUT, exist_ok=True)

# --- рендер тим самим кодом карти, але з перехопленням проміжного 4× (без зміни citydraw.py — кеш блоків не скидається)
src = inspect.getsource(CD.render)
old = "        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)\n        cls = clsb.copy()"
assert old in src
src = src.replace(old, "        _CAP['img2'] = cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_AREA)\n"
                       "        _CAP['gnd2'] = cv2.resize(img_ground, (W * 2, H * 2), interpolation=cv2.INTER_AREA)\n"
                       "        _CAP['gnd1'] = cv2.resize(img_ground, (W, H), interpolation=cv2.INTER_AREA)\n"
                       "        _CAP['cls2'] = cls[S // 4::S // 2, S // 4::S // 2].copy()\n" + old)
src = src.replace("def render(", "def render_cap(", 1)
CD.__dict__['_CAP'] = {}
exec(compile(src, CD.__file__, 'exec'), CD.__dict__)
tl, so, pr = [], [], []
img1, cls1 = CD.render_cap(x0 - M, y0 - M, W + 2 * M, H + 2 * M, tree_out=tl, ss=4, solid_out=so, props_out=pr)
C = CD._CAP
img1 = img1[M:M + H, M:M + W].copy(); cls1 = cls1[M:M + H, M:M + W].copy()
img2 = C['img2'][2 * M:2 * (M + H), 2 * M:2 * (M + W)].copy()
gnd2 = C['gnd2'][2 * M:2 * (M + H), 2 * M:2 * (M + W)]
gnd1 = C['gnd1'][M:M + H, M:M + W]
cls2 = C['cls2'][2 * M:2 * (M + H), 2 * M:2 * (M + W)].copy()

# дахи (і все, що поверх землі): там, де кадр відрізняється від «землі»
roof1 = (np.abs(img1.astype(int) - gnd1.astype(int)).sum(2) > 12)
roof2 = (np.abs(img2.astype(int) - gnd2.astype(int)).sum(2) > 12)
ROOF = 99
c1 = np.where(roof1, ROOF, cls1).astype(np.int32)
c2 = np.where(roof2, ROOF, cls2).astype(np.int32)

# --- B: колір з 1× з урахуванням меж (кожна поверхня бере колір лише від своїх пікселів) + візерунок
f1 = img1.astype(np.float32)
# кожен піксель 2× бере колір свого 1×-пікселя, а якщо той іншої поверхні — найближчого 1×-пікселя тієї ж поверхні
B = cv2.resize(f1, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
up = cv2.resize(c1.astype(np.float32), (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
for k in np.unique(c2):
    sel = (c2 == k) & (up != k)
    if not sel.any(): continue
    m1 = (c1 == k).astype(np.uint8)
    if m1.sum() == 0: continue
    _, lab = cv2.distanceTransformWithLabels(1 - m1, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    zs = np.flatnonzero(m1.ravel())
    ys, xs = np.nonzero(sel)
    L = lab[np.minimum(ys // 2, H - 1), np.minimum(xs // 2, W - 1)]
    pz = zs[np.clip(L - 1, 0, len(zs) - 1)]
    B[ys, xs] = f1.reshape(-1, 3)[pz]
# розмітка — окремий клас маски 2× (у справжньому конвеєрі citydraw знає, де малює фарбу): біла й жовта
f2 = img2.astype(np.float32); lum2 = f2.mean(2); loc2 = cv2.blur(lum2, (9, 9))
yel = (f2[..., 0] > 150) & (f2[..., 1] > 120) & (f2[..., 2] < 110)
road2 = cv2.erode((c2 == 5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # лише всередині проїжджої частини
wht = (lum2 > loc2 + 30) & (lum2 > 150) & road2
mark2 = ((yel & (c2 == 5)) | wht) & ~roof2
paint = np.where(yel[..., None], np.array([226, 184, 46], np.float32), np.array([222, 224, 220], np.float32))
# візерунки поверхонь у сітці 2× (у грі — плитки 64×64 у ресурсах, тут — та сама формула)
G = dict(zip('YARD GRASS WATER SIDE ROAD PATH PED RAIL DIRT PITCH TRACK PARK PAVED WOOD SAND'.split(), range(1, 16)))
yy, xx = np.mgrid[0:2 * H, 0:2 * W]
gx, gy = xx + 2 * x0, yy + 2 * y0                                        # світові координати 2× — візерунок не «пливе»
rng = np.random.default_rng(7)
def hash_noise(sc, seed):
    h = (np.sin((gx // sc) * 12.9898 + (gy // sc) * 78.233 + seed) * 43758.5453) % 1.0
    return h
n1, n2, n3 = hash_noise(1, 0), hash_noise(2, 3), hash_noise(1, 9)
det = np.ones((2 * H, 2 * W), np.float32)
def on(*names): return np.isin(c2, [G[n] for n in names])
# асфальт: зерно + рідкі світлі камінчики
m = on('ROAD'); det[m] = (0.94 + 0.10 * n1 + 0.04 * (n3 > 0.93))[m]
# тротуарна плитка 0,55 м (4 пікс), ряди зі зсувом; шви темніші
m = on('SIDE', 'PED', 'PAVED', 'PATH')
row = gy // 5; col = (gx + 2 * (row % 2)) // 5
seam = ((gy % 5) == 0) | (((gx + 2 * (row % 2)) % 5) == 0)
tile = 0.97 + 0.06 * ((np.sin(row * 7.1 + col * 3.3) * 999) % 1.0)
det[m] = np.where(seam, 0.92, tile)[m]
# трава: дрібні травинки, світлі й темні
m = on('GRASS', 'PARK', 'YARD', 'PITCH')
det[m] = (0.88 + 0.22 * n1 * (0.6 + 0.4 * n2))[m]
# ґрунт, пісок, доріжки: крупка
m = on('DIRT', 'SAND', 'TRACK', 'WOOD')
det[m] = (0.9 + 0.16 * n1 + 0.05 * (n3 > 0.9))[m]
# дахи: легка фактура, вздовж — смужки (шифер/профнастил) за кольором не вгадаєш, тож лише зерно
m = c2 == ROOF; det[m] = (0.96 + 0.06 * n1)[m]
Bd = np.clip(B * det[..., None], 0, 255).astype(np.uint8)
Bd[mark2] = paint[mark2].astype(np.uint8)

A = cv2.resize(img1, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
Cr = img2

def to565ish(a):                                                         # як на PSP (5 біт на канал)
    return ((a.astype(np.uint16) >> 3) << 3).astype(np.uint8)
for name, im in (('A_zaraz', A), ('B_detalni_tekstury', Bd), ('C_karta_2x', Cr)):
    cv2.imwrite(os.path.join(OUT, name + '.png'), cv2.cvtColor(to565ish(im), cv2.COLOR_RGB2BGR))

# --- смуга «як це працює»
pal = np.array([[0, 0, 0]] + [list(np.random.default_rng(i).integers(60, 230, 3)) for i in range(1, 16)], np.uint8)
mv = pal[np.clip(c2, 0, 15)]; mv[c2 == ROOF] = (40, 40, 40); mv[mark2] = paint[mark2].astype(np.uint8)
dv = np.clip((det - 0.8) / 0.4 * 255, 0, 255).astype(np.uint8); dv = np.stack([dv] * 3, 2)
crop = lambda a: a[60:60 + 136, 150:150 + 240]
strip = [cv2.resize(crop(cv2.resize(img1, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)), (480, 272), interpolation=cv2.INTER_NEAREST),
         cv2.resize(crop(mv), (480, 272), interpolation=cv2.INTER_NEAREST),
         cv2.resize(crop(dv), (480, 272), interpolation=cv2.INTER_NEAREST),
         cv2.resize(crop(Bd), (480, 272), interpolation=cv2.INTER_NEAREST)]
for i, s in enumerate(strip):
    cv2.imwrite(os.path.join(OUT, 'how_%d.png' % i), cv2.cvtColor(s, cv2.COLOR_RGB2BGR))
print('ok', OUT, 'класи 2×:', sorted(set(np.unique(c2).tolist())))
