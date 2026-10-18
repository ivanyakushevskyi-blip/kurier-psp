"""Пам'ятки (пілот 18.10.2026): Стара пожежна вежа (каланча), Подільська, 39.
Два варіанти для порівняння:
  A — строго згори: дах вежі з білим зубчастим парапетом і башточкою + довга тінь-силует (профіль вежі);
  B — «фасад»: вежа намальована спрайтом з південного боку (як у класичних 2D-іграх), основа — на місці вежі.
Масштаб гри 0,5 м/пікс. Запуск: python3 -I tools/landmarks.py pilot <out_dir>
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

OUT = (34, 34, 42)
Y = [(150, 112, 40), (196, 156, 62), (226, 190, 92), (242, 214, 128)]      # жовта штукатурка: тінь .. світло
WH = [(150, 150, 150), (200, 200, 196), (238, 238, 232)]                   # білий декор
RED = [(120, 40, 34), (160, 56, 44), (196, 76, 58)]
WIN = (52, 58, 74)
SUN = [(214, 150, 20), (246, 196, 40)]
DIAL = (70, 110, 190)

KAL = dict(name='Стара пожежна вежа', lat=49.4269138, lon=26.9814038, w=14, h_m=31)


def tower_facade():
    """Південний фасад вежі + вузька східна грань (об'єм). Повертає RGBA (h, w, 4), низ = основа."""
    W, H = 22, 66
    a = np.zeros((H, W, 4), np.uint8)
    def px(x, y, c):
        if 0 <= x < W and 0 <= y < H: a[y, x, :3] = c; a[y, x, 3] = 255
    def rect(x0, y0, x1, y1, c):
        for y in range(y0, y1):
            for x in range(x0, x1): px(x, y, c)
    cx = 10                                       # вісь фасаду; фасад x 3..17, східна грань 18..20
    # ствол вежі
    rect(4, 22, 17, 66, Y[2]); rect(17, 22, 20, 66, Y[1])
    # білі руста по кутах (через ряд)
    for y in range(24, 66, 3):
        rect(3, y, 6, y + 2, WH[2]); rect(15, y, 17, y + 2, WH[1]); rect(17, y, 19, y + 2, WH[0])
    # вікна ствола
    rect(9, 46, 12, 51, WIN); rect(9, 56, 12, 61, WIN); px(10, 45, WH[2]); px(11, 45, WH[2])
    # годинник-сонце
    cy = 37
    for k in range(16):
        ang = k * math.pi / 8
        r = 5.5 if k % 2 == 0 else 4.2
        px(int(round(cx + 0.5 + math.cos(ang) * r)), int(round(cy + math.sin(ang) * r)), SUN[0])
    for y in range(cy - 5, cy + 6):
        for x in range(cx - 5, cx + 7):
            d = math.hypot(x - cx - 0.5, y - cy)
            if d <= 4.3: px(x, y, SUN[1])
            if d <= 3.0: px(x, y, WH[2])
            if d <= 2.4: px(x, y, DIAL)
    px(cx, cy - 1, WH[2]); px(cx, cy, WH[2]); px(cx + 1, cy, WH[2])          # стрілки
    # карниз під верхнім ярусом
    rect(2, 21, 20, 23, WH[2]); rect(18, 21, 21, 23, WH[1])
    # верхній ярус з арковими вікнами
    rect(4, 11, 17, 21, Y[3]); rect(17, 11, 20, 21, Y[1])
    for x0 in (6, 12):
        rect(x0, 14, x0 + 3, 20, WIN); px(x0, 13, WH[2]); px(x0 + 1, 12, WH[2]); px(x0 + 2, 13, WH[2])
    # балкон-парапет із зубцями
    rect(1, 8, 21, 11, WH[1]); rect(1, 8, 21, 9, WH[2])
    for x in range(1, 21, 2): rect(x, 6, x + 1, 8, WH[2])
    # башточка нагорі
    rect(6, 1, 15, 8, Y[3]); rect(15, 1, 17, 8, Y[1]); rect(8, 3, 10, 7, WIN); rect(11, 3, 13, 7, WIN)
    for x in range(6, 17, 2): px(x, 0, WH[2])
    # обвідка
    m = a[..., 3] > 0
    ol = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
    a[ol, :3] = OUT; a[ol, 3] = 255
    return a


def tower_profile():
    """Ширина вежі (пікс) на кожній висоті (у пікс від основи) — для тіні-силуету."""
    prof = []
    for h in range(62):
        if h < 44: w = 14            # ствол
        elif h < 46: w = 18          # карниз
        elif h < 55: w = 14          # верхній ярус
        elif h < 58: w = 20          # балкон
        elif h < 60: w = 20 if h % 2 else 0
        else: w = 10                 # башточка
        prof.append(w)
    return prof


def blit_rgba(img, spr, x, y):
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]; m = s[..., 3] > 0
    img[y0:y1, x0:x1][m] = s[..., :3][m]


def variant_a(img, bx, by):
    """bx, by — центр основи вежі у вікні. Тінь на південний схід (як у грі), довжина ~0,75 від висоти."""
    prof = tower_profile()
    sh = np.zeros(img.shape[:2], np.uint8)
    for h, w in enumerate(prof):
        if w == 0: continue
        d = h * 0.75 / math.sqrt(2)
        cx, cy = bx + d, by + d
        cv2.line(sh, (int(cx - w / 2 * 0.7), int(cy + w / 2 * 0.7)), (int(cx + w / 2 * 0.7), int(cy - w / 2 * 0.7)), 1, 2)
    sh = sh.astype(bool)
    img[sh] = (img[sh] * 0.62).astype(np.uint8)
    # дах вежі: квадрат 14x14 з білим зубчастим парапетом, у центрі башточка з червоним дахом
    s = 7
    cv2.rectangle(img, (bx - s - 1, by - s - 1), (bx + s, by + s), OUT, -1)
    cv2.rectangle(img, (bx - s, by - s), (bx + s - 1, by + s - 1), WH[2], -1)
    for k in range(-s, s, 2):
        for (x, y) in ((bx + k, by - s), (bx + k, by + s - 1), (bx - s, by + k), (bx + s - 1, by + k)): img[y, x] = WH[0]
    cv2.rectangle(img, (bx - s + 2, by - s + 2), (bx + s - 3, by + s - 3), Y[1], -1)
    cv2.rectangle(img, (bx - 3, by - 3), (bx + 2, by + 2), OUT, -1)
    cv2.rectangle(img, (bx - 2, by - 2), (bx + 1, by + 1), RED[2], -1)
    img[by - 2:by, bx - 2:bx] = RED[2]; img[by:by + 2, bx:bx + 2] = RED[0]


def pilot(out_dir):
    import citydraw as CD, citygen as CG, sprites
    from palgame import PAL
    x, y = CG.px(KAL['lat'], KAL['lon']); gx, gy = int(round(x * CD.K)), int(round(y * CD.K))
    bx, by = gx, gy + 4                                  # основа — на південному фасаді будинку
    W, H = 480, 272
    x0, y0 = gx - W // 2, gy - 150
    # сам будинок (Подільська, 39): червоний вальмовий дах, як на фото, а не плаский
    D = CD.data()
    for b_ in D['B']:
        if b_['t'].get('addr:street') == 'Подільська вулиця' and b_['t'].get('addr:housenumber') == '39':
            b_['t'] = dict(b_['t'], building='house'); b_['roof'] = list(RED[1])
    base, _ = CD.render(x0, y0, W, H)
    pal = np.array(PAL + [(0, 0, 0)] * (256 - len(PAL)), np.uint8)
    car = sprites.car_frames()[8]
    def put_car(img):
        cx, cy = W // 2 + 70, 150 + 58
        for yy in range(car.shape[0]):
            for xx in range(car.shape[1]):
                if car[yy, xx]: img[cy - 14 + yy, cx - 14 + xx] = pal[car[yy, xx]]
    res = {}
    a = base.copy(); variant_a(a, bx - x0, by - y0); put_car(a); res['A'] = a
    b = base.copy()
    f = tower_facade()
    blit_rgba(b, f, bx - x0 - 10, by - y0 - f.shape[0] + 3)
    put_car(b); res['B'] = b
    c = base.copy(); put_car(c); res['0'] = c
    for k, v in res.items():
        cv2.imwrite(os.path.join(out_dir, 'kal_%s.png' % k), cv2.resize(v[..., ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    lab = lambda im, t: cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row = []
    for k, t in (('0', 'ZARAZ'), ('A', 'A: zgory + tin'), ('B', 'B: fasad')):
        im = cv2.resize(res[k][..., ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST); lab(im, t); row.append(im)
    cv2.imwrite(os.path.join(out_dir, 'kal_compare.png'), np.vstack(row))
    big = cv2.resize(f[..., :3][..., ::-1], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    big[cv2.resize(f[..., 3], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST) == 0] = (60, 60, 60)
    cv2.imwrite(os.path.join(out_dir, 'kal_sprite_x6.png'), big)


def demo_gif(out_path):
    """Макет поведінки в грі: вежа-позначка злегка «дихає», коли машина поруч — виїжджає табличка з назвою."""
    import citydraw as CD, citygen as CG, sprites
    from palgame import PAL
    from PIL import Image, ImageDraw, ImageFont
    x, y = CG.px(KAL['lat'], KAL['lon']); gx, gy = int(round(x * CD.K)), int(round(y * CD.K))
    bx, by = gx, gy + 4
    W, H = 480, 272
    x0, y0 = gx - W // 2 - 40, gy - 150
    D = CD.data()
    for b_ in D['B']:
        if b_['t'].get('addr:street') == 'Подільська вулиця' and b_['t'].get('addr:housenumber') == '39':
            b_['t'] = dict(b_['t'], building='house'); b_['roof'] = list(RED[1])
    base, cls = CD.render(x0, y0, W, H)
    # центр смуги Подільської під вежею
    col = cls[:, bx - x0 - 60]
    rows = [r for r in range(by - y0, H) if col[r] == 5]
    ry = (rows[0] + rows[min(len(rows) - 1, 14)]) // 2 + 4
    pal = np.array(PAL + [(0, 0, 0)] * (256 - len(PAL)), np.uint8)
    car = sprites.car_frames()[0]
    font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PressStart2P.ttf'), 8)
    name = KAL['name'].upper()
    tw = int(font.getlength(name))
    f = tower_facade()
    frames = []
    shown = 0.0
    N = 90
    for i in range(N):
        img = base.copy()
        cx = int(20 + (W - 40) * i / (N - 1)); cy = ry
        d = abs(cx - (bx - x0))
        shown = min(1.0, shown + 0.12) if d < 150 else max(0.0, shown - 0.12)
        bob = int(round(math.sin(i * 0.35) * 1.0))
        # тінь-«пляма» під позначкою
        cv2.ellipse(img, (bx - x0 + 1, by - y0 + 1), (9, 3), 0, 0, 360, tuple(int(v * 0.6) for v in (120, 120, 120)), -1)
        blit_rgba(img, f, bx - x0 - 10, by - y0 - f.shape[0] + 2 + bob)
        for yy in range(car.shape[0]):
            for xx in range(car.shape[1]):
                if car[yy, xx]: img[cy - 14 + yy, cx - 14 + xx] = pal[car[yy, xx]]
        if shown > 0:
            pw = int((tw + 12) * min(1.0, shown * 1.3))
            px0, py0 = bx - x0 + 14, by - y0 - 60 + bob
            pim = Image.fromarray(img); dr = ImageDraw.Draw(pim)
            dr.rectangle([px0, py0, px0 + pw, py0 + 14], fill=OUT)
            dr.rectangle([px0 + 1, py0 + 1, px0 + pw - 1, py0 + 13], fill=(44, 42, 62))
            dr.rectangle([px0 + 1, py0 + 1, px0 + pw - 1, py0 + 1], fill=(246, 196, 40))
            dr.line([(px0 - 3, py0 + 7), (px0, py0 + 7)], fill=OUT, width=1)
            nch = int(len(name) * max(0.0, min(1.0, (shown - 0.25) / 0.75)))
            dr.text((px0 + 6, py0 + 4), name[:nch], font=font, fill=(246, 230, 170))
            img = np.array(pim)
        frames.append(Image.fromarray(img).resize((W * 2, H * 2), Image.NEAREST))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)


# ======================================================================= шаблони позначок
# Кожна позначка — маленький піксельний «портрет» (вид спереду), низ спрайта = основа на карті.
# Повертають список кадрів RGBA (анімація: вогонь, фонтан).
GRAN = [(84, 86, 92), (118, 120, 126), (150, 152, 158), (186, 188, 192)]
METAL = {'bronze': [(58, 44, 30), (98, 74, 44), (140, 108, 62), (178, 146, 92)],
         'dark': [(40, 42, 48), (64, 68, 76), (92, 96, 104), (124, 128, 136)],
         'white': [(150, 150, 150), (196, 196, 192), (226, 226, 220), (246, 246, 240)]}
STONE = {'gray': GRAN, 'dark': [(44, 46, 52), (66, 70, 78), (92, 96, 104), (120, 124, 132)],
         'white': [(170, 170, 166), (204, 204, 198), (230, 230, 224), (248, 248, 242)],
         'tuff': [(110, 60, 48), (146, 84, 64), (176, 110, 84), (204, 140, 108)]}
WALL = {'gray': [(110, 112, 118), (150, 152, 156), (184, 186, 188)], 'beige': [(150, 128, 96), (196, 174, 136), (222, 204, 170)],
        'yellow': Y[1:], 'pink': [(160, 104, 100), (204, 146, 138), (228, 182, 172)], 'white': [(170, 170, 166), (220, 220, 214), (244, 244, 238)],
        'blue': [(70, 96, 140), (104, 136, 184), (150, 178, 214)], 'red': [(120, 50, 40), (164, 74, 56), (196, 104, 82)]}
GOLD = [(150, 104, 20), (214, 164, 40), (246, 210, 90)]
DOMES = {'gold': GOLD, 'blue': [(36, 66, 130), (60, 100, 180), (110, 150, 220)], 'green': [(30, 90, 64), (50, 130, 90), (96, 176, 130)]}
UA = [(40, 90, 190), (250, 210, 40)]


class Spr:
    def __init__(self, w, h):
        self.w, self.h = w, h; self.a = np.zeros((h, w, 4), np.uint8)

    def px(self, x, y, c):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h: self.a[y, x, :3] = c; self.a[y, x, 3] = 255

    def rect(self, x0, y0, x1, y1, c):                  # [x0, x1) x [y0, y1)
        for y in range(int(y0), int(y1)):
            for x in range(int(x0), int(x1)): self.px(x, y, c)

    def disc(self, cx, cy, r, c):
        for y in range(int(cy - r - 1), int(cy + r + 2)):
            for x in range(int(cx - r - 1), int(cx + r + 2)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r: self.px(x, y, c)

    def poly(self, pts, c):
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(pts) * 8).astype(np.int32)], 1, shift=3)
        self.a[m > 0, :3] = c; self.a[m > 0, 3] = 255

    def done(self):
        m = self.a[..., 3] > 0
        ol = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
        a = self.a.copy(); a[ol, :3] = OUT; a[ol, 3] = 255
        return a


def _pedestal(s, cx, base, w, h, col=GRAN):
    s.rect(cx - w // 2 - 1, base - 2, cx + w // 2 + 2, base + 1, col[1])                # сходинка
    s.rect(cx - w // 2, base - h, cx + w // 2 + 1, base - 1, col[2])
    s.rect(cx + w // 2 - 1, base - h, cx + w // 2 + 1, base - 1, col[1])                # тінь справа
    s.rect(cx - w // 2 - 1, base - h - 2, cx + w // 2 + 2, base - h, col[3])            # карниз


def _figure(s, cx, foot, col, pose='stand'):
    """Фігура ~16 пікс заввишки, ноги на висоті foot."""
    c0, c1, c2, c3 = col
    if pose == 'woman':
        s.poly([(cx - 4, foot), (cx + 5, foot), (cx + 2, foot - 10), (cx - 1, foot - 10)], c2)
        s.rect(cx + 2, foot - 10, cx + 4, foot, c1)
    else:
        s.rect(cx - 2, foot - 6, cx, foot, c1); s.rect(cx + 1, foot - 6, cx + 3, foot, c2)       # ноги
        s.rect(cx - 3, foot - 12, cx + 4, foot - 5, c2); s.rect(cx + 2, foot - 12, cx + 4, foot - 5, c1)   # тулуб
    s.rect(cx - 1, foot - 15, cx + 2, foot - 12, c3); s.px(cx + 1, foot - 13, c1)             # голова
    if pose == 'arm':
        s.rect(cx + 4, foot - 18, cx + 6, foot - 10, c2); s.rect(cx + 3, foot - 20, cx + 7, foot - 18, c3)   # рука з булавою
    elif pose == 'cosmo':
        s.disc(cx + 0.5, foot - 14, 3, (220, 220, 214)); s.rect(cx - 1, foot - 15, cx + 2, foot - 13, (60, 100, 170))
        s.rect(cx - 4, foot - 11, cx - 2, foot - 5, (200, 200, 196))
    else:
        s.rect(cx - 4, foot - 11, cx - 2, foot - 5, c1); s.rect(cx + 4, foot - 11, cx + 6, foot - 5, c1)    # руки


def tpl_statue(pose='stand', metal='bronze', ped=12):
    col = METAL[metal]
    W, H = 24, ped + 26
    s = Spr(W, H); cx, base = 11, H - 2
    _pedestal(s, cx, base, 10 if pose != 'group3' else 16, ped)
    foot = base - ped - 2
    if pose == 'group3':
        for dx, p in ((-6, 'woman'), (0, 'stand'), (6, 'woman')): _figure(s, cx + dx, foot, col, p)
    elif pose == 'angel':
        _figure(s, cx, foot, col, 'woman')
        s.poly([(cx - 2, foot - 13), (cx - 9, foot - 19), (cx - 8, foot - 7)], col[3])
        s.poly([(cx + 3, foot - 13), (cx + 10, foot - 19), (cx + 9, foot - 7)], col[2])
    else:
        _figure(s, cx, foot, col, pose)
    return [s.done()]


def tpl_stele(h=30, col='gray', top='flat'):
    C = STONE[col]
    W, H = 22, h + 12
    s = Spr(W, H); cx, base = 10, H - 2
    s.rect(cx - 9, base - 3, cx + 10, base + 1, GRAN[1]); s.rect(cx - 7, base - 5, cx + 8, base - 3, GRAN[2])
    y0 = base - 5 - h
    if top == 'khachkar':
        s.rect(cx - 5, y0, cx + 6, base - 5, C[2]); s.rect(cx + 4, y0, cx + 6, base - 5, C[1])
        s.rect(cx - 1, y0 + 3, cx + 2, y0 + 15, C[0]); s.rect(cx - 4, y0 + 6, cx + 5, y0 + 9, C[0])
    else:
        s.poly([(cx - 4, base - 5), (cx + 5, base - 5), (cx + 3, y0), (cx - 2, y0)], C[2])
        s.poly([(cx + 2, base - 5), (cx + 5, base - 5), (cx + 3, y0), (cx + 1, y0)], C[1])
        if top == 'star':
            for k in range(10):
                a = -math.pi / 2 + k * math.pi / 5; r = 4 if k % 2 == 0 else 1.8
                s.disc(cx + 0.5 + math.cos(a) * r * 0.7, y0 - 3 + math.sin(a) * r * 0.7, 0.9, GOLD[2])
            s.disc(cx + 0.5, y0 - 3, 1.6, GOLD[1])
        elif top == 'cross':
            s.rect(cx, y0 - 7, cx + 2, y0, C[3]); s.rect(cx - 2, y0 - 5, cx + 4, y0 - 3, C[3])
        elif top == 'bell':
            s.poly([(cx - 3, y0 + 8), (cx + 4, y0 + 8), (cx + 2, y0 + 2), (cx - 1, y0 + 2)], GOLD[1])
    return [s.done()]


def tpl_flame():
    frames = []
    for f in range(3):
        s = Spr(24, 26); cx, base = 11, 24
        for k in range(10):                                         # зірка-основа (видно скраю)
            a = k * math.pi / 5; r = 10 if k % 2 == 0 else 6
            s.disc(cx + 0.5 + math.cos(a) * r, base - 2 + math.sin(a) * r * 0.25, 1.2, GRAN[2])
        s.rect(cx - 7, base - 4, cx + 8, base, GRAN[1]); s.rect(cx - 4, base - 7, cx + 5, base - 4, (60, 60, 64))
        hgt = (10, 13, 11)[f]; sway = (0, 1, -1)[f]
        s.poly([(cx - 4, base - 7), (cx + 5, base - 7), (cx + 2 + sway, base - 7 - hgt), (cx - 1, base - 7 - hgt * 0.6)], (230, 100, 30))
        s.poly([(cx - 2, base - 7), (cx + 3, base - 7), (cx + 1 + sway, base - 7 - hgt * 0.7)], (250, 190, 50))
        s.poly([(cx - 1, base - 7), (cx + 2, base - 7), (cx + sway * 0.5 + 0.5, base - 9 - hgt * 0.3)], (255, 240, 170))
        frames.append(s.done())
    return frames


def tpl_fountain():
    frames = []
    for f in range(2):
        s = Spr(30, 30); cx, base = 14, 28
        s.rect(cx - 13, base - 5, cx + 14, base, GRAN[2]); s.rect(cx - 12, base - 5, cx + 13, base - 3, (70, 120, 180))
        s.rect(cx - 2, base - 12, cx + 3, base - 5, GRAN[1])
        _figure(s, cx, base - 12, METAL['bronze'], 'arm')
        for k, (dx, dy) in enumerate(((-5, 18), (6, 17), (-8, 12), (9, 11), (-10, 7), (11, 6))):
            if (k + f) % 2 == 0: s.px(cx + dx, base - dy + 10, (150, 200, 240))
        for dx in range(-11, 12, 3): s.px(cx + dx + f, base - 5, (200, 230, 250))
        frames.append(s.done())
    return frames


def tpl_heli():
    s = Spr(36, 22); base = 20
    s.rect(15, base - 6, 21, base + 1, GRAN[2])                                 # постамент
    s.poly([(6, base - 14), (24, base - 14), (27, base - 10), (24, base - 6), (8, base - 6), (5, base - 9)], (86, 104, 64))
    s.rect(24, base - 12, 35, base - 10, (76, 92, 56)); s.rect(33, base - 15, 35, base - 9, (76, 92, 56))
    s.rect(7, base - 13, 11, base - 10, (140, 180, 200)); s.rect(13, base - 12, 15, base - 10, (140, 180, 200))
    s.rect(1, base - 17, 32, base - 16, (50, 50, 52)); s.rect(14, base - 16, 16, base - 14, (50, 50, 52))
    s.px(5, base - 11, (250, 210, 40)); s.px(5, base - 12, (40, 90, 190))
    return [s.done()]


def tpl_church(domes=3, dome='gold'):
    D = DOMES[dome]
    W, H = 30, 44
    s = Spr(W, H); cx, base = 14, H - 2
    wl = WALL['white']
    s.rect(cx - 10, base - 16, cx + 11, base + 1, wl[1]); s.rect(cx + 8, base - 16, cx + 11, base + 1, wl[0])
    s.rect(cx - 2, base - 7, cx + 3, base + 1, (70, 52, 40)); s.px(cx - 1, base - 8, (70, 52, 40)); s.px(cx, base - 8, (70, 52, 40)); s.px(cx + 1, base - 8, (70, 52, 40))
    for x in (cx - 7, cx + 5): s.rect(x, base - 12, x + 2, base - 8, WIN)
    def onion(x, ytop, r):
        s.rect(x - r + 1, ytop + r * 2 - 1, x + r, ytop + r * 2 + 4, wl[2])                       # барабан
        s.disc(x + 0.5, ytop + r, r, D[1]); s.disc(x - 0.5, ytop + r - 1, r * 0.45, D[2])
        s.poly([(x - 1, ytop + 1), (x + 2, ytop + 1), (x + 0.5, ytop - 2)], D[1])
        s.rect(x, ytop - 7, x + 1, ytop - 1, GOLD[2]); s.rect(x - 1, ytop - 5, x + 2, ytop - 4, GOLD[2])
    onion(cx, base - 34, 4)
    if domes >= 3: onion(cx - 7, base - 26, 2.6); onion(cx + 8, base - 26, 2.6)
    if domes >= 5: onion(cx - 10, base - 21, 1.8); onion(cx + 11, base - 21, 1.8)
    return [s.done()]


def tpl_gothic():
    s = Spr(24, 52); cx, base = 11, 50
    wl = WALL['red']
    s.rect(cx - 9, base - 18, cx + 10, base + 1, wl[1]); s.rect(cx + 7, base - 18, cx + 10, base + 1, wl[0])
    s.rect(cx - 4, base - 34, cx + 5, base - 17, wl[2]); s.rect(cx + 3, base - 34, cx + 5, base - 17, wl[0])
    s.poly([(cx - 5, base - 34), (cx + 6, base - 34), (cx + 0.5, base - 48)], (90, 96, 110))
    s.rect(cx, base - 51, cx + 1, base - 47, GOLD[2]); s.rect(cx - 1, base - 50, cx + 2, base - 49, GOLD[2])
    s.disc(cx + 0.5, base - 27, 2.2, (200, 160, 60)); s.disc(cx + 0.5, base - 27, 1.2, (60, 90, 170))
    for x in (cx - 7, cx + 5): s.rect(x, base - 14, x + 2, base - 6, WIN)
    s.rect(cx - 1, base - 8, cx + 2, base + 1, (60, 44, 36))
    return [s.done()]


def tpl_classic(wall='beige', cols=4, flag=False, wide=False, masks=False, lyre=False, frame=False):
    C = WALL[wall]
    W = 34 if wide else 28
    s = Spr(W, 40); base = 38; x0, x1 = 2, W - 3
    s.rect(x0, base - 18, x1, base + 1, C[1]); s.rect(x1 - 2, base - 18, x1, base + 1, C[0])
    for x in range(x0 + 2, x1 - 2, 4): s.rect(x, base - 15, x + 2, base - 11, WIN); s.rect(x, base - 8, x + 2, base - 3, WIN)
    pw = 6 + cols * 3; px0 = W // 2 - pw // 2
    s.rect(px0, base - 18, px0 + pw, base + 1, C[2])
    for k in range(cols): s.rect(px0 + 2 + k * 3, base - 16, px0 + 3 + k * 3, base - 1, WH[2])
    s.rect(px0 - 1, base - 19, px0 + pw + 1, base - 17, WH[2]); s.rect(px0 - 1, base - 1, px0 + pw + 1, base + 1, GRAN[3])
    s.poly([(px0 - 1, base - 19), (px0 + pw + 1, base - 19), (W / 2, base - 25)], C[2])
    s.poly([(px0 + 2, base - 20), (px0 + pw - 2, base - 20), (W / 2, base - 23)], WH[2])
    if flag:
        s.rect(W // 2, base - 36, W // 2 + 1, base - 24, GRAN[1])
        s.rect(W // 2 + 1, base - 36, W // 2 + 8, base - 33, UA[0]); s.rect(W // 2 + 1, base - 33, W // 2 + 8, base - 30, UA[1])
    if masks:
        s.disc(W // 2 - 2.5, base - 30, 2.4, (240, 220, 120)); s.disc(W // 2 + 3.5, base - 30, 2.4, (220, 90, 70))
        s.px(W // 2 - 3, base - 31, OUT); s.px(W // 2 + 3, base - 31, OUT)
    if lyre:
        s.rect(W // 2 - 2, base - 33, W // 2 - 1, base - 26, GOLD[1]); s.rect(W // 2 + 1, base - 33, W // 2 + 2, base - 26, GOLD[1])
        s.rect(W // 2 - 2, base - 27, W // 2 + 2, base - 25, GOLD[2])
    if frame:
        s.rect(W // 2 - 4, base - 34, W // 2 + 5, base - 26, GOLD[1]); s.rect(W // 2 - 3, base - 33, W // 2 + 4, base - 27, (90, 150, 200))
        s.rect(W // 2 - 3, base - 30, W // 2 + 4, base - 27, (90, 150, 80))
    return [s.done()]


def tpl_mansion(wall='yellow'):
    C = WALL[wall]
    s = Spr(26, 28); base = 26
    s.rect(2, base - 14, 23, base + 1, C[1]); s.rect(21, base - 14, 23, base + 1, C[0])
    s.poly([(1, base - 14), (24, base - 14), (20, base - 21), (5, base - 21)], RED[1])
    s.rect(11, base - 25, 14, base - 19, RED[0])
    for x in (4, 8, 16, 20): s.rect(x, base - 11, x + 2, base - 6, WIN); s.px(x, base - 12, WH[2]); s.px(x + 1, base - 12, WH[2])
    s.rect(11, base - 7, 14, base + 1, (80, 56, 40)); s.rect(10, base - 9, 15, base - 7, WH[2])
    return [s.done()]


def tpl_hotel():
    s = Spr(22, 60); base = 58
    s.rect(3, base - 50, 18, base + 1, (200, 200, 194)); s.rect(16, base - 50, 18, base + 1, (150, 150, 146))
    for y in range(base - 46, base - 4, 4):
        for x in range(5, 16, 3): s.rect(x, y, x + 2, y + 2, (90, 120, 150))
    s.rect(2, base - 54, 19, base - 50, (190, 40, 40)); s.rect(4, base - 53, 17, base - 51, (250, 230, 200))
    s.rect(8, base - 4, 13, base + 1, (60, 60, 70))
    return [s.done()]


def tpl_cinema():
    s = Spr(30, 30); base = 28
    s.rect(2, base - 16, 27, base + 1, (150, 152, 158)); s.rect(25, base - 16, 27, base + 1, (110, 112, 118))
    s.rect(1, base - 22, 28, base - 16, (180, 40, 40))
    for x in range(3, 27, 3): s.px(x, base - 21, (250, 220, 120)); s.px(x + 1, base - 17, (250, 220, 120))
    s.rect(6, base - 20, 23, base - 18, (250, 240, 220))
    s.rect(5, base - 12, 24, base - 4, (60, 70, 90))
    for x in range(6, 24, 4): s.rect(x, base - 11, x + 2, base - 5, (110, 130, 160))
    return [s.done()]


def tpl_bus():
    s = Spr(34, 30); base = 28
    s.rect(2, base - 12, 31, base + 1, (190, 190, 186)); s.rect(29, base - 12, 31, base + 1, (140, 140, 136))
    s.rect(1, base - 16, 32, base - 12, (40, 90, 170)); s.rect(4, base - 15, 29, base - 13, (230, 236, 244))
    s.rect(3, base - 9, 30, base - 3, (80, 110, 140))
    s.rect(9, base - 28, 25, base - 17, (240, 196, 40)); s.rect(10, base - 27, 24, base - 22, (140, 190, 220))
    s.rect(11, base - 20, 13, base - 18, (250, 250, 240)); s.rect(21, base - 20, 23, base - 18, (250, 250, 240))
    return [s.done()]


LOVE_GLYPH = {'Я': ["01111", "10001", "10001", "01111", "00101", "01001", "10001"],
              'Х': ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
              'М': ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
              '♥': ["01010", "11111", "11111", "11111", "01110", "00100", "00000"]}


def tpl_love():
    s = Spr(36, 22); base = 20
    s.rect(1, base - 2, 35, base + 1, GRAN[2])
    x = 3
    for ch, col in (('Я', (240, 240, 236)), ('♥', (220, 50, 60)), ('Х', (240, 240, 236)), ('М', (240, 240, 236))):
        g = LOVE_GLYPH[ch]
        for gy, row in enumerate(g):
            for gx, v in enumerate(row):
                if v == '1': s.rect(x + gx * 1.5, base - 14 + gy * 1.7, x + gx * 1.5 + 2, base - 14 + gy * 1.7 + 2, col)
        x += 8
    return [s.done()]


def tpl_eiffel():
    s = Spr(20, 44); cx, base = 9, 42
    c = (110, 80, 56)
    s.poly([(cx - 8, base), (cx - 5, base), (cx, base - 30), (cx - 1, base - 30)], c)
    s.poly([(cx + 9, base), (cx + 6, base), (cx + 1, base - 30), (cx + 2, base - 30)], c)
    s.rect(cx - 6, base - 12, cx + 8, base - 10, c); s.rect(cx - 3, base - 24, cx + 5, base - 22, c)
    s.rect(cx, base - 40, cx + 2, base - 28, c); s.px(cx, base - 41, (230, 60, 50))
    return [s.done()]


def tpl_parksign():
    s = Spr(24, 22); base = 20
    s.rect(4, base - 10, 6, base + 1, (110, 76, 46)); s.rect(18, base - 10, 20, base + 1, (110, 76, 46))
    s.rect(1, base - 19, 23, base - 9, (150, 106, 62)); s.rect(2, base - 18, 22, base - 10, (178, 132, 80))
    s.disc(7.5, base - 15, 2.6, (60, 130, 60)); s.rect(7, base - 13, 8, base - 11, (90, 60, 40))
    s.disc(13.5, base - 16, 2.2, (80, 150, 70)); s.rect(13, base - 14, 14, base - 11, (90, 60, 40))
    s.disc(18.5, base - 15, 2.4, (50, 120, 56)); s.rect(18, base - 13, 19, base - 11, (90, 60, 40))
    return [s.done()]


def tpl_stadium():
    s = Spr(34, 22); base = 20
    s.poly([(1, base), (33, base), (30, base - 12), (4, base - 12)], (150, 152, 158))
    s.poly([(4, base - 2), (30, base - 2), (28, base - 10), (6, base - 10)], (70, 140, 60))
    s.rect(16, base - 10, 17, base - 2, (240, 240, 236))
    for x in (2, 31): s.rect(x, base - 20, x + 1, base - 12, (90, 90, 96)); s.rect(x - 1, base - 21, x + 2, base - 19, (250, 240, 180))
    return [s.done()]


TPL = dict(tower=lambda **k: [tower_facade()], statue=tpl_statue, stele=tpl_stele, flame=tpl_flame, fountain=tpl_fountain,
           heli=tpl_heli, church=tpl_church, gothic=tpl_gothic, classic=tpl_classic, mansion=tpl_mansion, hotel=tpl_hotel,
           cinema=tpl_cinema, bus=tpl_bus, love=tpl_love, eiffel=tpl_eiffel, parksign=tpl_parksign, stadium=tpl_stadium)


def sprite(e):
    """Кадри позначки для запису з data/landmarks.json (+ поле dome, якщо підібрано за дахом)."""
    p = dict(e.get('p', {}))
    if e['tpl'] == 'church': p['dome'] = e.get('dome', 'gold')
    return TPL[e['tpl']](**p)


def sheet(out):
    """Аркуш усіх позначок (для перевірки), x3."""
    import json
    L = json.load(open('data/landmarks.json'))
    tiles = []
    for e in L:
        fr = sprite(e)[0]
        t = np.full((72, 44, 3), 70, np.uint8)
        h, w = fr.shape[:2]; y0 = 70 - h; x0 = (44 - w) // 2
        m = fr[..., 3] > 0; t[y0:y0 + h, x0:x0 + w][m] = fr[..., :3][m]
        tiles.append(t)
    while len(tiles) % 12: tiles.append(np.full((72, 44, 3), 70, np.uint8))
    rows = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]
    img = np.vstack(rows)
    cv2.imwrite(out, cv2.resize(img[..., ::-1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


if __name__ == '__main__':
    if sys.argv[1] == 'pilot': pilot(sys.argv[2])
    if sys.argv[1] == 'gif': demo_gif(sys.argv[2])
    if sys.argv[1] == 'sheet': sheet(sys.argv[2])
