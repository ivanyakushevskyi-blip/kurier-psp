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


if __name__ == '__main__':
    if sys.argv[1] == 'pilot': pilot(sys.argv[2])
    if sys.argv[1] == 'gif': demo_gif(sys.argv[2])

