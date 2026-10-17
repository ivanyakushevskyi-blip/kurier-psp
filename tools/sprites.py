"""Courier-on-scooter sprite (32 rotations) + map icons, supersampled and quantised to the palette."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pal import PAL, R

SS = 4
SZ = 40          # sprite canvas (1x)
K = 1.22         # sprite scale
NROT = 32


def col(i):
    return PAL[i] + (255,)


def draw_scooter(lean=0.0):
    """scooter pointing east (+x), centred, drawn at SS scale"""
    S = SZ * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2

    def P(x, y): return (c + x * SS * K, c + y * SS * K)

    def rect(x0, y0, x1, y1, ci, r=0):
        d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=r * SS * K, fill=col(ci))

    def ell(x, y, rx, ry, ci):
        d.ellipse([P(x - rx, y - ry), P(x + rx, y + ry)], fill=col(ci))

    # wheels
    rect(-12, -1.6, -6.5, 1.6, R['tire'][0], 1.2)
    rect(6.5, -1.5, 12, 1.5, R['tire'][0], 1.2)
    # body
    d.polygon([P(-11, -3.2), P(4, -3.6), P(9, -2.0), P(10.5, 0), P(9, 2.0), P(4, 3.6), P(-11, 3.2)], fill=col(R['red'][1]))
    d.polygon([P(-10, -2.4), P(4, -2.8), P(8.5, -1.2), P(9.5, 0), P(8.5, 1.2), P(4, 2.8), P(-10, 2.4)], fill=col(R['red'][2]))
    d.polygon([P(-6, -1.6), P(4, -2.0), P(7.5, -0.5), P(7.5, 0.3), P(-6, 0.6)], fill=col(R['red'][3]))
    ell(9.6, 0, 1.0, 1.4, R['yellow'][3])            # headlight
    # handlebar
    rect(5.5, -6.2, 7.0, 6.2, R['steel'][1], 0.6)
    rect(5.2, -7.0, 7.4, -5.2, R['tire'][1], 0.5)
    rect(5.2, 5.2, 7.4, 7.0, R['tire'][1], 0.5)
    ell(7.2, -6.6, 0.9, 0.8, R['steel'][2]); ell(7.2, 6.6, 0.9, 0.8, R['steel'][2])   # mirrors
    # delivery box
    rect(-14.5, -4.8, -6.0, 4.8, R['box'][0], 1.0)
    rect(-14.0, -4.3, -6.5, 4.3, R['box'][2], 0.8)
    rect(-13.2, -3.5, -7.3, 3.5, R['box'][3], 0.6)
    rect(-10.6, -3.5, -9.6, 3.5, R['box'][1], 0)
    # rider: legs/seat
    rect(-5.6, -3.6, 0.5, 3.6, R['tire'][1], 1.5)
    # torso / shoulders (jacket)
    ell(-1.0, 0, 3.4, 5.6, R['blue'][0])
    ell(-0.6, 0, 2.8, 5.0, R['blue'][2])
    ell(-0.2, -0.8 + lean, 1.8, 3.2, R['blue'][3])
    # arms
    for sg in (-1, 1):
        d.line([P(0, sg * 4.6), P(4.6, sg * 5.6)], fill=col(R['blue'][1]), width=int(2.4 * SS * K))
        ell(5.6, sg * 5.6, 1.2, 1.2, R['skin'][2])
    # helmet
    ell(0.6, 0, 3.4, 3.4, R['tire'][0])
    ell(0.6, 0, 3.0, 3.0, R['white'][2])
    d.chord([P(-2.4, -3.0), P(3.6, 3.0)], -70, 70, fill=col(R['blue'][1]))
    ell(-0.6, -1.4, 1.0, 0.8, R['white'][2])
    d.line([P(-2.0, 0), P(0.6, 0)], fill=col(R['red'][2]), width=int(1.0 * SS * K))
    return im


def quantise(im, allowed):
    a = np.asarray(im, np.float32)
    al = a[..., 3] / 255.0
    pre = a[..., :3] * al[..., None]
    S = im.size[0] // SS
    pre = pre.reshape(S, SS, S, SS, 3).mean(axis=(1, 3))
    al = al.reshape(S, SS, S, SS).mean(axis=(1, 3))
    rgb = pre / np.maximum(al[..., None], 1e-4)
    pal = np.array([PAL[i] for i in allowed], np.float32)
    dd = ((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    out = np.where(al > 0.45, np.array(allowed, np.uint8)[dd.argmin(-1)], 0).astype(np.uint8)
    m = out > 0
    ol = ndi.binary_dilation(m) & ~m
    out[ol] = R['out'][0]
    return out


def scooter_frames():
    base = draw_scooter()
    allowed = sum((R[k] for k in ('red', 'box', 'blue', 'tire', 'white', 'skin', 'steel', 'yellow', 'ink')), [])
    frames = []
    for i in range(NROT):
        ang = i * 360.0 / NROT
        rot = base.rotate(-ang, resample=Image.BICUBIC, center=(SZ * SS / 2, SZ * SS / 2))
        frames.append(quantise(rot, allowed))
    return frames


def icon(kind):
    """14x14 map icon: 'shop' (bag) or 'home' (house)"""
    S = 16 * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    def P(x, y): return (x * SS, y * SS)
    if kind == 'shop':
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['green'][1]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['green'][2]))
        d.rounded_rectangle([P(4.5, 6.5), P(11.5, 12.5)], radius=SS, fill=col(R['white'][2]))
        d.arc([P(6, 3.5), P(10, 9)], 180, 360, fill=col(R['white'][2]), width=int(1.3 * SS))
    else:
        d.ellipse([P(1, 1), P(15, 15)], fill=col(R['yellow'][1]))
        d.ellipse([P(2, 2), P(14, 14)], fill=col(R['yellow'][2]))
        d.polygon([P(8, 3.5), P(12.8, 7.8), P(3.2, 7.8)], fill=col(R['red'][1]))
        d.rectangle([P(4.6, 7.6), P(11.4, 12.6)], fill=col(R['white'][2]))
        d.rectangle([P(7.0, 9.4), P(9.0, 12.6)], fill=col(R['box'][1]))
    allowed = sum((R[k] for k in ('green', 'yellow', 'white', 'red', 'box', 'ink')), [])
    return quantise(im, allowed)


if __name__ == '__main__':
    fr = scooter_frames()
    sheet = np.zeros((SZ, SZ * 8), np.uint8)
    for k in range(8):
        sheet[:, k * SZ:(k + 1) * SZ] = fr[k * 4]
    pal = np.array(PAL + [(255, 0, 255)] * (256 - len(PAL)), np.uint8)
    pal[0] = (255, 0, 255)
    Image.fromarray(pal[sheet]).resize((SZ * 8 * 6, SZ * 6), Image.NEAREST).save(sys.argv[1])
    ic = np.hstack([icon('shop'), icon('home')])
    Image.fromarray(pal[ic]).resize((ic.shape[1] * 8, ic.shape[0] * 8), Image.NEAREST).save(sys.argv[2])
