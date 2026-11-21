"""21.11.2026: «двері» замовлень — точка біля входу, до якої кур'єр підходить пішки.

Точки замовлень стоять надворі біля свого будинку: двері — найближча стіна, відступ ~1 м до точки.
Поверховість — з висоти цього будинку (для часу передачі).
"""
import numpy as np
import cv2


def origins(D, K):
    """Справжнє місце закладу/адреси (точка всередині будинку) за назвою — у mapdata точки вже перенесені на дорогу."""
    import citygen as CG
    MG = CG.MG
    o = {}
    for nid, t in D['n'].items():
        if nid in D['nodes'] and 'name' in t and (t.get('shop') or t.get('amenity')):
            x, y = CG.px(*D['nodes'][nid]); o.setdefault(('s', MG.clean(t['name'])), []).append((x * K, y * K))
    for w in D['w']:
        if w['closed'] and ('shop' in w['t'] or 'amenity' in w['t']) and 'name' in w['t']:
            P = np.array(w['pts']).mean(0); o.setdefault(('s', MG.clean(w['t']['name'])), []).append((P[0] * K, P[1] * K))
    for b in D['B']:
        t = b['t']
        if 'addr:street' in t and 'addr:housenumber' in t:
            P = np.array(b['q']).mean(0)
            o.setdefault(('a', MG.short_street(t['addr:street']) + ', ' + t['addr:housenumber']), []).append((P[0] * K, P[1] * K))
    return o


def build(pois, kind, org, solid, surf, bl, MPP, W, H):
    """kind: 's' — заклади, 'a' — адреси. Двері — на стіні свого будинку, найближчій до точки зупинки, ~1 м назовні."""
    R = int(round(60 / MPP))
    off = 1.0 / MPP
    G = 512
    grid = {}
    for t in bl:
        q = t[3]
        for gy in range(int(q[:, 1].min()) // G, int(q[:, 1].max()) // G + 1):
            for gx in range(int(q[:, 0].min()) // G, int(q[:, 0].max()) // G + 1):
                grid.setdefault((gx, gy), []).append(t)
    out, dd, miss, gate = [], [], 0, 0
    for p in pois:
        x, y = int(p['x']), int(p['y'])
        c = org.get((kind, p['name']), [])
        c = [q for q in c if (q[0] - x) ** 2 + (q[1] - y) ** 2 < R * R]
        if not c:
            out.append((x, y, 1)); miss += 1; continue
        bx, by = min(c, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        bx, by = int(bx), int(by)
        xa, ya = max(0, min(x, bx) - 120), max(0, min(y, by) - 120)
        xb, yb = min(W, max(x, bx) + 120), min(H, max(y, by) + 120)
        blk = (np.asarray(solid[ya:yb, xa:xb]) > 0).astype(np.uint8)
        n, lab = cv2.connectedComponents(blk, connectivity=8)
        lbx, lby = bx - xa, by - ya
        k = lab[lby, lbx]
        if k == 0:                                               # центр не в стіні (двір у формі «П» тощо) — найближчий будинок
            ys, xs = np.nonzero(blk)
            if len(xs) == 0:
                out.append((x, y, 1)); miss += 1; continue
            j = int(((xs - lbx) ** 2 + (ys - lby) ** 2).argmin()); k = lab[ys[j], xs[j]]
        ys, xs = np.nonzero(lab == k)
        lx, ly = x - xa, y - ya
        j = int(((xs - lx) ** 2 + (ys - ly) ** 2).argmin()); wx, wy = int(xs[j]), int(ys[j])
        vx, vy = lx - wx, ly - wy; L = max(1e-3, (vx * vx + vy * vy) ** 0.5)
        s_ = min(L, off + 1.5)
        fx, fy = int(round(wx + vx / L * s_)), int(round(wy + vy / L * s_))
        # двері мають бути досяжні від зупинки (паркан приватного сектору) — інакше біля паркану, найближче до будинку
        fr = ((blk == 0) & (np.asarray(surf[ya:yb, xa:xb]) != 2)).astype(np.uint8)
        nf, lf = cv2.connectedComponents(fr, connectivity=8)
        cs = lf[ly, lx]
        if cs and lf[min(fy, lf.shape[0] - 1), min(fx, lf.shape[1] - 1)] != cs:
            ys2, xs2 = np.nonzero(lf == cs)
            j = int(((xs2 - wx) ** 2 + (ys2 - wy) ** 2).argmin()); fx, fy = int(xs2[j]), int(ys2[j]); gate += 1
        fx, fy = xa + fx, ya + fy
        fl = 1
        for t in grid.get(((xa + wx) // G, (ya + wy) // G), []):
            if cv2.pointPolygonTest(t[3].astype(np.float32), (float(xa + wx), float(ya + wy)), True) >= -3:
                fl = max(1, int(round((t[2] or 6) / 3.0))); break
        out.append((fx, fy, fl)); dd.append(L * MPP)
    print('  двері (%s): %d, за парканом -> біля нього %d' % (kind, len(out), gate))
    print('  двері (%s): %d, без будинку %d, від зупинки медіана %.1f м, макс %.1f м' % (kind, len(out), miss, float(np.median(dd)) if dd else 0, max(dd) if dd else 0))
    return out
