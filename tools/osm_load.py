"""Parse OSM XML tiles into nodes/ways (merged, de-duplicated)."""
import math, glob, sys
import xml.etree.ElementTree as ET

LAT0, LON0 = 49.422, 26.995           # projection origin (approx city centre)
MLAT = 111320.0
MLON = 111320.0 * math.cos(math.radians(LAT0))


def proj(lat, lon):
    """metres: x east, y south"""
    return ((lon - LON0) * MLON, -(lat - LAT0) * MLAT)


def load(files):
    nodes, ways = {}, {}
    for f in files:
        root = ET.parse(f).getroot()
        for n in root.iter('node'):
            nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
        for w in root.iter('way'):
            wid = w.get('id')
            nds = [nd.get('ref') for nd in w.iter('nd')]
            tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
            ways[wid] = (nds, tags)
        for r in root.iter('relation'):
            pass
    out = []
    for wid, (nds, tags) in ways.items():
        pts = [proj(*nodes[i]) for i in nds if i in nodes]
        if len(pts) >= 2:
            out.append((wid, pts, tags, nds[0] == nds[-1]))
    return out


if __name__ == '__main__':
    ws = load(sorted(glob.glob(sys.argv[1] + '/t*.osm')))
    from collections import Counter
    c = Counter()
    for _, _, t, _ in ws:
        for k in ('highway', 'building', 'landuse', 'leisure', 'natural', 'waterway', 'amenity', 'railway'):
            if k in t: c[k + '=' + (t[k] if k != 'building' else '*')] += 1
    for k, v in c.most_common(45): print(v, k)
    names = Counter(t.get('name') for _, _, t, _ in ws if 'highway' in t and t.get('name'))
    print([n for n, _ in names.most_common(40)])
