"""Мультиполігони OSM (relation type=multipolygon): водосховище з островом, парки, будинки з внутрішніми дворами.
rings(osm_dir) -> [(role, [node ids], rel_tags, rel_id)], role 'outer'|'inner'."""
import glob, os
import xml.etree.ElementTree as ET

_cache = {}


def rings(osm_dir):
    if osm_dir in _cache: return _cache[osm_dir]
    ways, rels = {}, {}
    for f in sorted(glob.glob(os.path.join(osm_dir, 't*.osm'))):
        r = ET.parse(f).getroot()
        for w in r.iter('way'):
            ways[w.get('id')] = [nd.get('ref') for nd in w.iter('nd')]
        for rl in r.iter('relation'):
            t = {x.get('k'): x.get('v') for x in rl.iter('tag')}
            if t.get('type') != 'multipolygon': continue
            mem = [(m.get('ref'), m.get('role') or 'outer') for m in rl.iter('member') if m.get('type') == 'way']
            if rl.get('id') not in rels or len(mem) > len(rels[rl.get('id')][0]): rels[rl.get('id')] = (mem, t)
    out = []
    for rid, (mem, t) in rels.items():
        for role in ('outer', 'inner'):
            segs = [list(ways[w]) for w, r_ in mem if r_ == role and w in ways and len(ways[w]) >= 2]
            while segs:
                ring = segs.pop(0)
                grown = True
                while ring[0] != ring[-1] and grown:
                    grown = False
                    for i, s in enumerate(segs):
                        if s[0] == ring[-1]: ring += s[1:]
                        elif s[-1] == ring[-1]: ring += s[::-1][1:]
                        elif s[-1] == ring[0]: ring = s + ring[1:]
                        elif s[0] == ring[0]: ring = s[::-1] + ring[1:]
                        else: continue
                        segs.pop(i); grown = True; break
                if len(ring) >= 4:
                    if ring[0] != ring[-1]: ring = ring + [ring[0]]          # незамкнене (частина за межами даних) — замикаємо
                    out.append((role, ring, t, rid))
    _cache[osm_dir] = out
    return out
