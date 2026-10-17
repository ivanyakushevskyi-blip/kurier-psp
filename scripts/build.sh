#!/usr/bin/env bash
# Повна збірка: OSM -> карта (build/) -> ресурси (src/) -> EBOOT.PBP -> dist/
# Прапорці:  --host  додатково збирає ПК-версію і знімає кадри автотесту в build/frames
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "${PSPDEV:-}" && ! -d "$HOME/pspdev" && -d /home/claude/pspdev ]] && PSPDEV=/home/claude/pspdev
export PSPDEV="${PSPDEV:-$HOME/pspdev}"; export PATH="$PATH:$PSPDEV/bin"

# 1) межі ігрової зони (OSM)            -> build/bounds.npz
# 2) супутник: зшивання + прив'язка до OSM -> build/sat_map.npy  (довго; пропускається, якщо вже є)
# 3) будинки: колір даху, висота        -> build/buildings.json
# 4) карта міста                        -> build/map.npy, solid.npy, mapdata.json
# 5) пакування                          -> build/MAP.BIN + src/assets*
[[ -f build/bounds.npz ]] || python3 -I tools/bounds.py data/osm_v2 build/bounds.npz
if [[ ! -f build/sat_map.npy || "${SAT:-}" == "1" ]]; then
  python3 -I tools/satalign.py && python3 -I tools/satrefine.py
fi
python3 -I tools/buildings.py
python3 -I tools/citygen.py
python3 -I tools/bake.py build src

if [[ "${1:-}" == "--host" ]]; then
  mkdir -p build/frames
  gcc -DHOST -O2 -w -o build/host src/main.c src/assets_data.c -lm -lz
  ./build/host build/frames build/MAP.BIN
fi

( cd src && make -s clean >/dev/null 2>&1 || true; make -s )
rm -rf dist && mkdir -p dist/PSP/GAME/KurierKhm
cp src/EBOOT.PBP build/MAP.BIN dist/PSP/GAME/KurierKhm/
( cd dist && zip -qr KurierKhm_PSP.zip PSP )
echo ">> dist/KurierKhm_PSP.zip  (на PSP: PSP/GAME/KurierKhm/EBOOT.PBP + MAP.BIN)"
