#!/usr/bin/env bash
# Повна збірка: OSM -> карта (build/) -> ресурси (src/) -> EBOOT.PBP -> dist/
# Прапорці:  --host  додатково збирає ПК-версію і знімає кадри автотесту в build/frames
set -euo pipefail
cd "$(dirname "$0")/.."
export PSPDEV="${PSPDEV:-$HOME/pspdev}"; export PATH="$PATH:$PSPDEV/bin"

python3 -I tools/mapgen.py data/osm build
python3 -I tools/bake.py build src

if [[ "${1:-}" == "--host" ]]; then
  mkdir -p build/frames
  gcc -DHOST -O2 -w -o build/host src/main.c src/assets_data.c -lm -lz
  ./build/host build/frames
fi

( cd src && make -s clean >/dev/null 2>&1 || true; make -s )
rm -rf dist && mkdir -p dist/PSP/GAME/KurierKhm
cp src/EBOOT.PBP dist/PSP/GAME/KurierKhm/
( cd dist && zip -qr KurierKhm_PSP.zip PSP )
echo ">> dist/KurierKhm_PSP.zip  (на PSP: PSP/GAME/KurierKhm/EBOOT.PBP)"
