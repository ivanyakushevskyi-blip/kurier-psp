#!/usr/bin/env bash
# Повна збірка: OSM -> карта (build/) -> ресурси (src/) -> EBOOT.PBP -> dist/
# Прапорці:  --host  додатково збирає ПК-версію і знімає кадри автотесту в build/frames
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "${PSPDEV:-}" && ! -d "$HOME/pspdev" && -d /home/claude/pspdev ]] && PSPDEV=/home/claude/pspdev
export PSPDEV="${PSPDEV:-$HOME/pspdev}"; export PATH="$PATH:$PSPDEV/bin"

# 1) межі ігрової зони (полігон data/bounds_v2.json)          -> build/bounds.npz
# 2) підкладка: знімки Івана ref/khn + стара детальна (build_v1) -> build/sat_map.npy  (довго; лише якщо немає)
# 3) будинки: OSM (колір даху, висота) + ручні + Microsoft Footprints
# 4) карта міста (citydraw, 0,5 м/пікс)  -> build/map.npy, solid.npy, mapdata.json
# 5) пакування                          -> build/MAP.BIN + src/assets*
[[ -f build/bounds.npz ]] || python3 -I tools/bounds2.py
if [[ ! -f build/sat_map.npy || "${SAT:-}" == "1" ]]; then
  python3 -I tools/satalign2.py
fi
python3 -I tools/manualbld.py
python3 -I tools/buildings.py
python3 -I tools/msbld.py
python3 -I tools/citybuild.py
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
