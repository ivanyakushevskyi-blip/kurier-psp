#!/usr/bin/env bash
set -euo pipefail
cd /home/claude/kurier-psp
export PSPDEV=/root/pspdev PATH=$PATH:/root/pspdev/bin SS=2
date
python3 -I tools/citybuild.py ${NPROC:-2}
date
PARKMARKS=1 python3 -I tools/mappatch.py
python3 -I tools/poifix.py
python3 -I tools/lmbuild.py
python3 -I tools/exits.py | tail -1
date
python3 -I tools/bake.py build src
mkdir -p build/frames
gcc -DHOST -O2 -w -o build/host src/main.c src/gfx.c src/audio.c src/assets_data.c -lm -lz
./build/host build/frames build/MAP.BIN
( cd src && make -s clean >/dev/null 2>&1 || true; make -s )
rm -rf dist && mkdir -p dist/PSP/GAME/KurierKhm
cp src/EBOOT.PBP build/MAP.BIN dist/PSP/GAME/KurierKhm/
( cd dist && zip -qr KurierKhm_PSP.zip PSP )
date
echo REPAINT_DONE
