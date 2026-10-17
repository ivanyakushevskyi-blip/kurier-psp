#!/usr/bin/env bash
# Встановлює все потрібне для збірки: тулчейн PSPDEV + Python-бібліотеки.
# Ідемпотентний: повторний запуск нічого не ламає.
set -euo pipefail
PSPDEV_VERSION="${PSPDEV_VERSION:-v20261001}"
PSPDEV_DIR="${PSPDEV:-$HOME/pspdev}"

if [ ! -x "$PSPDEV_DIR/bin/psp-gcc" ]; then
  echo ">> Встановлюю PSPDEV $PSPDEV_VERSION у $PSPDEV_DIR"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/pspdev.tar.gz" \
    "https://github.com/pspdev/pspdev/releases/download/$PSPDEV_VERSION/pspdev-ubuntu-latest-x86_64.tar.gz"
  mkdir -p "$(dirname "$PSPDEV_DIR")"
  tar -xzf "$tmp/pspdev.tar.gz" -C "$(dirname "$PSPDEV_DIR")"
  rm -rf "$tmp"
fi

python3 -c "import numpy, scipy, PIL" 2>/dev/null || \
  pip install --quiet numpy scipy pillow --break-system-packages 2>/dev/null || \
  pip install --quiet numpy scipy pillow

echo ">> Готово. Додай у PATH:  export PSPDEV=$PSPDEV_DIR; export PATH=\$PATH:\$PSPDEV/bin"
"$PSPDEV_DIR/bin/psp-gcc" --version | head -1
