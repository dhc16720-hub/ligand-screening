#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# Reuse the OpenMP runtime bundled with this project's scikit-learn wheel.
# XGBoost and LightGBM otherwise look for a system Homebrew installation.
if [[ "$(uname -s)" == Darwin ]]; then
  for lib_dir in "$PROJECT_DIR"/.venv-allis/lib/python*/site-packages/sklearn/.dylibs; do
    if [[ -f "$lib_dir/libomp.dylib" ]]; then
      export DYLD_LIBRARY_PATH="$lib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
      break
    fi
  done
fi
echo 'Amadeus: http://127.0.0.1:8787 (Ctrl+C to stop)'
exec "$PROJECT_DIR/.venv-allis/bin/python" -u -m amadeus.server "$@"
