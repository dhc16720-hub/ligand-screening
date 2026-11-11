#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "$(uname -s)" == Darwin ]]; then
  for lib_dir in "$PROJECT_DIR"/.venv-topic1/lib/python*/site-packages/sklearn/.dylibs; do
    if [[ -f "$lib_dir/libomp.dylib" ]]; then
      export DYLD_LIBRARY_PATH="$lib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
      break
    fi
  done
fi
exec "$PROJECT_DIR/.venv-topic1/bin/python" "$@"
