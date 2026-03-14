#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASETS_PATH="${DATASETS_PATH:-Datasets}"
CLEAN=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_ssc_data.sh [options]

Options:
  --datasets-path PATH   Base datasets path (default: Datasets)
  --clean                Remove Datasets/SSC before preparing
  --check-only           Only print current SSC file layout and exit
  -h, --help             Show this help

Examples:
  bash scripts/prepare_ssc_data.sh
  bash scripts/prepare_ssc_data.sh --clean
  DATASETS_PATH=/data bash scripts/prepare_ssc_data.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets-path)
      DATASETS_PATH="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SSC_ROOT="${DATASETS_PATH%/}/SSC"
mkdir -p "${SSC_ROOT}"

echo "=== SSC root: ${SSC_ROOT}"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  find "${SSC_ROOT}" -maxdepth 3 -print 2>/dev/null | sort || true
  exit 0
fi

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "=== Removing existing ${SSC_ROOT}"
  rm -rf "${SSC_ROOT}"
  mkdir -p "${SSC_ROOT}"
fi

echo "=== Running SSC smoke probe (this triggers download/extract/preprocess if needed)"
./.venv/bin/python scripts/ssc_smoke.py --probe-loader --datasets-path "${DATASETS_PATH}"

echo "=== SSC layout after probe"
find "${SSC_ROOT}" -maxdepth 3 -print 2>/dev/null | sort || true

echo "=== Done"
