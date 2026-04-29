#!/usr/bin/env bash
# Build and (optionally) upload the DEFault++ package to PyPI.
#
# Usage:
#   scripts/build_pypi.sh                # build sdist + wheel, run twine check
#   scripts/build_pypi.sh --testpypi     # also upload to TestPyPI
#   scripts/build_pypi.sh --pypi         # also upload to the real PyPI
#
# The script never uploads without an explicit flag, so accidental
# runs only refresh the local ``dist/`` directory.
#
# Authentication for ``twine upload`` reads the standard config:
#   - ~/.pypirc, or
#   - environment variables TWINE_USERNAME / TWINE_PASSWORD, or
#   - PyPI API tokens (recommended): set
#         TWINE_USERNAME=__token__
#         TWINE_PASSWORD=<your-token>

set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "${CODE_ROOT}"

UPLOAD_TARGET=""
for arg in "$@"; do
  case "$arg" in
    --testpypi) UPLOAD_TARGET="testpypi" ;;
    --pypi)     UPLOAD_TARGET="pypi" ;;
    *) echo "[build_pypi] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=================================================================="
echo "  DEFault++ PyPI build"
echo "  CODE_ROOT     = ${CODE_ROOT}"
echo "  UPLOAD_TARGET = ${UPLOAD_TARGET:-<none>}"
echo "=================================================================="

# 1. Clean previous artifacts so the dist/ never mixes versions.
echo ""
echo "[1/4] Cleaning dist/ and *.egg-info"
rm -rf dist build src/*.egg-info src/defaultplusplus.egg-info defaultplusplus.egg-info

# 2. Ensure the build / upload tools are present.
echo ""
echo "[2/4] Ensuring build + twine are installed"
python -m pip install --quiet --upgrade build twine

# 3. Build sdist + wheel via PEP 517.
echo ""
echo "[3/4] python -m build"
python -m build

# 4. Validate the artifacts before they ever touch PyPI.
echo ""
echo "[4/4] twine check dist/*"
python -m twine check dist/*

ls -la dist/

if [[ -z "${UPLOAD_TARGET}" ]]; then
  echo ""
  echo "Build complete. Pass --testpypi or --pypi to upload."
  exit 0
fi

if [[ "${UPLOAD_TARGET}" == "testpypi" ]]; then
  echo ""
  echo "Uploading to TestPyPI"
  python -m twine upload --repository testpypi dist/*
  echo ""
  echo "Smoke-test the install with:"
  echo "    pip install --index-url https://test.pypi.org/simple/ \\"
  echo "                --extra-index-url https://pypi.org/simple/ \\"
  echo "                defaultplusplus"
elif [[ "${UPLOAD_TARGET}" == "pypi" ]]; then
  echo ""
  echo "Uploading to PyPI (production)"
  python -m twine upload dist/*
  echo ""
  echo "Released. Verify with:"
  echo "    pip install --upgrade defaultplusplus"
fi
