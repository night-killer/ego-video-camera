#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
require_toolchain "${CUDA118_ROOT}"
ensure_conda_env megasam 3.10
ENV_PREFIX="$(env_path megasam)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.0.1+cu118" "0.15.2+cu118" \
  "https://download.pytorch.org/whl/cu118"
"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/megasam-h100.txt"

activate_cuda_toolchain "${CUDA118_ROOT}" "${ENV_PREFIX}"
log "building xFormers 0.0.22.post7 for H100 instead of using its old sm_86 wheel"
XFORMERS_BUILD_TYPE=Release "${ENV_PREFIX}/bin/python" -m pip install \
  --force-reinstall --no-deps --no-build-isolation --no-binary=xformers \
  "xformers==0.0.22.post7"
"${ENV_PREFIX}/bin/python" - <<'PY'
import xformers
from xformers.ops import memory_efficient_attention  # noqa: F401

print(f"xformers={xformers.__version__}")
PY

build_torch_scatter "${ENV_PREFIX}" "${CUDA118_ROOT}"
build_droid_extensions "${ENV_PREFIX}" "${CUDA118_ROOT}" \
  "${REPO_ROOT}/thirdparty/MegaSaM/base" legacy

assert_torch_cuda "${ENV_PREFIX}" "11.8"
verify_method "${ENV_PREFIX}" megasam
