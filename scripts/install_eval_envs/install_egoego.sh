#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
require_toolchain "${CUDA118_ROOT}"
ensure_conda_env egoego 3.10
ENV_PREFIX="$(env_path egoego)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.0.1+cu118" "0.15.2+cu118" \
  "https://download.pytorch.org/whl/cu118"
"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/egoego-h100.txt"

build_pytorch3d "${ENV_PREFIX}" "${CUDA118_ROOT}"
build_torch_scatter "${ENV_PREFIX}" "${CUDA118_ROOT}"
build_droid_extensions "${ENV_PREFIX}" "${CUDA118_ROOT}" \
  "${REPO_ROOT}/thirdparty/DROID-SLAM" standard

assert_torch_cuda "${ENV_PREFIX}" "11.8"
verify_method "${ENV_PREFIX}" egoego

