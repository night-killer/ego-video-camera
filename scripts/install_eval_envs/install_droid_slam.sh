#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
require_toolchain "${CUDA124_ROOT}"
ensure_conda_env droid_slam 3.10
ENV_PREFIX="$(env_path droid_slam)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.6.0+cu124" "0.21.0+cu124" \
  "https://download.pytorch.org/whl/cu124"
"${ENV_PREFIX}/bin/python" -m pip install \
  "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
  "matplotlib==3.9.2" scipy tqdm PyYAML

build_torch_scatter "${ENV_PREFIX}" "${CUDA124_ROOT}"
build_droid_extensions "${ENV_PREFIX}" "${CUDA124_ROOT}" \
  "${REPO_ROOT}/thirdparty/DROID-SLAM" standard

assert_torch_cuda "${ENV_PREFIX}" "12.4"
verify_method "${ENV_PREFIX}" droid_slam
