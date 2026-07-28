#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
require_toolchain "${CUDA118_ROOT}"
ensure_conda_env hawor 3.10
ENV_PREFIX="$(env_path hawor)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.0.1+cu118" "0.15.2+cu118" \
  "https://download.pytorch.org/whl/cu118"

# Legacy setup.py metadata imports Torch. Isolate libtorch before any such pip
# subprocess so an activated parent environment cannot supply incompatible
# Torch shared libraries.
activate_cuda_toolchain "${CUDA118_ROOT}" "${ENV_PREFIX}"

"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/hawor-h100.txt"

# MMCV 1.3.9 imports pkg_resources while generating metadata, which fails in
# pip's isolated build environment. Its runtime dependencies are listed in the
# requirements file so dependency resolution can stay disabled here.
"${ENV_PREFIX}/bin/python" -m pip install --no-deps --no-build-isolation \
  "mmcv==1.3.9"

# Chumpy's legacy setup.py imports pip while generating metadata, so PEP 517
# build isolation cannot build it. Keep the upstream commit fixed and isolated
# from dependency resolution.
"${ENV_PREFIX}/bin/python" -m pip install --no-deps --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"

build_pytorch3d "${ENV_PREFIX}" "${CUDA118_ROOT}"
build_torch_scatter "${ENV_PREFIX}" "${CUDA118_ROOT}"
build_droid_extensions "${ENV_PREFIX}" "${CUDA118_ROOT}" \
  "${REPO_ROOT}/thirdparty/HaWoR/thirdparty/DROID-SLAM" legacy

assert_torch_cuda "${ENV_PREFIX}" "11.8"
verify_method "${ENV_PREFIX}" hawor
