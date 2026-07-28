#!/usr/bin/env bash

# Shared helpers for the H100 evaluation environments. This file is sourced by
# the individual installers and is not intended to be run directly.

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${INSTALL_DIR}/../.." && pwd)"
EVAL_ENV_ROOT="${EVAL_ENV_ROOT:-/data/aigc/cyb/zxgu/env}"
CUDA118_ROOT="${EVAL_ENV_ROOT}/_eval_cuda118"
CUDA124_ROOT="${EVAL_ENV_ROOT}/_eval_cuda124"
ORIGINAL_PATH="${PATH}"
MAX_JOBS="${MAX_JOBS:-8}"

export CONDA_ENVS_PATH="${EVAL_ENV_ROOT}"

log() {
  printf '[eval-env] %s\n' "$*" >&2
}

die() {
  printf '[eval-env] ERROR: %s\n' "$*" >&2
  exit 1
}

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return
  fi
  command -v conda 2>/dev/null || die "conda is not available in PATH"
}

require_repo() {
  [[ -f "${REPO_ROOT}/configs/ego_pose_benchmark.yaml" ]] || \
    die "run this installer from the ego-video-camera checkout"
  mkdir -p "${EVAL_ENV_ROOT}"
}

require_h100() {
  if [[ "${SKIP_H100_CHECK:-0}" == "1" ]]; then
    log "SKIP_H100_CHECK=1; skipping the GPU model check"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  if command -v rg >/dev/null 2>&1; then
    printf '%s\n' "${gpu_names}" | rg -qi 'H100' || \
      die "an H100 was not found (detected: ${gpu_names//$'\n'/, })"
  elif [[ "${gpu_names}" != *H100* ]]; then
    die "an H100 was not found (detected: ${gpu_names//$'\n'/, })"
  fi
  log "detected GPU: ${gpu_names//$'\n'/, }"
}

env_path() {
  printf '%s/%s\n' "${EVAL_ENV_ROOT}" "$1"
}

ensure_conda_env() {
  local name="$1"
  local python_version="$2"
  local prefix
  prefix="$(env_path "${name}")"
  local conda_bin
  conda_bin="$(find_conda)"

  if [[ -e "${prefix}" && ! -d "${prefix}/conda-meta" ]]; then
    die "${prefix} exists but is not a Conda environment"
  fi
  if [[ ! -x "${prefix}/bin/python" ]]; then
    log "creating ${name} at ${prefix} (Python ${python_version})"
    "${conda_bin}" create --yes --prefix "${prefix}" \
      --override-channels -c conda-forge "python=${python_version}" pip
  else
    log "reusing existing environment ${prefix}"
  fi

  "${prefix}/bin/python" - "${python_version}" <<'PY'
import sys
expected = tuple(map(int, sys.argv[1].split(".")))
actual = sys.version_info[: len(expected)]
if actual != expected:
    raise SystemExit(f"Python mismatch: expected {expected}, found {actual}")
PY
}

install_packaging_tools() {
  local prefix="$1"
  local setuptools_version="${2:-69.5.1}"
  "${prefix}/bin/python" -m pip install --upgrade \
    "pip==24.3.1" "setuptools==${setuptools_version}" "wheel==0.45.1" "ninja==1.11.1.3"
}

install_torch_wheels() {
  local prefix="$1"
  local torch_version="$2"
  local torchvision_version="$3"
  local index_url="$4"
  "${prefix}/bin/python" -m pip install --upgrade --index-url "${index_url}" \
    "torch==${torch_version}" "torchvision==${torchvision_version}"
}

python_site_packages() {
  local env_prefix="$1"
  env -u PYTHONHOME -u PYTHONPATH \
    "${env_prefix}/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_path("purelib"))
PY
}

assert_torch_cuda() {
  local prefix="$1"
  local expected_cuda="$2"
  "${prefix}/bin/python" - "${expected_cuda}" <<'PY'
import sys
import torch
expected = sys.argv[1]
actual = torch.version.cuda
if actual != expected:
    raise SystemExit(f"PyTorch CUDA mismatch: expected {expected}, found {actual}")
print(f"torch={torch.__version__} torch.version.cuda={actual}")
PY
}

require_toolchain() {
  local toolkit_root="$1"
  [[ -x "${toolkit_root}/bin/nvcc" ]] || die \
    "missing ${toolkit_root}/bin/nvcc; run install_shared_resources.sh first"
}

activate_cuda_toolchain() {
  local toolkit_root="$1"
  local env_prefix="$2"
  require_toolchain "${toolkit_root}"

  export CUDA_HOME="${toolkit_root}"
  export CONDA_PREFIX="${env_prefix}"
  export PATH="${toolkit_root}/bin:${env_prefix}/bin:${ORIGINAL_PATH}"
  local site_packages
  site_packages="$(python_site_packages "${env_prefix}")"
  local runtime_library_path="${env_prefix}/lib:${toolkit_root}/lib64:${toolkit_root}/lib:${toolkit_root}/targets/x86_64-linux/lib"
  if [[ -d "${site_packages}/torch/lib" ]]; then
    runtime_library_path="${site_packages}/torch/lib:${runtime_library_path}"
  fi
  export LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LIBRARY_PATH="${toolkit_root}/lib64:${toolkit_root}/lib:${toolkit_root}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export CMAKE_PREFIX_PATH="${toolkit_root}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export TORCH_CUDA_ARCH_LIST="9.0+PTX"
  export FORCE_CUDA="1"
  export MAX_JOBS

  local gcc="${toolkit_root}/bin/x86_64-conda-linux-gnu-gcc"
  local gxx="${toolkit_root}/bin/x86_64-conda-linux-gnu-g++"
  [[ -x "${gcc}" && -x "${gxx}" ]] || die "Conda C/C++ compiler is missing in ${toolkit_root}"
  export CC="${gcc}"
  export CXX="${gxx}"
  export CUDAHOSTCXX="${gxx}"
}

build_torch_scatter() {
  local env_prefix="$1"
  local toolkit_root="$2"
  activate_cuda_toolchain "${toolkit_root}" "${env_prefix}"
  local source_root="${REPO_ROOT}/thirdparty/DROID-SLAM/thirdparty/pytorch_scatter"
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ego-torch-scatter-sm90.XXXXXX")"
  local staged_source="${build_dir}/pytorch_scatter"
  mkdir -p "${staged_source}"
  cp -a "${source_root}/." "${staged_source}/"
  rm -rf -- "${staged_source}/build" "${staged_source}/torch_scatter.egg-info"

  local torch_version
  torch_version="$(env -u PYTHONHOME -u PYTHONPATH \
    "${env_prefix}/bin/python" - <<'PY'
import torch

print(torch.__version__.split("+", 1)[0])
PY
)"
  if [[ "${torch_version}" == 2.0.* ]]; then
    log "patching torch-scatter 2.1.2 sources for PyTorch ${torch_version}"
    local source_file
    while IFS= read -r -d '' source_file; do
      sed -i \
        -e 's/c10::cuda::MaybeSetDevice(/cudaSetDevice(/g' \
        -e 's/std::optional/torch::optional/g' \
        -e 's/std::nullopt/torch::nullopt/g' \
        "${source_file}"
    done < <(find "${staged_source}/csrc" -type f \
      \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
      -print0)
    if grep -R -E -q 'c10::cuda::MaybeSetDevice|std::optional|std::nullopt' \
      "${staged_source}/csrc"; then
      die "failed to make torch-scatter sources compatible with PyTorch ${torch_version}"
    fi
  fi

  log "building torch-scatter for sm_90 in ${env_prefix}"
  FORCE_CUDA=1 "${env_prefix}/bin/python" -m pip install \
    --force-reinstall --no-cache-dir --no-deps --no-build-isolation \
    "${staged_source}"
  "${env_prefix}/bin/python" - <<'PY'
import torch
import torch_scatter

print(f"torch-scatter={torch_scatter.__version__} torch={torch.__version__}")
PY
  rm -r -- "${build_dir}"
}

build_droid_extensions() {
  local env_prefix="$1"
  local toolkit_root="$2"
  local source_root="$3"
  local layout="$4"
  activate_cuda_toolchain "${toolkit_root}" "${env_prefix}"

  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ego-droid-sm90.XXXXXX")"
  mkdir -p "${build_dir}/dist"
  log "building ${layout} DROID/lietorch extensions for sm_90 from ${source_root}"
  (
    cd "${build_dir}"
    EGO_DROID_SOURCE_ROOT="${source_root}" \
    EGO_DROID_LAYOUT="${layout}" \
      "${env_prefix}/bin/python" "${INSTALL_DIR}/build_droid_h100.py" \
      bdist_wheel --dist-dir "${build_dir}/dist"
  )

  local wheels=("${build_dir}/dist/"*.whl)
  [[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || \
    die "expected one DROID extension wheel in ${build_dir}/dist"
  "${env_prefix}/bin/python" -m pip install --force-reinstall --no-deps "${wheels[0]}"
  rm -r -- "${build_dir}"
}

build_pytorch3d() {
  local env_prefix="$1"
  local toolkit_root="$2"
  activate_cuda_toolchain "${toolkit_root}" "${env_prefix}"
  "${env_prefix}/bin/python" -m pip install \
    "fvcore==0.1.5.post20221221" "iopath==0.1.10"
  log "building PyTorch3D v0.7.4 for sm_90 in ${env_prefix}"
  FORCE_CUDA=1 "${env_prefix}/bin/python" -m pip install \
    --force-reinstall --no-deps --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"
}

verify_method() {
  local env_prefix="$1"
  local method="$2"
  local site_packages
  site_packages="$(python_site_packages "${env_prefix}")"
  local runtime_library_path="${env_prefix}/lib"
  if [[ -d "${site_packages}/torch/lib" ]]; then
    runtime_library_path="${site_packages}/torch/lib:${runtime_library_path}"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    runtime_library_path="${runtime_library_path}:${LD_LIBRARY_PATH}"
  fi

  # Conda does not guarantee that `conda run` or an activated parent shell
  # puts this environment's PyTorch libraries before another environment's.
  # A mixed torch Python package/libtorch pair can fail before CUDA is checked.
  env \
    -u PYTHONHOME \
    CONDA_PREFIX="${env_prefix}" \
    PATH="${env_prefix}/bin:${ORIGINAL_PATH}" \
    LD_LIBRARY_PATH="${runtime_library_path}" \
    PYTHONPATH="${REPO_ROOT}/src" \
    "${env_prefix}/bin/python" "${INSTALL_DIR}/verify_env.py" \
    --repo-root "${REPO_ROOT}" --method "${method}" --require-h100
}
