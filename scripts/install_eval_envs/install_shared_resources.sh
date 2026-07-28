#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100

install_toolchain() {
  local prefix="$1"
  local cuda_version="$2"
  local gcc_version="$3"
  local channel_label="$4"
  local conda_bin
  conda_bin="$(find_conda)"

  if [[ -e "${prefix}" && ! -d "${prefix}/conda-meta" ]]; then
    die "${prefix} exists but is not a Conda environment"
  fi
  if [[ ! -d "${prefix}/conda-meta" ]]; then
    log "installing shared CUDA ${cuda_version} build toolchain at ${prefix}"
    "${conda_bin}" create --yes --prefix "${prefix}" \
      --override-channels -c "nvidia/label/${channel_label}" -c conda-forge \
      "cuda-toolkit=${cuda_version}" "gxx_linux-64=${gcc_version}" "cmake<4" ninja
  elif [[ ! -x "${prefix}/bin/nvcc" ]]; then
    log "completing shared CUDA ${cuda_version} build toolchain at ${prefix}"
    "${conda_bin}" install --yes --prefix "${prefix}" \
      --override-channels -c "nvidia/label/${channel_label}" -c conda-forge \
      "cuda-toolkit=${cuda_version}" "gxx_linux-64=${gcc_version}" "cmake<4" ninja
  else
    log "reusing CUDA toolchain ${prefix}"
  fi
  "${prefix}/bin/nvcc" --version
}

install_dinov2() {
  local destination="${REPO_ROOT}/thirdparty/dinov2"
  # This is the last small, standard-backbone-only line after the positional
  # embedding interpolation fix. Newer hubconf.py revisions eagerly import
  # Cell-DINO/XRay-DINO modules that this benchmark never uses.
  local revision="e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
  if [[ ! -e "${destination}" ]]; then
    log "cloning DINOv2 at ${revision}"
    git clone --filter=blob:none --no-checkout \
      https://github.com/facebookresearch/dinov2.git "${destination}"
    git -C "${destination}" fetch origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
  fi
  [[ -d "${destination}/.git" ]] || die "${destination} is not a Git checkout"
  if [[ -n "$(git -C "${destination}" status --short)" ]]; then
    die "${destination} has local changes; refusing to change its revision"
  fi
  if [[ ! -f "${destination}/hubconf.py" || \
        "$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)" != "${revision}" ]]; then
    git -C "${destination}" fetch origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
  fi
  [[ -f "${destination}/hubconf.py" ]] || die "DINOv2 hubconf.py is missing"
  log "DINOv2 ready at ${destination}"
}

install_toolchain "${CUDA118_ROOT}" "11.8" "11" "cuda-11.8.0"
install_toolchain "${CUDA124_ROOT}" "12.4" "12" "cuda-12.4.0"
install_dinov2

log "shared resources are ready"
