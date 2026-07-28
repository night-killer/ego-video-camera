#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo

ensure_conda_env orb_slam3_build 3.10
BUILD_ENV="$(env_path orb_slam3_build)"
CONDA_BIN="$(find_conda)"
"${CONDA_BIN}" install --yes --prefix "${BUILD_ENV}" --override-channels -c conda-forge \
  "cmake<4" ninja patchelf pkg-config \
  "gcc_linux-64=11" "gxx_linux-64=11" \
  "glew=2.2.0" "boost-cpp=1.85.*" openssl \
  "libopencv=4.11.0=headless_py310*" "eigen=3.4.*" \
  "libgl-devel=1.7.*" "libopengl-devel=1.7.*" "libegl-devel=1.7.*"

ORB_CC="${BUILD_ENV}/bin/x86_64-conda-linux-gnu-cc"
ORB_CXX="${BUILD_ENV}/bin/x86_64-conda-linux-gnu-c++"
[[ -x "${ORB_CC}" && -x "${ORB_CXX}" ]] || \
  die "Conda C/C++ compiler is missing in ${BUILD_ENV}"
CMAKE_COMPILER_ARGS=(
  "-DCMAKE_C_COMPILER=${ORB_CC}"
  "-DCMAKE_CXX_COMPILER=${ORB_CXX}"
)

# Keep an activated model environment out of the C++ search path and expose
# only the dedicated Conda build/runtime prefix.
run_orb_build() {
  env \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    -u CMAKE_PREFIX_PATH \
    -u CMAKE_LIBRARY_PATH \
    -u CMAKE_INCLUDE_PATH \
    -u CPATH \
    -u CPLUS_INCLUDE_PATH \
    -u LIBRARY_PATH \
    -u LD_LIBRARY_PATH \
    -u PKG_CONFIG_PATH \
    CONDA_PREFIX="${BUILD_ENV}" \
    CPATH="${BUILD_ENV}/include" \
    CPLUS_INCLUDE_PATH="${BUILD_ENV}/include" \
    LIBRARY_PATH="${BUILD_ENV}/lib" \
    LD_LIBRARY_PATH="${BUILD_ENV}/lib" \
    PKG_CONFIG_PATH="${BUILD_ENV}/lib/pkgconfig:${BUILD_ENV}/share/pkgconfig" \
    PATH="${BUILD_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$@"
}

PANGOLIN_SOURCE="${EVAL_ENV_ROOT}/_eval_sources/Pangolin-v0.6"
PANGOLIN_PREFIX="${EVAL_ENV_ROOT}/_eval_pangolin06"
PANGOLIN_BUILD="${EVAL_ENV_ROOT}/_eval_build/Pangolin-v0.6"
PANGOLIN_REVISION="dd801d244db3a8e27b7fe8020cd751404aa818fd"
if [[ ! -e "${PANGOLIN_SOURCE}" ]]; then
  mkdir -p "$(dirname "${PANGOLIN_SOURCE}")"
  git clone --filter=blob:none --no-checkout \
    https://github.com/stevenlovegrove/Pangolin.git "${PANGOLIN_SOURCE}"
  git -C "${PANGOLIN_SOURCE}" fetch origin "${PANGOLIN_REVISION}"
  git -C "${PANGOLIN_SOURCE}" checkout --detach "${PANGOLIN_REVISION}"
fi
[[ -d "${PANGOLIN_SOURCE}/.git" ]] || die "invalid Pangolin checkout: ${PANGOLIN_SOURCE}"
if [[ -n "$(git -C "${PANGOLIN_SOURCE}" status --short)" ]]; then
  die "Pangolin checkout has local changes: ${PANGOLIN_SOURCE}"
fi
if [[ "$(git -C "${PANGOLIN_SOURCE}" rev-parse HEAD 2>/dev/null || true)" != "${PANGOLIN_REVISION}" ]]; then
  git -C "${PANGOLIN_SOURCE}" fetch origin "${PANGOLIN_REVISION}"
  git -C "${PANGOLIN_SOURCE}" checkout --detach "${PANGOLIN_REVISION}"
fi

log "building Pangolin v0.6 into ${PANGOLIN_PREFIX}"
run_orb_build "${BUILD_ENV}/bin/cmake" --fresh \
  -S "${PANGOLIN_SOURCE}" -B "${PANGOLIN_BUILD}" \
  "${CMAKE_COMPILER_ARGS[@]}" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PANGOLIN_PREFIX}" \
  -DCMAKE_PREFIX_PATH="${BUILD_ENV}" \
  -DCMAKE_BUILD_RPATH="${BUILD_ENV}/lib" \
  -DCMAKE_INSTALL_RPATH="${BUILD_ENV}/lib" \
  -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PYPANGOLIN_MODULE=OFF \
  -DBUILD_PANGOLIN_VIDEO=OFF -DDISPLAY_X11=OFF -DDISPLAY_WAYLAND=OFF \
  -DBUILD_PANGOLIN_TOON=OFF -DBUILD_PANGOLIN_LIBPNG=OFF \
  -DBUILD_PANGOLIN_LIBJPEG=OFF -DBUILD_PANGOLIN_LIBTIFF=OFF \
  -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_ZSTD=OFF \
  -DBUILD_PANGOLIN_LZ4=OFF
run_orb_build "${BUILD_ENV}/bin/cmake" --build "${PANGOLIN_BUILD}" \
  --parallel "${MAX_JOBS}"
run_orb_build "${BUILD_ENV}/bin/cmake" --install "${PANGOLIN_BUILD}"

ORB_ROOT="${REPO_ROOT}/thirdparty/ORB_SLAM3"
PREFIX_PATH="${PANGOLIN_PREFIX};${BUILD_ENV}"
ORB_BUILD_ROOT="${EVAL_ENV_ROOT}/_eval_build/ORB-SLAM3"
log "building ORB-SLAM3 DBoW2"
run_orb_build "${BUILD_ENV}/bin/cmake" --fresh -S "${ORB_ROOT}/Thirdparty/DBoW2" \
  -B "${ORB_BUILD_ROOT}/DBoW2" -G Ninja \
  "${CMAKE_COMPILER_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" -DCMAKE_BUILD_RPATH="${BUILD_ENV}/lib"
run_orb_build "${BUILD_ENV}/bin/cmake" --build "${ORB_BUILD_ROOT}/DBoW2" \
  --parallel "${MAX_JOBS}"

log "building ORB-SLAM3 g2o"
run_orb_build "${BUILD_ENV}/bin/cmake" --fresh -S "${ORB_ROOT}/Thirdparty/g2o" \
  -B "${ORB_BUILD_ROOT}/g2o" -G Ninja \
  "${CMAKE_COMPILER_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" -DCMAKE_BUILD_RPATH="${BUILD_ENV}/lib"
run_orb_build "${BUILD_ENV}/bin/cmake" --build "${ORB_BUILD_ROOT}/g2o" \
  --parallel "${MAX_JOBS}"

log "building the ORB-SLAM3 library"
run_orb_build "${BUILD_ENV}/bin/cmake" --fresh \
  -S "${ORB_ROOT}" -B "${ORB_BUILD_ROOT}/ORB_SLAM3" -G Ninja \
  "${CMAKE_COMPILER_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" -DCMAKE_BUILD_RPATH="${BUILD_ENV}/lib"
run_orb_build "${BUILD_ENV}/bin/cmake" --build "${ORB_BUILD_ROOT}/ORB_SLAM3" \
  --target ORB_SLAM3 --parallel "${MAX_JOBS}"

log "building the headless benchmark runner"
RUNNER_BUILD="${REPO_ROOT}/build/orb_slam3_benchmark"
run_orb_build "${BUILD_ENV}/bin/cmake" --fresh \
  -S "${REPO_ROOT}/tools/orb_slam3_benchmark" \
  -B "${RUNNER_BUILD}" -G Ninja -DORB_SLAM3_ROOT="${ORB_ROOT}" \
  "${CMAKE_COMPILER_ARGS[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" -DCMAKE_BUILD_RPATH="${BUILD_ENV}/lib"
run_orb_build "${BUILD_ENV}/bin/cmake" --build "${RUNNER_BUILD}" \
  --parallel "${MAX_JOBS}"

RUNTIME_RPATH="${BUILD_ENV}/lib:${PANGOLIN_PREFIX}/lib:${PANGOLIN_PREFIX}/lib64:${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib"
RPATH_TARGETS=(
  "${PANGOLIN_PREFIX}/lib/libpangolin.so"
  "${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so"
  "${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.so"
  "${ORB_ROOT}/lib/libORB_SLAM3.so"
  "${RUNNER_BUILD}/orb_slam3_monocular"
)
for target in "${RPATH_TARGETS[@]}"; do
  [[ -f "${target}" ]] || die "missing ORB-SLAM3 build artifact: ${target}"
  "${BUILD_ENV}/bin/patchelf" --force-rpath --set-rpath "${RUNTIME_RPATH}" "${target}"
done
install -Dm755 "${RUNNER_BUILD}/orb_slam3_monocular" \
  "${REPO_ROOT}/build/benchmark/orb_slam3_monocular"

if command -v rg >/dev/null 2>&1; then
  unresolved="$(run_orb_build ldd "${REPO_ROOT}/build/benchmark/orb_slam3_monocular" | rg 'not found' || true)"
else
  unresolved="$(run_orb_build ldd "${REPO_ROOT}/build/benchmark/orb_slam3_monocular" | grep 'not found' || true)"
fi
if [[ -n "${unresolved}" ]]; then
  run_orb_build ldd "${REPO_ROOT}/build/benchmark/orb_slam3_monocular" >&2
  die "ORB-SLAM3 runner still has unresolved shared libraries"
fi
log "ORB-SLAM3 runner ready: ${REPO_ROOT}/build/benchmark/orb_slam3_monocular"
