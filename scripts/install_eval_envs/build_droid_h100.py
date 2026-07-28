#!/usr/bin/env python3
"""Build DROID and lietorch extensions without editing third-party setup.py files."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def required_directory(value: str, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"Missing {label}: {path}")
    return path


source_root = required_directory(os.environ.get("EGO_DROID_SOURCE_ROOT", ""), "DROID source root")
layout = os.environ.get("EGO_DROID_LAYOUT", "")
if layout not in {"standard", "legacy"}:
    raise SystemExit("EGO_DROID_LAYOUT must be 'standard' or 'legacy'")

thirdparty = source_root / "thirdparty"
lietorch_root = required_directory(str(thirdparty / "lietorch"), "lietorch source root")
lietorch_package = required_directory(str(lietorch_root / "lietorch"), "lietorch package")
eigen_root = lietorch_root / "eigen" if layout == "standard" else thirdparty / "eigen"
required_directory(str(eigen_root), "Eigen headers")

gencode = [
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
]


def cuda_extension(name: str, sources: list[Path], *, includes: list[Path], optimize: str):
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing extension sources for {name}: {missing}")
    return CUDAExtension(
        name,
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in includes],
        extra_compile_args={"cxx": [optimize], "nvcc": [optimize, *gencode]},
    )


extensions = [
    cuda_extension(
        "droid_backends",
        [
            source_root / "src" / "droid.cpp",
            source_root / "src" / "droid_kernels.cu",
            source_root / "src" / "correlation_kernels.cu",
            source_root / "src" / "altcorr_kernel.cu",
        ],
        includes=[eigen_root],
        optimize="-O3",
    ),
    cuda_extension(
        "lietorch_backends",
        [
            lietorch_package / "src" / "lietorch.cpp",
            lietorch_package / "src" / "lietorch_gpu.cu",
            lietorch_package / "src" / "lietorch_cpu.cpp",
        ],
        includes=[lietorch_package / "include", eigen_root],
        optimize="-O2",
    ),
]

if layout == "standard":
    extras = lietorch_package / "extras"
    extensions.append(
        cuda_extension(
            "lietorch_extras",
            [
                extras / "altcorr_kernel.cu",
                extras / "corr_index_kernel.cu",
                extras / "se3_builder.cu",
                extras / "se3_inplace_builder.cu",
                extras / "se3_solver.cu",
                extras / "extras.cpp",
            ],
            includes=[],
            optimize="-O2",
        )
    )

setup(
    name=f"ego-eval-droid-extensions-{layout}",
    version="0.1.0",
    packages=find_packages(where=str(lietorch_root), include=["lietorch", "lietorch.*"]),
    package_dir={"": str(lietorch_root)},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"egg_info": {"egg_base": str(Path.cwd())}},
)
