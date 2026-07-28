# H100 Eval 环境安装计划

这套脚本只面向当前仓库的 `configs/ego_pose_benchmark.yaml`。所有 Conda
环境和共享 CUDA 编译工具链默认放在：

```text
/data/aigc/cyb/zxgu/env
```

脚本不会删除已有环境；如果同名目录不是 Conda 环境，或 Python 版本不符，
会直接停止。CUDA 扩展只为 H100 的 `sm_90` 编译，同时保留 `compute_90`
PTX。除 `prepare_worldsearcher.sh` 外，各脚本不会修改 `worldsearcher`。

## 1. 环境划分

| Benchmark 方法 | Conda 环境 | 安装策略 |
|---|---|---|
| DA3-Streaming、DA3 direct | `worldsearcher` | 直接复用现有 Torch 2.8/cu128 和源码注入 |
| VGGT-Omega | `worldsearcher` | 直接复用，源码由 worker 注入 |
| LingBot-Map | `worldsearcher` | 直接复用，源码由 worker 注入 |
| ViPE | `worldsearcher` | 复用 Torch/CUDA；只补 `rerun-sdk`、`pyarrow`、`python-pycg` 并按 `sm_90` 编译 ViPE 扩展 |
| ORB-SLAM3 worker | `worldsearcher` | Python/OpenCV 复用；C++ runner 单独构建 |
| VGGT-SLAM | `vggt_slam` | Python 3.11、Torch 2.3.1/cu121；固定 SALAD/VGGT fork revision |
| ReViV | `reviv` | Python 3.12、Torch 2.6/cu124；跳过 benchmark 不使用的 GeoCalib 本地路径 |
| EgoM2P | `egom2p` | Python 3.12、Torch 2.6/cu124 |
| DROID-SLAM | `droid_slam` | Python 3.10、Torch 2.6/cu124；DROID/lietorch/torch-scatter 重编 `sm_90` |
| MegaSaM | `megasam` | Python 3.10、Torch 2.0.1/cu118；xFormers 和 DROID 栈重编 `sm_90` |
| HaWoR | `hawor` | Python 3.10、Torch 2.0.1/cu118；PyTorch3D 和内置 DROID 栈重编 `sm_90` |
| EgoEgo-adapted | `egoego` | Python 3.10、Torch 2.0.1/cu118；只装 camera worker 依赖，不装 Mujoco/gym |

另外会创建两个只用于编译、不会被 scheduler 调用的共享前缀：

```text
/data/aigc/cyb/zxgu/env/_eval_cuda118
/data/aigc/cyb/zxgu/env/_eval_cuda124
```

这样 MegaSaM、HaWoR、EgoEgo 共用一份 CUDA 11.8 toolkit，DROID/ReViV
共用一份 CUDA 12.4 toolkit，不在每个环境中重复安装 nvcc。

ORB-SLAM3 还会在同一根目录创建 `orb_slam3_build` 编译/运行时环境，以及
`_eval_sources`、`_eval_build`、`_eval_pangolin06` 三个源码/构建前缀；它们
不会被 benchmark scheduler 通过 `conda run` 调用，但 C++ runner 的 RPATH
会固定指向 `orb_slam3_build/lib`。

## 2. GPU 机器前置检查

需要满足：

- `nvidia-smi` 能看到 H100，compute capability 为 9.0；
- 驱动能够运行 CUDA 12.8，因为 `worldsearcher` 是 Torch 2.8/cu128；
- `conda`、`git`、网络可用；
- 至少保留足够空间给 7 个模型环境、2 个 CUDA toolkit 和编译缓存。

先进入仓库并暴露目标环境目录：

```bash
cd /data/aigc/cyb/zxgu/code/ego-video-camera
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/data/aigc/cyb/zxgu/env
```

如果机器上的 GPU 不可见，模型安装脚本和 CUDA smoke test 会停止。
`SKIP_H100_CHECK=1` 只建议用于在无 GPU 节点执行第一步的共享 toolkit/DINOv2
准备，不能跳过后续环境的 CUDA smoke test，也不能代替最终 H100 验证。

## 3. 安装顺序

第一步安装共享 CUDA toolkit，并在 `thirdparty/dinov2` 准备固定 revision
`e1277af2ba9496fbadf7aec6eba56e8d882d1e35` 的离线 torch.hub checkout：

```bash
bash scripts/install_eval_envs/install_shared_resources.sh
```

第二步补齐共享环境。脚本使用 `--no-deps`，并在操作前后断言 Torch 和
NumPy import 版本完全一致，不会让 pip 自动重解共享环境：

```bash
bash scripts/install_eval_envs/prepare_worldsearcher.sh
```

第三步逐个安装独立环境。建议串行执行，避免多个 CUDA 编译同时抢内存和
磁盘；某个脚本失败不会影响其他环境：

```bash
bash scripts/install_eval_envs/install_vggt_slam.sh
bash scripts/install_eval_envs/install_reviv.sh
bash scripts/install_eval_envs/install_egom2p.sh
bash scripts/install_eval_envs/install_droid_slam.sh
bash scripts/install_eval_envs/install_megasam.sh
bash scripts/install_eval_envs/install_hawor.sh
bash scripts/install_eval_envs/install_egoego.sh
```

ORB-SLAM3 的 C++ 依赖也安装到 `/data/aigc/cyb/zxgu/env/orb_slam3_build`，
不需要 sudo 或 apt。脚本固定使用 Conda GCC 11、GLEW 2.2、headless OpenCV
4.11、Eigen 3.4、Boost、OpenSSL 和 GLVND：

```bash
bash scripts/install_eval_envs/install_orb_slam3.sh
```

脚本会关闭 benchmark 不需要的 Pangolin X11/Wayland、Python 和视频输入模块，
并清除已激活模型环境对 CMake 搜索路径的影响。可以直接重跑；fresh configure
不会复用之前失败时缓存的 OpenGL/GLEW 路径。

最后统一做 import、CUDA、`sm_90` 扩展和 ORB runner 检查：

```bash
bash scripts/install_eval_envs/verify_all_envs.sh
```

每个独立安装脚本末尾也会执行对应检查，所以脚本退出码为 0 才表示该环境
准备完成。

## 4. `worldsearcher` 的已知风险

当前环境实际 `import numpy` 为 2.2.5，但 site-packages 同时残留了
`numpy-1.26.4.dist-info` 和 `numpy-2.2.5.dist-info`。因此不要依据
`pip show numpy` 降级或重装 NumPy，也不要对 DA3/VGGT-Omega 执行会自动
解析 `numpy<2` 的 `pip install -e`。

DA3、VGGT-Omega、LingBot-Map 的源码 import 已通过；但 DA3 和
VGGT-Omega 的项目元数据仍声明 `numpy<2`，所以 H100 上必须各跑一次真实
单序列推理。如果其中一个失败，不要直接修共享环境，应先保留日志，再为该
方法拆独立 NumPy 1.26 环境。

## 5. Eval 调用

调度器使用 `conda run -n <conda_env>`，因此每次运行 benchmark 的 shell
都必须设置 `CONDA_ENVS_PATH`。建议让 `worldsearcher` 的 ffmpeg 和 Python
也排在 PATH 前面：

```bash
cd /data/aigc/cyb/zxgu/code/ego-video-camera
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/data/aigc/cyb/zxgu/env
export PATH=/data/aigc/cyb/zxgu/env/worldsearcher/bin:$PATH
PYTHON=/data/aigc/cyb/zxgu/env/worldsearcher/bin/python
```

先检查 70 个 clips、权重、环境、DINOv2 和 ORB executable：

```bash
$PYTHON scripts/verify_eval_checkpoints.py --hash
$PYTHON scripts/run_pose_benchmark.py inventory
$PYTHON scripts/run_pose_benchmark.py preflight
$PYTHON scripts/run_pose_benchmark.py plan
```

在正式 1856-run 矩阵前，至少对每个环境选一个短序列做真实推理。下面是
单方法示例；替换 `--methods` 即可逐个验证：

```bash
$PYTHON scripts/run_pose_benchmark.py \
  --output-root outputs/ego_pose_h100_smoke \
  run \
  --methods da3_direct \
  --sequences tum_rgbd/rgbd_dataset_freiburg1_desk \
  --force
```

确认 smoke test 后，完整流程可以拆开执行并随时续跑：

```bash
$PYTHON scripts/run_pose_benchmark.py run --resume
$PYTHON scripts/run_pose_benchmark.py evaluate --resume
$PYTHON scripts/run_pose_benchmark.py report
```

也可以一次调用：

```bash
$PYTHON scripts/run_pose_benchmark.py all --resume
```

当前 scheduler 是单 GPU 严格串行；`--resume` 会跳过已经成功的 run。建议
正式运行时按 `--methods` 分批，先完成一个方法的 70 个 clips，再进入下一
个方法，便于定位老项目在 H100 上的兼容问题。

## 6. 兼容性边界

ReViV、EgoM2P、DROID 和 ViPE 使用原生支持 Hopper 的现代 CUDA 栈。
MegaSaM、HaWoR、EgoEgo 是从 CUDA 11.x 老环境迁移到 H100 的兼容构建；
脚本解决的是依赖、编译架构和 import 问题，不能替代加载真实 checkpoint 的
端到端 smoke test。尤其应检查 MegaSaM 的 xFormers、HaWoR 的 PyTorch3D
rasterizer，以及三套 DROID backend 第一次实际 CUDA kernel 调用。HaWoR 的
旧版 MMCV/Chumpy 使用固定版本和非隔离构建，不能再单独升级 pip、setuptools
或 YAPF 后覆盖安装。
