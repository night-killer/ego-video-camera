# Ego RGB 相机位姿统一 Benchmark

该 benchmark 在固定的 10 Hz RGB 输入上统一运行 11 个正式方法和 2 个消融方法。数据清单包含 9 个数据集、70 个 clips；完整矩阵包含 1610 个正式运行和 36 个消融运行，共 1646 项。

本实现不会创建 Conda 环境、安装依赖或下载模型。`preflight` 只检查本地仓库、checkpoint、输入、工具和已有 Conda 环境；`run` 根据 `CONDA_ENVS_PATH` 解析绝对 prefix，并直接使用 `<prefix>/bin/python` 及该环境的运行库，避免 GPU 节点上的同名环境或 Conda 二次解析覆盖 benchmark 环境。

H100 环境划分、逐模型安装脚本、`sm_90` 编译方式和完整执行顺序见
[`scripts/install_eval_envs/README_zh.md`](../scripts/install_eval_envs/README_zh.md)。

## 方法与数据

正式方法：DA3-Streaming、VGGT-Omega、LingBot-Map、VGGT-SLAM、ViPE、ReViV-512、EgoM2P、DROID-SLAM、ORB-SLAM3、MegaSaM、HaWoR camera module。

消融方法：DA3 direct、ReViV-256。两个消融只运行配置中固定的 9 段子集。

EgoEgo 的官方 checkpoint 依赖预计算的 512 维光流特征，但官方没有发布可复现这些特征的完整提取器权重。因此 EgoEgo adapter、环境脚本和已下载资源只作为历史实验材料保留，不进入正式矩阵、评测或报告。

数据集：EgoBody、Princeton365、TUM RGB-D、Bonn RGB-D Dynamic、OpenLORIS Office、DROID wrist、HoloAssist、RH20T wrist、Stera10M。

方法、权重、环境、seed、参数和固定子集均定义在 `configs/ego_pose_benchmark.yaml`。worker manifest 只包含 RGB 路径、时间戳，以及方法被允许使用时的相机内参；不包含 reference、GT 路径、reference grade 或 `clip.json`。

## 命令

在仓库根目录执行：

```bash
python scripts/run_pose_benchmark.py inventory
python scripts/run_pose_benchmark.py preflight
python scripts/run_pose_benchmark.py plan
python scripts/run_pose_benchmark.py run --resume
python scripts/run_pose_benchmark.py evaluate --resume
python scripts/run_pose_benchmark.py report
```

环境和权重尚未准备好时，可只生成完整命令矩阵：

```bash
python scripts/run_pose_benchmark.py run --dry-run
```

准备完成后可执行完整流程：

```bash
python scripts/run_pose_benchmark.py all --resume
```

八卡机器使用 sequence 互斥分片脚本。脚本先执行完整 preflight，按 clip
时长和计划运行数均衡分配 GPU 0-7，分片结束后补跑失败项并统一评测、出报告：

```bash
bash scripts/run_pose_benchmark_8gpu.sh
```

分片计划写入 `outputs/ego_pose_benchmark/multi_gpu_plan.json`，各卡调度日志写入
`outputs/ego_pose_benchmark/launcher_logs/`。重复执行会通过 `--resume` 跳过已有成功结果。
只检查分片而不启动 worker 时使用：

```bash
BENCHMARK_PLAN_ONLY=1 bash scripts/run_pose_benchmark_8gpu.sh
```

所有执行型子命令都支持过滤：

```bash
python scripts/run_pose_benchmark.py run \
  --methods 'da3_*' --methods droid_slam \
  --datasets egobody \
  --sequences 'recording_20210907_*' \
  --resume
```

过滤参数接受 shell 风格 glob，可重复，也可用逗号分隔。`--resume` 跳过已有成功结果并重试未完成项；已有失败结果只有在 `--resume` 或 `--force` 下才会重新运行；`--force` 也会重跑成功项。

ORB-SLAM3 的无 viewer runner 使用 H100 环境计划中的独立脚本构建：

```bash
bash scripts/install_eval_envs/install_orb_slam3.sh
```

DA3-Streaming、VGGT-SLAM 和 MegaSaM 会强制共用本地 DINOv2 torchhub 仓库，不允许运行时联网。准备环境时需将 DINOv2 checkout 放在 `thirdparty/dinov2`；`preflight` 会同时检查目录和 `hubconf.py`。SALAD 与 Depth Anything 的完整权重继续使用配置中各自的本地 checkpoint。

## 输出

默认输出根目录是 `outputs/ego_pose_benchmark`：

```text
outputs/ego_pose_benchmark/
  inventory.json
  plan.json
  preflight.json
  execution_summary.json
  evaluation_summary.json
  cache/frames/
  runs/<method>/<dataset>/<sequence>/seed_<seed>/
    worker_manifest.json
    run.json
    stdout.log
    stderr.log
    worker_events.jsonl
    telemetry.json
    prediction.npz
    prediction.json
    evaluation.json
  report/
    metrics_report_zh.md
    metrics_report.json
    leaderboard.csv
    sequence_metrics.csv
    run_metrics.csv
    benchmark_metrics.png
    benchmark_rotation_rmse.png
    benchmark_rpe_1s.png
```

单卡 scheduler 严格串行运行。每项记录 wall time、模型就绪时间、首个预测时间、CPU RAM 峰值、GPU VRAM 峰值和私有临时目录磁盘峰值；模型就绪和首个预测时间都从 worker 子进程启动时刻计算，因此包含 Conda 启动开销。状态区分 success、timeout、OOM、method failure、invalid output、input error 和 evaluation failure。

## 指标与聚合

评测协议包括 raw metric、initial-SE3、5 秒/10 秒 Prefix-Sim3 和 Oracle-Sim3。Metric-scale 方法以 initial-SE3 为 primary protocol；非 metric-scale 方法以 5 秒 Prefix-Sim3 为 primary protocol。

指标包含 ATE、旋转误差、0.1/1/5/10 秒 RPE、最终漂移、尺度误差和尺度漂移、输出与可评分 coverage、初始化/丢失/恢复/reset、静止抖动、速度/加速度/jerk 误差，以及置信度排序指标。

报告先在每个 method/sequence 内聚合多个 seed，并报告 seed 标准差；随后在 A、B1、B2 三个 reference grade 内跨 sequence 聚合。95% 置信区间采用 sequence bootstrap，固定 seed 为 20260724，默认 10000 次。失败运行不以零误差计入均值；没有真实结果时 Markdown、CSV、JSON 和 PNG 均显示 `pending`。

三种 reference grade 表示参考轨迹来源和可信度，并非片段难度：A 使用外部或公开数据集提供的高质量独立真值；B1 使用 HoloLens tracking、ARKit 等设备自身的 VIO/跟踪轨迹；B2 使用机器人正运动学与手眼标定得到的相机轨迹。B1、B2 都不称为严格 GT，三个等级分别统计和排名。

报告生成三项主要误差图：`benchmark_metrics.png` 为 ATE RMSE，`benchmark_rotation_rmse.png` 为旋转 RMSE，`benchmark_rpe_1s.png` 为 1 秒相对平移 RPE RMSE。每张图按 A、B1、B2 分栏，误差棒为 sequence bootstrap 95% 置信区间，三个误差指标均为越低越好。
