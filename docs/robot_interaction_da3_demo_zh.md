# DA3 机器人交互 Ego/Exo Demo

本入口使用 `data/ego_pose_eval_robot_interaction_rgb` 中的 DROID 与 RH20T
机器人腕部 RGB，只把 Ego RGB 送入 DA3-Streaming。输出保持 1920x1080
三联画：左侧 Ego RGB，右上绿色 `Kinematic Reference`，右下橙色
`Active Ego Foundation Model`。画面和轨迹图中不显示 DA3 名称；DA3 仅作为
内部推理后端。

主入口：

```bash
./run_robot_demo.sh --help
```

默认配置是 `configs/robot_interaction_da3_demo.yaml`，默认输出目录是
`outputs/robot_interaction_da3_demo`。配置优先级仍为 CLI > `EGO_*` 环境变量
> YAML。

## 固定的 7 段数据

DROID 按同步覆盖率、腕部相机中心画内率和画面边界余量选择外部相机：

- `AUTOLab+44bb9c36+2023-11-25-10h-11m-14s`：`24400334`
- `IPRL+7790ec0a+2023-04-19-17h-53m-26s`：`28221883`
- `WEIRD+30c3da59+2024-01-08-17h-59m-55s`：`23804457`

`REAL+abf65a9e+2023-04-06-14h-26m-59s` 的最佳候选仍为 0% 画内，
且 5 秒腕部轨迹跨度只有微米量级，因此只生成 `exclusion.json`，不进入
正式 demo。

RH20T 的四段 cfg3 数据统一使用视野余量最大的 `f0172289`。这份 demo 清单与
原有 4 段 RGB 评测清单相互独立：原 task 1/4/6/8 中只有 task 1 具备完整的
`f0172289` 数据，不能作为统一 exo demo。正式选段为：

| sequence | 15 秒窗口起点 | 同步/画内率 | 3 秒 prefix 位移跨度 |
| --- | ---: | ---: | ---: |
| `task_0012_user_0010_scene_0008_cfg_0003` | 10.5 s | 96.00% | 0.115 m |
| `task_0015_user_0010_scene_0005_cfg_0003` | 11.5 s | 95.33% | 0.249 m |
| `task_0016_user_0010_scene_0009_cfg_0003` | 5.7 s | 95.33% | 0.145 m |
| `task_0017_user_0010_scene_0002_cfg_0003` | 10.6 s | 96.00% | 0.202 m |

最终顺序固定为 DROID 三段，再 RH20T 四段。

同步使用最近时间戳且要求 `|delta t| <= 50 ms`。正式门槛为同步覆盖率至少
95%、腕部相机中心画内率至少 70%。

## 数据准备

原 16 段 Ego RGB 评测 manifest 保持独立。`--prepare-exo` 在每段
`clip/exo/` 下生成帧、时间映射、相机、外参和校验 manifest，并在数据根目录
单独生成 `robot_exo_manifest.json`。

只准备已有 DROID RGB 子集的三段 exo：

```bash
./run_robot_demo.sh --dataset droid --prepare-exo
```

RH20T 可以使用调用者提供的官方 cfg3 包。文件会校验大小和 SHA-256，但不会
被删除：

```bash
./run_robot_demo.sh \
  --dataset rh20t \
  --prepare-exo \
  --rh20t-archive /path/to/RH20T_cfg3.tar.gz
```

不传 `--rh20t-archive` 时会从官方 Google Drive 或固定 revision 的镜像断点
下载 27,399,012,782 字节的 cfg3 包。流程自有的完整包在成功提取后删除；
`--keep-source` 会保留它。TAR 采用流式扫描，只落盘四段所需的
`f0172289` 视频、时间戳和标定成员。

也可以在统一评测下载器中一次完成基础 RGB 和 exo：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --plan configs/ego_pose_eval_robot_interaction_rgb.yaml \
  --data-root data/ego_pose_eval_robot_interaction_rgb \
  --datasets droid_wrist,rh20t_wrist \
  --robot-with-exo \
  --rh20t-archive /path/to/RH20T_cfg3.tar.gz
```

## CPU 验收

先渲染绿色运动学参考，确认同步、投影和相机方向：

```bash
./run_robot_demo.sh --dataset all --validate-reference
```

不依赖真实数据或 CUDA 的三联画 mock：

```bash
./run_robot_demo.sh --mock
```

mock 和真实视频都用 ffprobe 检查 H.264、1920x1080、`yuv420p`、FPS 与
帧数。右侧两个面板使用同一张 exo 底图，绿色和橙色主标记不会跨面板出现。

## DA3 与坐标约定

统一记号为 `T_A_B`，即 `X_A = T_A_B @ X_B`。DROID 与 RH20T 运动学
参考都保存为相机到机器人参考系的 C2W。RH20T 的固定 exo 位姿由官方
extrinsics 推导，并通过方向闭环检查；内参按 MP4 实际解码尺寸缩放。

机器人入口直接读取 `da3/da3_poses_raw.npz["c2w"]`。它是 DA3 官方 OpenCV
C2W，相机轴固定为 `+X=right`、`+Y=down`、`-Y=up`、`+Z=gaze`。这里不会
应用 EgoBody 入口的 `diag(1,-1,-1)` 右乘基变换。

运动学 reference 只在推理完成后用于 Sim(3)/SE(3) 对齐、评测和 exo
投影；DA3 调用只接收 Ego RGB 路径、帧 ID 和时间戳。低置信、缺失姿态或
退化对齐均隐藏橙色标记，不插值，也不以 reference 替代预测。

## GPU 命令

真实 DA3 action 不会隐式下载大文件，并会在启动推理前一次性检查全部所选
片段。先准备 exo；以下 action 在 CPU 上运行，RH20T 缺包时会断点下载
27.4 GB cfg3 包：

```bash
bash outputs/robot_interaction_da3_demo/gpu_commands.sh prepare-exo
```

若已有官方 TAR，使用上文的 `./run_robot_demo.sh --dataset rh20t
--prepare-exo --rh20t-archive ...`，避免重新下载。

生成不自动执行的 CUDA 命令：

```bash
./run_robot_demo.sh --generate-gpu-commands
bash outputs/robot_interaction_da3_demo/gpu_commands.sh --help
```

主要 action：

```bash
bash outputs/robot_interaction_da3_demo/gpu_commands.sh smoke
bash outputs/robot_interaction_da3_demo/gpu_commands.sh droid
bash outputs/robot_interaction_da3_demo/gpu_commands.sh rh20t
bash outputs/robot_interaction_da3_demo/gpu_commands.sh formal-all
bash outputs/robot_interaction_da3_demo/gpu_commands.sh compose
```

如预检报告缺失 exo，CLI 会列出所有缺失片段并在任何 DA3 推理启动前退出。
准备完成后重新执行原 action；命令自带 `--resume`，已经完成的结果不会重跑。

默认 10 FPS、resolution 504、chunk/overlap 60/30、confidence threshold
1.5。三级 OOM action 依次为 `oom-fps`、`oom-392`、`oom-336`。

`formal-all` 为每段生成 `reference_only_overlay.mp4`、
`comparison_prefix.mp4`、`comparison_oracle.mp4`、preview、三种对齐轨迹图、
`metrics.json`、`frame_mapping.json` 和原始 DA3 NPZ/JSON，最后把 7 个 prefix
视频合成为 `comparison_all_prefix.mp4`。
