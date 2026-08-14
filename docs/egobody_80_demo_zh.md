# EgoBody 80 段元数据选择器

`scripts/select_egobody_80.py` 根据 Motion-X 公布的
`egobody_description_all.csv` 生成 40 段桌面/头部运动候选和 40 段行走候选。
脚本只读取动作 CSV、PV 位姿文本和文件名索引，不下载、不解码 RGB，也不会修改
EgoBody archive。

## 生成选择清单

默认参数是桌面 40 段、行走 40 段：

```bash
PYTHONPATH=src python scripts/select_egobody_80.py \
  --data-root /data/aigc/cyb/zxgu/data/EgoBody \
  --action-csv /tmp/egobody_description_all.csv \
  --output-root outputs/egobody_80
```

首次运行可从公开 Motion-X 仓库取得动作表（只包含动作区间，不包含 RGB）：

```bash
curl -L --fail --proto '=https' --proto-redir '=https' \
  'https://raw.githubusercontent.com/IDEA-Research/Motion-X/main/mocap-dataset-process/egobody_description_all.csv' \
  -o /tmp/egobody_description_all.csv
sha256sum /tmp/egobody_description_all.csv
```

期望 SHA-256 为
`97da5af056948a086c199046932bf8c3a064f3d034d5517da7d077bce1fb8cdd`。

参数：

- `--data-root`：EgoBody 根目录，默认 `/data/aigc/cyb/zxgu/data/EgoBody`。
- `--action-csv`：Motion-X 动作表，默认 `/tmp/egobody_description_all.csv`。
- `--output-root`：输出目录，默认 `outputs/egobody_80`。
- `--desktop-count` / `--walking-count`：两类主片段数量，默认 `40` / `40`。

输出为 `egobody_80_manifest.json` 和 `egobody_80_manifest.csv`。JSON 保存源 CSV
SHA-256、动作行、候选/备用片段、去重校验和 PV 指标状态；CSV 是便于审阅的扁平视图。
当前本地 PV 代理筛选要求转动幅度和 P95 角速度两个门槛同时满足；若下载后使用更
可信的 head tracking 复核失败，应从 `reserve_clips` 替换，而不是降低门槛。

## 选择性下载

认证恢复后，使用独立入口按 manifest 下载。默认从 30 FPS 源流抽取 8 FPS，80 段共约
8,320 张 ego RGB（桌面 40×20 秒、行走 40×6 秒，而不是下载完整的
`egocentric_color.zip`）；下载器通过官方 ZIP
central directory 和 HTTP Range 只 materialize 所需成员，并可断点续传：

```bash
PYTHONPATH=src python scripts/download_egobody_80.py \
  --manifest outputs/egobody_80/egobody_80_manifest.json \
  --data-root /data/aigc/cyb/zxgu/data/EgoBody_demo80 \
  --netrc-file /path/to/egobody.netrc \
  --accept-egobody-license \
  --sample-fps 8 \
  --workers 8
```

需要同步的 master Kinect exo RGB 时追加 `--with-exo`；这会按精确 source frame ID
查找 `kinect_color.zip`，缺失的 exo 帧会在每个 clip 的 `exo_frame_count` 中体现。
也可以用 `--ego-archive`（以及可选的 `--exo-archive`）指向已经依法取得的本地官方
ZIP 做离线重跑。输出位于 `data-root/clips/<clip_id>/`，包含 `frames/`、`frames.csv`、
PV 文本和 `clip.json`。认证失败只写入不含凭据的 `download_blocked.json`。

当前 80 段 ego 数据已经存在时，不必重新读取 `egocentric_color.zip`。直接只补三联画
右侧需要的 Kinect master 帧：

```bash
./run_egobody_demo80.sh prepare-exo \
  --netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc
```

该命令只请求 `frames.csv` 中 8,320 个精确 source frame ID。默认要求全部 exo 帧
存在；如果官方 archive 本身有缺帧，会以非零状态退出并写
`exo_download_manifest.json`，不会进入 GPU 推理。

## 运行 ActiMind Ego 估计

独立入口 `run_egobody_demo80.sh` 直接消费选择性下载目录，不会调用旧的
Easy/Medium/Hard 选择流程。默认配置为 `configs/egobody_demo80_actimind.yaml`，
输入 `/data/aigc/cyb/zxgu/data/EgoBody_demo80`，输出
`outputs/egobody_demo80_actimind`。画面中的模型名称统一为
`ActiMind Ego Estimation`；DA3 只作为内部推理后端和产物字段名。
这批片段没有独立 gaze/head-tracking 文件，因此右上 GT 和右下估计均使用
PV 相机中心作为 `Head Proxy`，不将它们标成真实 head pose。

先在 CPU 上检查 `download_manifest.json`、每段 `clip.json`、`frames.csv` 和
PV reference；追加 `--decode-images` 会逐张解码 ego/exo 图像：

```bash
./run_egobody_demo80.sh validate --all --require-exo --decode-images
```

检查会按 `frames.csv` 顺序读取图像，不会 glob `frames/`；因此下载重跑留下的旧图片
不会混入推理。完整 80 段运行命令为：

```bash
CUDA_VISIBLE_DEVICES=7 ./run_egobody_demo80.sh \
  run \
  --all \
  --run-da3 \
  --render \
  --evaluate \
  --resume
```

也可以重复传入 `--clip-id` 只运行指定片段：

```bash
CUDA_VISIBLE_DEVICES=7 ./run_egobody_demo80.sh \
  run \
  --clip-id DESK_001 \
  --clip-id WALK_001 \
  --run-da3 \
  --render \
  --evaluate \
  --resume
```

`run --category desktop` 和 `run --category walking` 可按类别筛选；
`run --category all` 覆盖两类。
批量调度时可追加 `--continue-on-error`，让失败片段写入汇总后继续，但命令最终仍以
非零状态报告任何失败。配置与路径覆盖以 `./run_egobody_demo80.sh run --help` 为准。

### 8 卡并行

`run_egobody_demo80_8gpu.sh` 将 80 段固定均分给 8 张卡：每卡 5 段桌面和 5 段行走，
都是 1,040 个采样帧。每卡保持一个进程，日志和 worker summary 写入
`outputs/egobody_demo80_actimind/launcher_logs/`，全部结束后原子合并为根目录的
`run_summary.json`。直接在 tmux 中运行：

```bash
./run_egobody_demo80_8gpu.sh
```

默认使用 GPU `0,1,2,3,4,5,6,7` 和上述默认输出目录。可用
`DEMO80_GPUS` 和 `DEMO80_OUTPUT_ROOT` 覆盖：

```bash
DEMO80_GPUS=0,1,2,3,4,5,6,7 \
DEMO80_OUTPUT_ROOT=/path/to/output \
./run_egobody_demo80_8gpu.sh
```

脚本启动前会验证 8 张卡和全部 exo 输入，并用 `flock` 防止同一输出目录被两个
tmux 任务重复写入。中断或部分失败后重跑同一命令即可；launcher 始终传入
`--resume`。

## 帧和运动指标

Motion-X 的 `frame_interval_end` 是包含端点的。选择器保留
`frame_end_inclusive`，同时输出半开区间 `frame_start_inclusive`、
`frame_end_exclusive`；例如 20 秒窗口是 600 帧，`end_exclusive = start + 600`。
桌面候选只使用 `body_idx_0=1`（佩戴 HoloLens 的人），行走候选按 recording 合并
动作区间，避免两个演员的标签造成重复 RGB 时间段。行走片段固定为 6 秒：严格的
20 秒非重叠行走窗口不足 40 段，因此不能诚实地把 40 段都标成 20 秒。桌面排序优先使用已有 PV 文本
计算的相机旋转代理（转动幅度、平均/P95 角速度）；这些不是经过认证的头部姿态，
JSON 中会明确写出 `computed_pv_camera_rotation_proxy_*` 或 `pending_*` 状态。动作
文字本身不能证明发生了头部运动，下载并取得可信 head pose 后仍应复核并替换失败片段。

## 认证阻塞

当前 EgoBody 彩色 archive 需要 ETH Zürich 站点认证。未提供有效凭据时，服务器会
返回 HTTP `401 Unauthorized`（过期凭据也可能返回 `403`）。这不是选择器错误：本脚本
可以在只有公开动作 CSV 和本地 PV 文本时生成清单，但不能凭空解析缺失的彩色 ZIP。
请在获得授权后按仓库下载器的说明提供权限为 `600` 的 netrc/认证文件，再执行实际
归档下载和 head-pose 验收；不要把凭据路径或内容写入 manifest、日志或文档。
