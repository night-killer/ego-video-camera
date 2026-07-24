# EgoBody + DA3 Ego/Exo 位姿 Demo

本仓库从 EgoBody 的 egocentric RGB 估计 DA3 相机轨迹，并把 GT 与 DA3 结果分别投影到同一张、未经去畸变的 master Kinect RGB 上。最终视频为 1920×1080 三联画：左侧 Ego RGB，右上仅显示绿色 GT，右下仅显示橙色 DA3。

面向“彩色、非鱼眼、机器人 ego 优先”的下一轮数据集、模型微调和统一评测设计，见 [彩色透视 Ego Video 位姿估计实验报告](docs/ego_pose_rgb_pinhole_finetune_report_zh.md)。该报告同时审计了现有 `core65` 的鱼眼/灰度偏差、当前仅有 DA3 可运行的问题，以及最新 ReViV 的微调方案。

## 已固定的实现条件

- Python：`/data/aigc/cyb/zxgu/env/worldsearcher/bin/python`
- ffmpeg / ffprobe：来自同一 conda 环境，编码为 H.264、`yuv420p`、`faststart`
- DA3 checkpoint：`/data/aigc/cyb/zxgu/ckpt/DA3NESTED-GIANT-LARGE`
- checkpoint 状态：`user_validated_local`
- `model.safetensors` SHA-256：`8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c`
- 同目录的 `*.partial` 只会记录为 ignored，不会被删除或加载
- DA3 官方 submodule commit：`41736238f5bced4debf3f2a12375d2466874866d`
- 初始环境预检使用 CPU load-only；三个正式 clip 与 smoke 的 DA3 推理随后已在 `CUDA_VISIBLE_DEVICES=7` 上完成

初始化源码：

```bash
git submodule update --init --recursive
```

所有入口都会把本仓库的 `thirdparty/Depth-Anything-3/src` 放到导入路径最前面，并核对 commit，避免误用环境中其他 editable 安装。不会修改 DA3 submodule。

## 坐标、同步与 Head Pose 约定

统一记号为 `T_A_B`，即 `X_A = T_A_B @ X_B`。

- PV 文本中的 16 个数按 C-order 解析为 `T_W_E`
- `holo_to_kinect12.json` 的 `trans` 解析为 `T_K_W`
- GT ego pose：`T_K_E = T_K_W @ T_W_E`
- DA3-Streaming 保存的 stitched pose 是 C2W，局部相机基为 OpenCV 的 `+X` 向右、`+Y` 向下、`+Z` 向前。adapter 保留官方 C2W/W2C，并在后处理入口右乘 `diag(1,-1,-1,1)`，转换为 EgoBody PV/HoloLens 的相机基：`+X` 向右、`+Y` 向上、`-Z` 为视线前向；该固定基变换不改变相机中心
- Sim(3) scale 只作用于相机中心；旋转为 `R_W_E = R_align @ R_D_E`
- Ego/Exo 同步只使用精确 frame ID。EgoBody 没有提供 exo timestamp 时，mapping 中写 `null`，并标记 `sync_basis=exact_frame_id`
- gaze CSV 中的完整 4×4 矩阵是 `T_W_Q`，Head frame 的前向为 `-Z`

投影标注只绘制三根从中心向外的语义箭头：红色 `R` 表示头右方，绿色 `UP` 表示头上方，蓝色 `GAZE` 表示视线方向。真实 Head frame 和 EgoBody PV camera-center proxy 均使用 `(+X,+Y,-Z)`。中心点和历史轨迹仍以绿色区分 GT、橙色区分 DA3。

在 calibration prefix 内估计固定外参：

```text
T_E_Q = inv(T_K_E_gt) @ T_K_Q_gt
```

需要至少 20 个合法配对、覆盖率至少 80%，且全部合法配对相对鲁棒外参的 P95 残差不超过 5 cm / 10°。不满足时永久标注 `Head proxy = ego camera center`；本项目不下载或依赖 SMPL-X。

## 分阶段运行

主入口：

```bash
./run_demo.sh --help
```

配置优先级为 CLI > `EGO_*` 环境变量 > YAML。认证文件路径只接受为本次 CLI 参数，不会进入 resolved config、报告或生成命令。

### 1. 环境、模型与 CPU load-only

此命令会读取正式 safetensors、在 CPU 上构造模型，但不会调用 inference：

```bash
./run_demo.sh --inspect-environment --verify-model-load --inspect-data
```

输出 `system_info.json`、`model_inventory.json`、`data_inventory.json` 和 `execution_status.json`。

### 2. 官方数据下载与基础解压

认证文件必须是权限 `600` 的普通文件。下载器只允许 manifest 中的文件和 `https://egobody.ethz.ch/data/dataset/`，使用 `.part`、`curl -C -`、重试、Content-Length、ETag、ZIP central directory、全成员 CRC、SHA-256 和原子改名。每个未完成对象都有不含凭据的 `*.remote.json` identity sidecar；续传前及完成后都会确认官方对象未变化。多进程以每对象文件锁串行写入，并原子合并 `download_manifest.json`，不会互相覆盖条目。

```bash
./run_demo.sh \
  --download \
  --netrc-file <EgoBody-netrc-path> \
  --extract-base
```

默认单连接完整对象下载使用 `curl -C -`。官方服务器对单连接限速时，也可显式启用可恢复的固定 32 MiB Range 分段；已完成分段和合法 HTTP 206 的部分响应会保留。每个活动 `.transfer` 同步保存不含凭据的响应头，跨进程恢复时必须同时匹配 Content-Range、Content-Length、ETag 和官方对象总长度；缺少或不匹配响应头的临时块会被保守丢弃。连接数只控制 worker 并发，中断后可以调整：

```bash
./run_demo.sh \
  --download \
  --download-name egocentric_color.zip \
  --download-connections 16 \
  --netrc-file <EgoBody-netrc-path>
```

archives 保存到 `/data/aigc/cyb/zxgu/data/EgoBody/_archives`。基础阶段完整解压 metadata、calibrations、Kinect 参数和全部 `*_pv.txt`；clip 选择阶段只提取候选稀疏图像、最终 Ego RGB、对应 master Kinect RGB 及最终 gaze CSV。ZIP 路径会规范化到数据模态根目录，并拒绝路径穿越和 symlink。

若认证过期或返回 401/403，会生成 `download_blocked.json`；该文件不包含认证路径或认证内容。

### 3. Easy / Medium / Hard 选择

```bash
./run_demo.sh --extract-base --select-clips --generate-gpu-commands
```

若两个 RGB 全包仍在后台下载，可以先用同一官方源建立稀疏 ZIP64 cache，只取 central directory、全部 PV 文本和候选/最终成员，以提前完成真实 Demo；该辅助链路使用可恢复的固定官方 HTTPS Range 分段，认证路径仍不会写入任何输出。稀疏 ZIP 视图位于本机 `/tmp/ego_video_camera_remote_zip_cache/`（`/data` 文件系统不保留 sparse hole），实际提取结果仍进入配置的数据根目录：

```bash
./run_demo.sh \
  --select-clips --remote-selective \
  --download-connections 8 \
  --netrc-file <EgoBody-netrc-path>
```

稀疏 cache 不会冒充完整 archive，也不会生成完整 archive SHA；完整 archive 下载、SHA 和 ZIP 校验仍由第 2 阶段独立完成。

候选来自 train/val，默认 20 秒、8 FPS。统计平移、平均/P95 角速度、转身幅度、同步率、缺帧率、master Kinect 画内率，并只为轨迹候选池提取稀疏图像以计算纹理、清晰度、帧间运动和非刚性光流残差。优先画内率 ≥80%、缺帧率 ≤5%，最终三个 clip 必须来自不同 recording，Hard 必须保留可跟踪帧。

输出包括：

- `selection/selected_clips.json`：实际 sequence、逐帧 frame ID / timestamp 与全部统计
- `selection/selection_report.md`：选择理由
- `selection/contact_sheet.jpg`
- `gpu_commands.sh`：无占位的 smoke、三个单片段正式命令、一次运行全部 clips、compose 和三级 OOM 子命令

### 4. 先验收真实 GT-only

```bash
./run_demo.sh --validate-gt
```

输出 `gt_only_overlay.mp4`、preview、逐帧 mapping 和 `gt_validation.json`。报告包含 Z 范围、画内率、pixel bounds、实际 calibration / camera / gaze 路径和 ffprobe 结果。应先查看 preview，确认没有 Z 反向、上下/左右翻转、轴交换或不合理的佩戴者头部位置，再在 GPU 上运行 DA3。

### 5. CPU mock 三联画

```bash
./run_demo.sh --mock
```

mock 会生成可播放的 1920×1080 视频并验证：右上没有 DA3 主色、右下没有 GT 主色、两个 exo 面板来自相同底图。

### 6. 真实 DA3、评估和合成

真实 selection 完成后，先查看自动生成脚本的显式子命令；不带 action 时只显示帮助，不会意外依次执行正式任务和 OOM 方案：

```bash
bash outputs/egobody_da3_toy/gpu_commands.sh --help
bash outputs/egobody_da3_toy/gpu_commands.sh smoke
bash outputs/egobody_da3_toy/gpu_commands.sh formal-all
```

`formal-all` 使用正常配置依次运行 Easy、Medium、Hard，并在成功后生成合集。也可使用 `easy`、`medium`、`hard` 分开调度，最后执行 `compose`。

也可以运行单个 recording：

```bash
CUDA_VISIBLE_DEVICES=7 ./run_demo.sh \
  --sequence-id <selected-recording-name> \
  --run-da3 --render-comparison --evaluate --resume
```

DA3 adapter 只接收 Ego RGB 路径，并明确调用 `use_ray_pose=True`；不会传入 GT extrinsics、GT intrinsics 或 pose-conditioned 数据。loop closure 关闭，因此不需要额外 SALAD checkpoint。默认参数为 resolution 504、chunk/overlap 60/30。

低置信或无效姿态不会用 GT 替代，默认不插值，视频会显示 `DA3 prediction unavailable`。输出同时保存 raw W2C、stitched C2W、预测内参、raw/normalized confidence、模型/源码/checkpoint 信息和 pose convention。

对齐与评估包含：

- Sim(3) full/oracle
- Sim(3) calibration-prefix：3 秒 → 5 秒 → clip 前 30%，退化时明确失败
- SE(3) full
- ATE RMSE/median/P95、约 1 秒 RPE、逐帧旋转误差、最终漂移
- 有效、低置信、插值、同步和投影画内比例
- confidence/error Pearson 与 Spearman 相关性
- 真实 Head Pose 模式下的 3D position、2D pixel 和 orientation error

每个 clip 输出 prefix/oracle 视频与 preview；`--compose-all-toys` 生成 Easy → Medium → Hard 合集。所有真实视频都由 ffprobe 复核 codec、尺寸、pixel format、帧数和时长。

## OOM 降级顺序

不自动更换模型，按以下顺序重跑：

1. 8 → 5 FPS，保持 504 和 60/30
2. resolution 504 → 392，chunk/overlap 60/30 → 30/15
3. resolution 336，chunk/overlap 20/10

生成的 `gpu_commands.sh` 已包含这三级实际命令。

官方 DA3-Streaming 基准在 resolution 504、chunk 60 时报告的峰值显存为 12.7 GB（504×154 输入）到 21.2 GB（504×378 输入）。因此本 Demo 的默认配置建议从至少 24 GB 显存开始；实际值仍取决于输入长宽比、CUDA/PyTorch 版本和显卡，当前 CPU 主机未实测。显存不足时按上述顺序使用 `oom-fps`、`oom-392`、`oom-336`，不自动切换 checkpoint。

## 本次真实数据验收结果

严格 20 秒窗口的实际选择（目标 8 FPS）为：

- Easy：`recording_20210921_S11_S10_02` / `2021-09-21-145953`，160/160 帧，画内率 100%，缺帧率 0%，真实 Head Pose
- Medium：`recording_20210929_S15_S11_03` / `2021-09-29-152630`，156/160 帧，画内率 97.5%，缺帧率 2.5%；prefix 只有 18 个合法 head 配对，因此明确使用 camera-center proxy
- Hard：`recording_20210910_S06_S05_01` / `2021-09-10-171420`，160/160 帧，画内率 100%，缺帧率 0%，真实 Head Pose

三段 GT-only 均已用 ffprobe 验证为 H.264、1920×1080、`yuv420p`、20.000 秒。Medium 只编码 156 个真实帧，实际 CFR 为 7.8 FPS，并永久标注 2.5% source sampling gaps；没有伪造或插值帧。GPU smoke 和三个正式 clip 均已完成，执行产物保存在各 recording 的 `da3/` 目录。

DA3 输出的坐标基修正后，直接复用现有 GPU 推理产物重新完成了对齐、评估和渲染，没有重跑模型：

- Medium：oracle 旋转误差 median/P95 为 1.46°/4.40°，prefix 为 7.26°/9.19°
- Hard：oracle 旋转误差 median/P95 为 5.14°/7.25°；prefix 的 median 为 34.62°，表明这段短 prefix 的外推质量明显较差
- Easy：GT 相机中心在整段内仅移动 3.02 cm，低于 10 cm 的可观测性阈值；oracle、prefix 和 SE(3) 均明确标记为 `degenerate`，视频不绘制伪可靠的 DA3 箭头
- 5 秒 smoke：GT/DA3 平移跨度均小于 1.2 cm，也明确标记为 `degenerate`；它只证明真实推理产物可生成和读取，不作为对齐精度样例

所有正式 DA3 命令均由 `gpu_commands.sh` 固定到 7 号卡。坐标基修正属于确定性的 CPU 后处理，原始 DA3 C2W/W2C 仍完整保留。

## 测试

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python -m pytest -q
```

单测覆盖 pose inverse、`T_A_B` 链、Sim(3)/SE(3)、旋转对齐、畸变投影、frame sync、Head 转换、prefix 退化、letterbox、配置优先级、ZIP 安全、HTTP 206 跨进程断点恢复和 DA3 无 GT 泄漏。

## 主要输出

```text
outputs/egobody_da3_toy/
├── system_info.json
├── model_inventory.json
├── data_inventory.json
├── execution_status.json
├── mock/
├── selection/
├── <recording>/
│   ├── gt_only_overlay.mp4
│   ├── da3/da3_poses_raw.{json,npz}
│   ├── alignment_*.json
│   ├── metrics.json
│   ├── comparison_prefix.mp4
│   └── comparison_oracle.mp4
└── comparison_all_toys.mp4
```
