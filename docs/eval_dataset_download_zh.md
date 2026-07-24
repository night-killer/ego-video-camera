# Ego 6DoF Evaluation 数据下载说明

> 固定 profile：`ego_pose_eval_core65_v2`
>
> 目标数据量：7 个数据源、112 个 clip、3900 秒（65 分钟）、统一 10 FPS。ORE 不在本次 6DoF GT 下载范围内。

## 1. 直接使用

先只查看计划；该命令不联网、不写文件：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download --dry-run
```

下载全部固定子集。ADT/HOT3D 与 EgoBody 都需要先阅读官方许可；EgoBody
还需要注册邮件中提供的、权限为 `600` 的 netrc：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65 \
  --accept-aria-licenses \
  --accept-egobody-license \
  --egobody-netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc
```

下载可反复执行：普通文件使用 `.part` 断点续传，远程 ZIP 只缓存 central
directory 与选中成员的压缩区间。EgoBody 全部 clip 成功后会自动清除稀疏
缓存；中途失败则保留缓存以便续传。加 `--keep-source` 才保留这些缓存。

建议先按数据源分批：

```bash
# 第一批：严格头戴 GT + device reference + 人群压力
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets adt,egobody,incrowd_vi \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65 \
  --accept-aria-licenses \
  --accept-egobody-license \
  --egobody-netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc

# 第二批：灰度头显与长时轨迹
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets monado,lamaria \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65

# 第三批：非严格头戴的 RGB 泛化
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets princeton365 \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65
```

下载后检查 112 条是否齐全：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py verify \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65
```

## 2. 脚本实际下载什么

| 数据源 | 固定量 | 默认最小下载方式 |
|---|---:|---|
| ADT | 24 × 30 s | 官方 Dataset Explorer 的 H.264 RGB preview + `main_groundtruth`；本地裁到 10 FPS |
| EgoBody | 20 × 20 s | 对认证后的 `egocentric_color.zip` 做 Range，只取 4000 张 10 FPS PV RGB；另取所选窗口的 PV pose、裁剪后的 head tracking 和对应标定 |
| Monado | 16 × 30 s | 对 Hugging Face 上的原始 ZIP 做 HTTP Range，只取 cam0 的 10 FPS PNG、GT CSV 和设备标定 |
| Princeton365 | 18 × 30 s | 对 validation TAR 做 HTTP Range，只取 user MP4、内参、相对变换和 GT trajectory；跳过 depth/stereo/IMU |
| HOT3D Aria | 16 × 20 s | 官方 RGB preview + MPS trajectory/calibration + ground-truth + hand data；不下载点云和 object assets |
| InCrowd-VI | 12 × 30 s | 官方小型 `*_WOA.mp4` + `trj_gt_sec_wxyz.txt` + 标定说明 |
| LaMAria | 6 × 180 s | 对 9–30 GB ASL ZIP 做 HTTP Range，只取 cam0 的 10 FPS PNG；另取 pinhole calibration 与 pseudo-dense GT |

默认模式是为这次 demo/evaluation 控制下载量的 `preview` 模式。ADT/HOT3D 的 preview 是官方 1408×1408 H.264 RGB，但不是原始 VRS；`clip.json` 会明确记录这一点。

EgoBody 默认不下载第三视角图像，因为 6DoF evaluation 只需要 PV RGB 与
reference pose。若还要生成现有三联画 demo，可额外执行：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets egobody \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65 \
  --accept-egobody-license \
  --egobody-netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc \
  --egobody-with-exo
```

该选项只取与 4000 张 PV 输入按 frame ID 同步的 master Kinect RGB，不会
下载 352 GB 的完整 `kinect_color.zip`。

如果论文主表需要严格复现 Aria 原始传感器时间戳和编码，应切到：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets adt,hot3d \
  --aria-mode raw \
  --accept-aria-licenses \
  --adt-cdn-file /path/to/ADT_download_urls.json \
  --hot3d-cdn-file /path/to/Hot3DAria_download_urls.json \
  --hot3d-downloader /path/to/hot3d/hot3d/data_downloader/dataset_downloader_base_main.py \
  --data-root /data/aigc/cyb/zxgu/data/ego_pose_eval_core65
```

ADT 官方链接清单有效期为 14 天。raw 模式只请求 ADT 的 VRS 与 main ground truth，以及 HOT3D 的 VRS、MPS trajectory/calibration 和当前清单中命中 hand/mask/metadata 的组；不会请求 semidense point cloud、eye gaze 或 depth。

## 3. 磁盘和传输预期

最终评测输入严格限制为 65 分钟，但上游发布单元不总能精确裁到这 65 分钟：

- ADT/HOT3D preview 模式通常约 5–7 GB 传输；
- EgoBody 只传输 4000 张选中 PV 图像、20 个 PV metadata 成员、20 个
  head-tracking 成员和小型标定；认证 ZIP 不会整包下载。加 exo 时只增加
  同步的 master Kinect 帧；按现有 JPEG 大小，默认结果约 1.5–3 GB，
  exo 预计再增加约 1–2 GB；
- Princeton365 的 user MP4 位于 TAR 内，所选 indoor/outdoor 记录需要先取完整 MP4，预计约 17.5 GB，是默认 profile 的主要传输开销；
- Monado 与 LaMAria 不下载整包，只取所选 PNG 和小型 metadata；
- 默认在成功生成 clip 后删除临时完整 MP4；加 `--keep-source` 才保留。

因此建议为 preview profile 预留约 30–50 GB。`--aria-mode raw` 还需额外预留几十 GB，具体取决于当期官方归档大小。

## 4. 输出结构

```text
ego_pose_eval_core65/
├── evaluation_manifest.json
├── _cache/
│   └── remote_zip/                 # 稀疏 ZIP 索引，不是完整归档
├── adt/
│   ├── _sources/                   # 仅下载/处理中存在
│   └── clips/<sequence>/
│       ├── video.mp4               # 10 FPS
│       ├── reference/main_groundtruth/aria_trajectory.csv
│       └── clip.json
├── egobody/
│   ├── _shared/kinect_master/Color.json # 仅 --egobody-with-exo
│   └── clips/<recording>/
│       ├── frames/*.jpg              # 20 s × 10 FPS
│       ├── frames.csv
│       ├── frame_manifest.csv
│       ├── exo_frames/*.jpg          # 仅 --egobody-with-exo
│       ├── reference/pv_trajectory.csv
│       ├── reference/pv_camera.json
│       ├── reference/head_hand_eye.csv
│       ├── reference/calibration/*.json
│       └── clip.json
├── monado/
│   ├── _shared/<device>/calibration.json
│   └── clips/<sequence>/
│       ├── frames/*.png
│       ├── frames.csv
│       ├── reference/gt_trajectory.csv
│       └── clip.json
└── ...
```

每个 `clip.json` 记录 source sequence、起止时间、分层标签、参考等级、文件大小与 SHA256；`evaluation_manifest.json` 记录每个数据源是完成还是因许可/临时清单缺失而暂停。

## 5. 已知边界

1. **EgoBody 是 device reference，不是独立 mocap GT。** `pv_trajectory.csv`
   来自官方逐帧 `pv2world_transform`，`head_hand_eye.csv` 来自 HoloLens head
   tracking；两者都不能改名为外部独立真值。20 条中 9 条来自官方 val，
   另 11 条用于补足完整 200 帧、原始 HoloLens stream 唯一性以及
   kitchen/foodlab/快速转头覆盖；四个运动层级各 5 条。官方注册凭据当前
   有效期为 7 天；过期后更新同一 netrc 并重跑即可从保留的选中成员区间
   续传。
2. **Monado 与 LaMAria 是灰度输入。** 公平输入时复制为 3 通道，结果不能与普通 RGB 域无说明地混平均。
3. **Princeton365 不是严格头戴。** 它只进入 RGB 几何泛化表。
4. **InCrowd-VI 全部是室内。** 当前 12 条按 high/medium/low crowd density 各 4 条分层，不再使用错误的 indoor/outdoor 网格。
5. **LaMAria 默认不选 moving-platform。** 公开 moving-platform 序列只有 sparse control points，没有可用于逐帧 6DoF 误差的 pseudo-dense GT；当前 6 条均有公开 pseudo-dense trajectory。

固定抽样清单位于 [`configs/ego_pose_eval_core65.yaml`](../configs/ego_pose_eval_core65.yaml)，下载与整理实现位于 [`scripts/download_eval_datasets.py`](../scripts/download_eval_datasets.py)。
