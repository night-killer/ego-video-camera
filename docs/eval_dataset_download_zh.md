# Ego 6DoF Evaluation 数据下载说明

本项目现在有两个主评测 profile，以及一个兼容旧实验的诊断 profile：

| Profile | 用途 | 已落盘规模 | Reference |
|---|---|---:|---|
| `ego_pose_eval_native_rgb_v1` | 原生彩色、透视相机主榜 pilot | 12 clips / 180 s / 1800 frames / 907,006,401 bytes | A：外部或数据集官方 GT |
| `ego_pose_eval_robot_interaction_rgb_v2` | 机器人腕部与真人交互压力测试 | 16 clips / 200 s / 2000 frames / 1,242,321,622 bytes | B1/B2：设备跟踪或机器人运动学 |
| `ego_pose_eval_core65_v2` | 旧版广覆盖诊断集 | 112 clips / 3900 s | 混合 reference 与相机设置，不进统一主榜 |

机器可读的完整状态在
[`configs/ego_pose_eval_resource_status.yaml`](../configs/ego_pose_eval_resource_status.yaml)。
以下命令均使用仓库内已经存在的 `worldsearcher` 环境，不会安装依赖。

## 1. 两个主评测 Profile

### 1.1 原生 RGB / pinhole / A 级 pilot

固定数据是 TUM RGB-D、Bonn RGB-D Dynamic 和 OpenLORIS Office 各 4 段，
每段 15 秒、10 FPS。OpenLORIS 只选 D435i 的
`d400_color_optical_frame`，明确排除 T265 fisheye 流。

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py plan \
  --plan configs/ego_pose_eval_native_rgb.yaml \
  --data-root data/ego_pose_eval_native_rgb

/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --plan configs/ego_pose_eval_native_rgb.yaml \
  --data-root data/ego_pose_eval_native_rgb

/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py verify \
  --plan configs/ego_pose_eval_native_rgb.yaml \
  --data-root data/ego_pose_eval_native_rgb
```

当前严格验证结果是三个数据源均 `4/4`，总计 `12 clips / 180 s /
1800 RGB frames`。完整上游归档均未保留。

### 1.2 Robot / interaction RGB / B 级 pilot

固定数据为 DROID wrist、HoloAssist、RH20T cfg3 wrist 和 Stera-10M 各 4 段：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py plan \
  --plan configs/ego_pose_eval_robot_interaction_rgb.yaml \
  --data-root data/ego_pose_eval_robot_interaction_rgb

/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --plan configs/ego_pose_eval_robot_interaction_rgb.yaml \
  --data-root data/ego_pose_eval_robot_interaction_rgb \
  --ffmpeg /data/aigc/cyb/zxgu/env/worldsearcher/bin/ffmpeg

/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py verify \
  --plan configs/ego_pose_eval_robot_interaction_rgb.yaml \
  --data-root data/ego_pose_eval_robot_interaction_rgb
```

已有 RH20T 完整归档时可避免重新下载：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --plan configs/ego_pose_eval_robot_interaction_rgb.yaml \
  --data-root data/ego_pose_eval_robot_interaction_rgb \
  --datasets rh20t_wrist \
  --rh20t-archive /path/to/RH20T_cfg3.tar.gz \
  --ffmpeg /data/aigc/cyb/zxgu/env/worldsearcher/bin/ffmpeg
```

`--rh20t-archive` 指向的文件归调用者所有：脚本会检查固定的
27,399,012,782 bytes 和 SHA-256，但不会删除它。脚本自己下载到 `_cache`
的归档会在 4 段全部完成后删除；加 `--keep-source` 才保留。默认下载可在
Google Drive 限流时使用固定 revision 的字节一致 Hugging Face 镜像，整包
promotion 前仍必须通过 SHA-256。

当前严格验证结果为：

| 数据集 | Clip / 秒 / 帧 | Reference 与采样语义 |
|---|---:|---|
| DROID wrist | 4 / 20 / 200 | B2；按 H5 `estimated_capture` 时间采样，再按 H5 frame index 解码 MP4；动态 `camera_to_robot_base` |
| HoloAssist | 4 / 60 / 600 | B1；按 `Pose_sync` video time 与 row index；动态 `camera_to_hololens_world` |
| RH20T cfg3 wrist | 4 / 60 / 600 | B2；按 `timestamps.npy` 的真实毫秒时间选 MP4 frame index；由 TCP 与手眼标定导出 `camera_to_aligned_robot_base` |
| Stera-10M | 4 / 60 / 600 | B1；1280x720 原生 RGB；按 MP4 frame index 对齐同 index ARKit pose，并导出 `camera_optical_to_arkit_world` |

RH20T 的发布 MP4 容器标称 25 FPS，但真实采集时间约为 8--9 Hz。脚本不能按
MP4 PTS 取帧，否则会把运动加速约三倍。统一到 10 Hz 时使用最近原生帧，
`clip.json` 会逐段记录唯一源帧数、重复输出数和最大重采样误差；质量门拒绝
误差超过 250 ms 的窗口。

Stera 固定 revision 为 `548a1f26741647126e4a6347b29b46759e43ebb5`，仓库
实际包含 575 个完整 session。脚本只下载 4 个固定 session 的 RGB、HDF5、
hierarchy 和 calibration，并逐文件检查 size/SHA-256。MP4 是固定 15 FPS，
且帧数与 HDF5 pose 数量一一对应；绝对 ARKit 时间可能包含采集 pause，所以
必须按 MP4 frame index 取 pose。官方 pose 是 `camera_link -> ARKit world`，
RGB reference 使用
`R_world_optical = R_world_link @ R_optical_to_link`。窗口内 tracking 必须全为
`normal`，相邻选中 pose gap 不得超过 250 ms。重建子集需要本机已经通过
`hf auth login` 登录获批账号；token 不写入 manifest。

DROID 和 RH20T 是运动学参考，HoloAssist 与 Stera 是设备跟踪参考，四者都
不能进入外部 mocap A 榜。

## 2. 主 Profile 输出与清理

```text
ego_pose_eval_{native_rgb,robot_interaction_rgb}/
├── evaluation_manifest.json
└── <dataset>/clips/<sequence>/
    ├── frames/*.png
    ├── frames.csv
    ├── frame_manifest.csv
    ├── reference/
    │   ├── camera.json
    │   └── <dataset-specific trajectory/calibration>
    └── clip.json
```

`verify` 不只数文件，还会重算每张图和 reference 文件的 SHA-256，检查 RGB
模式、分辨率、pinhole/fisheye 标志、frame manifest 与固定帧数。下载完成后
不保留完整 TAR/TGZ/MP4/H5 episode；HoloAssist 只留下可恢复的远程 TAR 小型
索引和官方 split 文件，Stera 的完整 MP4/HDF5 在 4 段生成后删除。

## 3. Legacy `core65` 使用

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
  --data-root data/ego_pose_eval_core65 \
  --accept-aria-licenses \
  --accept-egobody-license \
  --egobody-netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc
```

下载可反复执行：普通文件使用 `.part` 断点续传，远程 ZIP 缓存 central
directory 与选中成员的压缩区间。部分 ZIP 的成员布局会让区间缓存很大；
中途失败时保留它是为了续传，全部数据通过 `verify` 后应删除 `_cache`。
当前已验证副本只保留在 `data/ego_pose_eval_core65`，没有保留该下载缓存；
仓库外曾用于下载的重复中间目录也已在独立 verify 后删除。

建议先按数据源分批：

```bash
# 第一批：严格头戴 GT + device reference + 人群压力
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets adt,egobody,incrowd_vi \
  --data-root data/ego_pose_eval_core65 \
  --accept-aria-licenses \
  --accept-egobody-license \
  --egobody-netrc-file /data/aigc/cyb/zxgu/.secrets/egobody.netrc

# 第二批：灰度头显与长时轨迹
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets monado,lamaria \
  --data-root data/ego_pose_eval_core65

# 第三批：非严格头戴的 RGB 泛化
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py download \
  --datasets princeton365 \
  --data-root data/ego_pose_eval_core65
```

下载后检查 112 条是否齐全：

```bash
/data/aigc/cyb/zxgu/env/worldsearcher/bin/python \
  scripts/download_eval_datasets.py verify \
  --data-root data/ego_pose_eval_core65
```

## 4. Legacy 脚本实际下载什么

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
  --data-root data/ego_pose_eval_core65 \
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
  --data-root data/ego_pose_eval_core65
```

ADT 官方链接清单有效期为 14 天。raw 模式只请求 ADT 的 VRS 与 main ground truth，以及 HOT3D 的 VRS、MPS trajectory/calibration 和当前清单中命中 hand/mask/metadata 的组；不会请求 semidense point cloud、eye gaze 或 depth。

## 5. Legacy 磁盘和传输预期

最终评测输入严格限制为 65 分钟，但上游发布单元不总能精确裁到这 65 分钟：

- ADT/HOT3D preview 模式通常约 5–7 GB 传输；
- EgoBody 只传输 4000 张选中 PV 图像、20 个 PV metadata 成员、20 个
  head-tracking 成员和小型标定；认证 ZIP 不会整包下载。加 exo 时只增加
  同步的 master Kinect 帧；按现有 JPEG 大小，默认结果约 1.5–3 GB，
  exo 预计再增加约 1–2 GB；
- Princeton365 的 user MP4 位于 TAR 内，所选 indoor/outdoor 记录需要先取完整 MP4，预计约 17.5 GB，是默认 profile 的主要传输开销；
- Monado 与 LaMAria 不下载整包，只取所选 PNG 和小型 metadata；
- 默认在成功生成 clip 后删除临时完整 MP4；加 `--keep-source` 才保留。

因此建议下载过程中为 preview profile 预留约 30–50 GB；清理缓存后，当前
固定副本为 13,968,075,942 bytes。`--aria-mode raw` 还需额外预留几十 GB，
具体取决于当期官方归档大小。

## 6. Legacy 输出结构

```text
ego_pose_eval_core65/
├── evaluation_manifest.json
├── _cache/
│   └── remote_zip/                 # 仅下载过程中存在；成功后应删除
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

## 7. Legacy 已知边界

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
