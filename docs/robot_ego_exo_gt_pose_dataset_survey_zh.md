# 机器人 Ego / 第三人称视频 / GT 位姿数据集调研

> 调研日期：2026-07-23
> 目标：寻找可替代当前 EgoBody demo 数据的数据集，即同一段机器人运动中同时包含 ego RGB 视频、同步第三人称 RGB 视频，以及 ego 相机（理想情况下是机器人头部）的逐帧 GT 6DoF 位姿。

## 结论摘要

有类似数据，但“真实机器人头部相机 + 同步第三人称视频 + 可用的逐帧 6DoF GT”这一严格交集很小。

1. **字面上最符合的是 Oxford-IHM**：Toyota HSR 的头部 RGB-D、静态外部 RGB-D、Vicon 和完整 ROS TF 都在同一 rosbag 中。它最接近当前 EgoBody 的数据契约，但只有约 60 分钟、内容是室内导航/人机交互，并且需要申请访问。
2. **最适合快速迁移现有 demo 的是 RH20T**：把腕部/手内相机当作 robot ego，使用全局相机当作第三人称视角；数据公开了时间戳、相机标定、机器人 TCP 位姿以及固定手眼变换，可生成逐帧 ego 相机 SE(3)。它不是真正的“头部”，但对机器人操作场景最实用。
3. **需要更大规模和场景多样性时选 DROID**：腕部相机、两个外部相机和逐帧机器人状态齐全；原始 HDF5 还保存逐帧腕部相机外参。应只使用 2025 年发布的高精度标定子集，并注意必须读取 raw 数据，RLDS 简化版不包含完成本任务所需的全部相机外参。
4. **HABIT 的呈现形式非常合适，但不满足 GT 条件**：它有五路同步 RGB（机器人中心、双腕、人类头戴、第三人称），且提供约 1 GB 的 60-task sample；不过公开 LeRobot schema 没有相机内外参，也没有人类头部位姿，不能直接做当前 demo 的定量 GT 对比。
5. REASSEMBLE、FurnitureBench 等虽然有腕部和外部视频及末端位姿，但缺少公开的腕部相机到末端的固定手眼变换；在作者补充标定前，只能把末端位姿当 proxy，不能标为相机 GT。

因此建议采用双线方案：**申请 Oxford-IHM 验证“真正机器人头部”版本，同时先用 RH20T cfg5 做可落地的机器人操作版 demo**。DROID 放在 RH20T 跑通后扩规模。

## 1. 与当前 demo 对齐的判定标准

当前仓库的 EgoBody demo 使用同步的 HoloLens PV ego RGB 和 Kinect 第三人称 RGB，并把逐帧头部/相机位姿变换到第三人称相机坐标系中，详见仓库的 [README_zh.md](../README_zh.md)。要无损替换数据集，至少需要：

- 同一 episode 的 ego RGB；
- 同一 episode 的第三人称 RGB，且能看到机器人或其运动空间；
- 两路视频有可核对的时间戳或硬同步关系；
- 每个 ego 帧有公制 6DoF 位姿，且能通过外参变换到第三人称相机坐标系；
- ego 和 exo 相机内参、外参及变换方向有文档或可验证实现。

本报告严格区分四种“GT”：

| 等级 | 含义 | 可否作为当前 demo 的 GT |
|---|---|---|
| 直接测量 | Vicon / OptiTrack 等外部系统测得的 6DoF | 是，优先级最高 |
| 运动学派生 | 机器人编码器/TCP 位姿 + 已标定的固定手眼变换 | 可以，但应标注为 kinematic reference，而不是独立 mocap GT |
| 代理位姿 | 只有末端、底盘或里程计，缺少相机固定外参 | 否；只能用于可视化或弱监督 |
| 估计位姿 | SLAM / VIO / AMCL 结果 | 否；不能同时充当待评估算法的 GT |

## 2. 候选总览

### 2.1 严格候选与工程上可用的候选

| 数据集 | Ego | 第三人称 | 位姿来源 | 规模/访问 | 本任务判断 |
|---|---|---|---|---|---|
| **Oxford-IHM** | HSR 头部 ASUS Xtion RGB-D，30 Hz | 静态 RealSense D435 RGB-D，30 Hz | Vicon 100 Hz + `/tf` / `/tf_static` | 约 60 分钟；需 GDPR 申请 | **A：本次调研中唯一明确的字面匹配** |
| **RH20T** | 1–2 路手内相机，RGB 10 Hz | 每套平台 8–10 路全局 RGB-D | TCP 100 Hz + 固定 `tc_mat` + 相机标定 | 11 万+序列；可按 cfg 下载 | **A-：最推荐的实用替代，ego 是手腕而非头部** |
| **DROID** | 腕部 ZED Mini | 两路外部 ZED 2 | raw HDF5 中逐帧相机外参；由机器人 Cartesian state 和手眼标定生成 | 7.6 万条/350 h；raw 8.7 TB | **A-：规模最好，但数据和标定筛选成本高** |

### 2.2 视觉结构接近，但 GT 条件不完整

| 数据集 | 已满足部分 | 缺口 | 判断 |
|---|---|---|---|
| **HABIT** | 五路同步 RGB：机器人中心、双腕、人类头戴、全局 exo；10,563 episodes / 164.19 h | 公开 schema 无相机内外参、腕部手眼变换和人类头部位姿 | 很适合展示或无 GT 推理，不适合现有定量评测 |
| **REASSEMBLE** | 双外部 HAMA RGB、腕部 D435i、逐流时间戳、末端 7DoF pose | mocap JSON 只有两路外部相机、事件相机和任务板；没有 `hand` 相机位姿/手眼变换 | 获得作者补充标定后可升级为强候选 |
| **FurnitureBench** | 腕部 + 正面视频，原始采集另有后视；末端位置/四元数 | 发布数据/采集代码记录前后相机到 base 的外参，但没有真实腕部相机到末端的外参 | 只能使用末端 proxy；数据许可范围也需向作者确认 |
| **NavWareSet** | HSR 头部 RGB-D + Ground-Truth Recording Station 外部 RGB | 发布的机器人 pose 是 SLAM 提取的平面 `x,y,yaw`，不是直接测量的头部 6DoF | 适合社会导航，不适合本项目 GT |
| **PUT Messor II** | 机载 Xtion RGB-D + OptiTrack 相机轨迹，5 个序列 | 页面虽展示 external/on-board 预览，但下载结构只列一套 `images.zip`；未发现独立同步 exo RGB 发布 | 需联系作者确认；本次抽查中部分图像下载链接已失效 |

### 2.3 有真正机器人头部 ego，但没有第三人称视频

| 数据集 | 头部/位姿信息 | 不满足原因 |
|---|---|---|
| **HIW-500** | Unitree G1 头部双目 RGB 30 FPS、29DoF、IMU、里程计、相机标定；500+ h | 没有外部第三人称相机 |
| **AgiBot World Beta** | 头部 yaw/pitch、底盘 odometry、相机内外参、百万级轨迹 | 发布的多路视角均为机器人机身/双腕等 onboard 相机，没有 exo；odometry 也不是独立 GT |
| **Humanoid Everyday** | Unitree G1/H1 ego RGB-D、IMU、odometry/kinematics、`head_rmat` | 没有同步第三人称视频 |
| **M2DGR** | 地面机器人多路机载 RGB（含 `/camera/head`），室内段有 Vicon GT | 七路 RGB 都安装在机器人上，不是外部第三人称视角 |

这几类数据说明：大规模 humanoid 数据已经普遍包含头部相机和机器人状态，但公开采集通常没有外部相机；反过来，多视角操作数据通常把移动 ego 相机安装在腕部而不是头部。

## 3. 重点数据集分析

### 3.1 Oxford Indoor Human Motion（最严格匹配）

[官方数据页](https://ori.ox.ac.uk/publications/datasets/oxford-indoor-human-motion-dataset-2024)明确给出约 60 分钟 rosbag、静态和机器人视角 RGB-D，以及 100 Hz mocap。其[详细 About 页](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/about/)进一步列出：

- Toyota HSR 头部 ASUS Xtion Pro Live RGB-D；
- 静态 Intel RealSense D435 RGB-D；
- Vicon 以 100 Hz 追踪人、机器人、外部相机和目标位置；
- `/tf` 和 `/tf/static` 提供机器人及相机 frame；
- 机器人头部 RGB 与外部 RGB 都为 30 Hz；
- 相机传感器使用带反光标记的定制支架。

这意味着可以把 Vicon/TF 统一到 world frame 后得到：

```text
T_exo_ego(t) = inverse(T_world_exo) @ T_world_head_camera(t)
```

它与当前 EgoBody 的 `world -> ego/head -> Kinect/exo` 变换链最相似。主要限制是：

- 机器人由操作者在 7.1 m × 4.2 m 场地中移动、跟随行人，内容不是操作任务；
- 只有约 60 分钟，场景和主体数量有限；
- 因包含人物影像，[下载需要说明用途和 GDPR 保护策略](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/downloads/)；
- 许可是 CC BY-NC-SA 4.0，面向非商业学术使用。

**建议用途**：作为“真正机器人头部”版的定量 sanity check 和论文图，而不是大规模训练集。

### 3.2 RH20T（最推荐先做）

[RH20T 官方页](https://rh20t.github.io/)给出的采集平台包含 1–2 路手内相机、8–10 路全局 RGB-D 相机，并声明所有相机相对机器人 base 完成标定且时间同步。RGB 为 10 Hz，gripper Cartesian pose 为 100 Hz；总计超过 11 万条真实接触丰富的机器人操作序列。

其[官方 API](https://github.com/rh20t/rh20t_api)提供：

- 每个 configuration 的手内相机 serial 和固定 TCP-to-camera 变换 `tc_mat`（见 [`configs/configs.json`](https://github.com/rh20t/rh20t_api/blob/main/configs/configs.json)）；
- `tcp_base.npy` 和按图像时间戳插值的 `get_tcp_aligned`；
- base-aligned 相机外参和 TCP/world/camera 变换函数（见 [`transforms.py`](https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/transforms.py) 与 [`scene.py`](https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/scene.py)）。

统一变换约定后，逐帧 ego 相机 pose 可写成：

```text
T_base_ego(t) = T_base_tcp(t) @ T_tcp_ego
T_exo_ego(t)  = inverse(T_base_exo) @ T_base_ego(t)
```

需要使用 API 或通过一次重投影检查确认矩阵的 active/passive 方向，不能只按变量名猜测。

下载方面，适合先做 cfg5：

- 320×180 video-compressed RGB：8.2 GB；
- LowDim：6.3 GB；
- Calibration：79.1 MB；
- 合计约 14.6 GB 压缩数据，适合先验证整个链路；
- 展示质量确认后，再换 640×360 RGB（cfg5 为 37 GB）。

许可需按 scene 分开：`scene_0001`–`scene_0005` 属 RH20T-C，CC BY-SA 4.0；`scene_0006`–`scene_0010` 属 RH20T-NC，CC BY-NC 4.0。官方页还提示视频可能包含志愿者脸和声音，使用与展示前需要做隐私审查。

**建议用途**：当前 demo 的首选机器人操作版。界面文案应从 “head pose” 改成 “ego-camera pose (kinematic reference)” 或明确写 “wrist-camera pose”。

### 3.3 DROID（扩规模首选）

[DROID 官方项目页](https://droid-dataset.github.io/)给出 76k demonstrations / 350 h、564 scenes、86 tasks；标准硬件是 Franka、两个可调外部 ZED 2 和一个腕部 ZED Mini。官方[数据文档](https://droid-dataset.github.io/droid/the-droid-dataset)显示每步有 wrist RGB、两路 exterior RGB 和 6D Cartesian state；raw 版还包含三路 full-HD 视频、`trajectory.h5` 和相机原始信息。

对本项目最关键的是，官方采集代码中的 [`get_camera_extrinsics`](https://github.com/droid-dataset/droid/blob/main/droid/robot_env.py#L97-L105)会用当前 `cartesian_position` 更新手部相机外参，并将其写入 observation。raw HDF5 因而包含逐帧 wrist camera-to-base pose，而不只是末端 proxy。

但有三个实际风险：

1. 官方在 2025 年为 36k episodes 发布了[更高精度相机标定](https://huggingface.co/KarlP/droid)，并明确说明原始标定较 noisy。建议只从这 36k 中选 episode；该补丁同时提供 cam-to-base、cam-to-cam 和约 72k 条内参。
2. 1.7 TB RLDS 版的公开 schema 只列 RGB 和机器人状态，不暴露本任务需要的逐帧相机外参；要做严格 GT，应读取 raw `trajectory.h5` 并与 MP4/相机 capture timestamp 对齐。
3. raw 数据总量约 8.7 TB。应按 episode ID 选择性下载，而不是整桶同步。

[DROID 论文](https://autolab.berkeley.edu/assets/publications/media/2024-RSS-DROID.pdf)说明完整数据按 CC BY 4.0 发布。它比 RH20T 更有场景多样性，但首次接入的数据筛选和版本管理成本更高。

**建议用途**：RH20T adapter 跑通后复用相同的数据接口，扩大场景和任务覆盖。

### 3.4 HABIT（2026 年的新数据，展示强、GT 弱）

[HABIT 官方页面](https://habit-dataset.github.io/)和[数据卡](https://huggingface.co/datasets/configinc/HABIT)给出：10,563 episodes、164.19 h、60 tasks、5.91M frames、253 GB、10 FPS、CC BY 4.0。每个 episode 有五路同步 RGB：

- `front_view`：机器人中心前视；
- `left_wrist_view` / `right_wrist_view`：双腕；
- `human_front_view`：人类头戴 ego；
- `exo_view`：覆盖整个人机工作区的第三人称视角。

它在“ego + exo 的视觉呈现”上非常契合，而且 sample 配置覆盖全部 60 个任务、约 1 GB。问题是当前公开 `meta/info.json` / `meta/modality.json` 只包含双臂 Cartesian、joint、gripper 状态和视频键，没有相机内参、外参、腕部手眼变换或人类头部 SE(3)。机器人中心相机又是固定视角，不会产生当前 demo 想评估的移动 ego trajectory。

**建议用途**：可用于快速制作多视角展示、验证 UI 和视频同步；如需定量位姿，先向作者索要 wrist-to-EEF、exo-to-base 和内参。不要把公开的 EEF pose 直接标为 camera GT。

### 3.5 REASSEMBLE（轻量，但差一个关键标定）

[REASSEMBLE 项目页](https://tuwien-asl.github.io/REASSEMBLE_page/)给出 4,551 demonstrations（4,035 成功）、781 分钟、两路外部 HAMA RGB、腕部 RealSense D435i、末端位姿和独立传感器时间戳。其[正式数据发布页](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44)为 CC BY 4.0，总计 54.8 GiB，易于下载。

但正式发布页也说明 mocap 只在每次采集开始时测量相机与 board pose，避免持续 mocap 干扰 event camera；公开 JSON 只有 `Hama1`、`Hama2`、`DAVIS346` 和 `NIST_Board1`，没有腕部 `hand` 相机。HDF5 虽有逐帧末端 `pose`，但没有文档化的 hand-to-EEF 固定变换和相机内参。

**建议用途**：联系作者补齐两项标定后可成为很好的小规模 pilot；在此之前只能作为视觉 demo 或末端 proxy 实验。

## 4. 不建议作为主数据的代表性近似集

- [FurnitureBench](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html)：5,100 条成功示范、219.6 h，腕部与前视图、EEF pose 齐全；但未公开真实腕部手眼外参，且数据页没有清晰声明数据本身的许可，仓库 MIT 许可不能自动等同于数据许可。
- [RoboSet](https://robopen.github.io/roboset/teleoperation.html)：约 30,050 trajectories、四视角和 EEF 状态，但当前页面仍只明确开放约 9,500 条 teleop 轨迹，固定腕部外参/统一相机标定文档不足。
- [NavWareSet](https://anr-navware.github.io/navwareset/)：有 HSR 头部视频与外部 Ground-Truth Recording Station RGB，但所谓 robot pose 由 SLAM 提取，最终 CSV 只有平面 `x,y,yaw`。
- [PUT Messor II](https://lrm.put.poznan.pl/put-messor-ii-state-estimation-dataset/)：OptiTrack 相机轨迹很合适，但第三人称录像是否实际发布不清楚，当前部分下载链接不可用。
- [HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500)、[AgiBot World](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)、[Humanoid Everyday](https://github.com/physical-superintelligence-lab/Humanoid-Everyday)：非常适合研究 robot-head ego，但需要自行补采同步 exo 相机。
- [Ego-Exo4D](https://docs.ego-exo4d-data.org/overview/)：同步 ego/exo 很强，但主体是人类，Aria trajectory 是 MPS/VIO 估计而非 mocap GT；它更适合当视觉域预训练数据，不是机器人 GT benchmark。

## 5. 与现有代码的接入映射

建议在现有 EgoBody I/O 上方定义一个与数据集无关的最小协议：

| 统一字段 | RH20T | DROID | Oxford-IHM |
|---|---|---|---|
| `ego_rgb[t]` | in-hand `color.mp4` | wrist MP4 | HSR head RGB topic |
| `exo_rgb[t]` | 任一 global `color.mp4` | exterior 1/2 MP4 | static D435 RGB topic |
| `ego_timestamp[t]` | `timestamps.npy` | camera capture timestamp | ROS message timestamp |
| `T_world_ego[t]` | TCP + `tc_mat` 派生 | raw HDF5 `camera_extrinsics` | Vicon + TF |
| `T_world_exo` | calibration / base-aligned extrinsic | 高精度 cam-to-base 补丁 | Vicon / TF static |
| `K_exo` | calibration | SVO 或 intrinsics patch | `camera_info` |

随后保留当前 pipeline 的 DA3 推理、Sim(3) 对齐、轨迹绘制和第三人称合成，只替换数据读取与坐标变换模块。

实现时有四个硬性校验：

1. **时间同步**：记录每个配对帧的时间差；对 10 Hz 数据建议不超过 50 ms，并把超过阈值的帧剔除，而不是静默 nearest-neighbor。
2. **变换方向**：用已知 TCP/机器人模型或 AprilTag 做一次像素重投影，确认 `camera-to-world` 与 `world-to-camera` 没有取反错误。
3. **GT 不泄漏**：GT 只用于尺度/评估的既定阶段；保持当前 DA3 推理不读取 GT 的约束。
4. **位姿命名**：Oxford 可写 `head-camera mocap/TF GT`；RH20T/DROID 应写 `wrist-camera kinematic reference`；缺手眼标定的数据只能写 `EEF proxy`。

## 6. 建议的落地顺序

### P0：RH20T 小规模验证

1. 下载 cfg5 的 320×180 RGB、LowDim、Calibration，约 14.6 GB。
2. 只使用 `scene_0001`–`scene_0005`，避免混入 NC 子集；从一个 in-hand 相机和一个无遮挡的 global 相机开始。
3. 自动筛选 3 段约 20 秒片段，要求 wrist camera 同时有明显平移和旋转、机器人始终在 exo 画面内、时间戳缺失率低。
4. 先通过 GT 轨迹重投影和坐标系单测，再运行 DA3；避免用 DA3 结果反向调外参。
5. 若 320×180 的展示效果不足，再只替换选中片段对应的 640×360 RGB。

### P1：Oxford-IHM 严格头部版本

提交访问申请，说明仅用于机器人相机位姿研究、访问控制、加密存储、不会公开原始人物画面。数据拿到后先解析 ROS topic/TF tree，再选短片跑与 EgoBody 相同的 head-camera 评测。

### P2：DROID 扩规模

从高精度标定覆盖的 36k episode 中筛选；先下载一个 raw episode 验证 HDF5/MP4/timestamp/外参 convention，再批量扩展。不要以 2 GB 的 `droid_100` RLDS sample 作为 GT 接入依据，因为其简化 schema 不含完整相机外参。

### 可选：HABIT 展示分支

直接使用约 1 GB sample 做五视角 UI 和人机交互内容展示，但关闭 GT 误差曲线，或明确标成 `no pose GT`。若作者补充相机标定，可将双腕视角接入 P0 的同一 adapter。

## 7. 最终决策

- **若“头部”不可妥协**：选 Oxford-IHM；现阶段没有发现同等开放、同时更大规模的严格替代。
- **若目标是尽快把现有 demo 迁到真实机器人数据**：选 RH20T，接受 wrist-as-ego，并准确标注 GT 来源。
- **若目标是规模与多样性**：选 DROID raw + 2025 calibration subset。
- **若目标只是多视角呈现效果**：HABIT sample 最省事，但不能用于 GT 轨迹结论。
- **不建议**在缺少 hand-eye calibration 时把 EEF pose 当成 camera/head GT；这会引入恒定但不可忽略的平移和旋转偏差，并污染 ATE/RPE、投影和可视化结论。

综合可用性、数据量、标定完整度和现有代码迁移成本，本项目的推荐优先级是：

```text
RH20T pilot  ->  Oxford-IHM strict head validation  ->  DROID scale-up
      \
       -> HABIT visual-only branch（可选）
```
