# 机器人 Ego / 第三人称视频 / GT 位姿数据集调研

> 调研日期：2026-07-23
> 目标：寻找可替代当前 EgoBody demo 数据的数据集，即同一段机器人运动中同时包含 ego RGB 视频、同步第三人称 RGB 视频，以及 ego 相机（理想情况下是机器人头部）的逐帧 GT 6DoF 位姿。

## 结论摘要

有类似数据，但“真实机器人头部相机 + 同步第三人称视频 + 可用的逐帧 6DoF GT”这一严格交集很小。

1. **严格匹配有两套，而不是一套：Oxford-IHM 与 Vernissage。** Oxford-IHM 的 HSR 头部 RGB-D、静态外部 RGB-D、Vicon 和 ROS TF 在同一 rosbag；Vernissage 则有 NAO 头部 RGB、三路外部 HD、100 Hz Vicon、同步文件，并直接发布 `nao.csv: Nao's head pose` 的数据格式说明。两者都需要申请。Vernissage 的机器人底座不移动，只有头部转向/点头；其 HD 相机到 Vicon 的标定在论文中注明“未验证”，拿到数据后必须先做重投影检查。
2. **MuMMER 是第二梯队中最接近“真头部”的数据。** Pepper 头部相机、静态 Kinect 第三人称相机、D435、机器人 `joint_states` 全部由同一台电脑写入 ROS bag，时间同步很好；可通过 Pepper 模型和头部 yaw/pitch 派生相机姿态。但公开文档没有给 Kinect-to-robot 外参，也没有声称对机器人做了独立 mocap，因此应称为 `kinematic reference`，不是跨相机 GT。Zenodo 同样需要申请。
3. **无需审批、现在就能下载的 RoboCup 2023–2024 只能用于视觉 demo。** 它有 TIAGo 头部 RGB-D、`joint_states`、`tf/tf_static`、`robot_description`、里程计和手机第三人称视频，Zenodo 为 CC BY 4.0；但论文没有给手机视频的时钟偏移或外参。抽查 `receptionist_try_2` 时，rosbag 为 197.855 s，而对应公开 MP4 只有 27.136 s，说明不能按同起点/同长度直接配对，更不能把 odometry/TF 当 mocap GT。
4. **最适合快速迁移现有 demo 的真实机器人数据仍是 RH20T。** 把腕部/手内相机当作 robot ego，使用全局相机当作第三人称视角；TCP、固定手眼变换、时间戳和多相机标定足以生成逐帧 ego 相机 SE(3)。它不是真正的头部，但数据可用性最好。DROID 可在 adapter 跑通后用于扩规模。
5. **若必须立即获得“严格、无噪声”的成对数据，可用 RoboTwin 2.0 生成仿真 pilot。** 开启 `third_view: true` 后可同时保存移动腕部 ego 与 observer 第三人称 RGB；每帧已经返回相机内参、外参和 `cam2world`。observer 的固定 pose 在源码中硬编码但默认没有写入 pkl，需要补存一次。它只能验证数据接口和几何链路，不能替代真实机器人结论。
6. HABIT、HARPER、REASSEMBLE、FurnitureBench 等各差一个关键条件：要么没有相机位姿，要么第三人称视频没有同步/外参，要么缺少手眼标定，不能在报告或 UI 中标成 head-camera GT。

因此建议采用三线方案：**Oxford-IHM 继续等待；同时申请 Vernissage 和 MuMMER；立即用 RH20T（真实腕部）或 RoboTwin（仿真精确 GT）跑通 adapter。** RoboCup 可用来快速看真实 TIAGo 头部画面和呈现形式，但关闭 GT 误差曲线。

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
| **Oxford-IHM** | HSR 头部 ASUS Xtion RGB-D，30 Hz | 静态 RealSense D435 RGB-D，30 Hz | Vicon 100 Hz + `/tf` / `/tf_static` | 约 60 分钟；需 GDPR 申请 | **A：严格匹配，移动底座 + 头部** |
| **Vernissage** | NAO 头部单目 RGB，约 15 Hz | 3 路 HD，1080p/25 Hz | Vicon 100 Hz，发布格式直接含 `nao.csv` 头位姿 | 13 × 约 11 分钟；邮件申请 | **A：严格匹配，底座静止、头部转动；外参需复核** |
| **MuMMER** | Pepper 头部 RGB/深度；另有头部 D435 | 静态 Kinect v2，15 Hz | 头部关节编码器 + Pepper 运动学 | 33 段/1 h 29 min；Zenodo restricted | **B+：真头部、同步好；缺跨相机外参和独立 GT** |
| **RH20T** | 1–2 路手内相机，RGB 10 Hz | 每套平台 8–10 路全局 RGB-D | TCP 100 Hz + 固定 `tc_mat` + 相机标定 | 11 万+序列；可按 cfg 下载 | **A-：最推荐的实用替代，ego 是手腕而非头部** |
| **DROID** | 腕部 ZED Mini | 两路外部 ZED 2 | raw HDF5 中逐帧相机外参；由机器人 Cartesian state 和手眼标定生成 | 7.6 万条/350 h；raw 8.7 TB | **A-：规模最好，但数据和标定筛选成本高** |

### 2.2 可直接下载，但不满足严格 GT

| 数据集 | 已满足部分 | 缺口 | 判断 |
|---|---|---|---|
| **RoboCup 2023–2024 ROSbag** | TIAGo 头部 RGB-D、joint/TF/URDF/odom、手机第三人称视频；按任务直接从 Zenodo 下载 | 手机视频无公开同步偏移和外参；odom/TF 不是外部 GT，且 MP4 可能只覆盖 rosbag 的一部分 | 可立即做真实 robot-head 视觉 demo；不可做 GT 评测 |
| **Barcelona Robot Lab** | 机载双目、17 路固定 IP 相机、时间戳、内外参；完整集 6.4 GB，可直接下载 | 机器人运动只有轮式 odometry + compass，非 6DoF GT | 很适合测试多摄像头读取和 UI；不适合定量轨迹基准 |
| **RoboTwin 2.0（仿真）** | 可同步生成移动腕部 RGB、observer RGB，每帧精确内外参/`cam2world` | 非真实数据；默认发布配置 `third_view: false`，需本地生成；ego 是腕部而非头部 | 最快的严格几何单测和 adapter pilot |

### 2.3 视觉结构接近，但 GT 条件不完整

| 数据集 | 已满足部分 | 缺口 | 判断 |
|---|---|---|---|
| **HABIT** | 五路同步 RGB：机器人中心、双腕、人类头戴、全局 exo；10,563 episodes / 164.19 h | 公开 schema 无相机内外参、腕部手眼变换和人类头部位姿 | 很适合展示或无 GT 推理，不适合现有定量评测 |
| **HARPER** | Spot 机载相机、OptiTrack 中的机器人 rigid body/运动学、外部 RGB；Spot 与 OptiTrack 对齐误差低于 2 ms | 官方 README 明确说外部 RGB 尚未与其余数据对齐；Spot 也没有“头部”相机；当前脚本下载链接实测返回 403 | 研究内容很接近，但当前发布状态不适合直接接入 |
| **REASSEMBLE** | 双外部 HAMA RGB、腕部 D435i、逐流时间戳、末端 7DoF pose | mocap JSON 只有两路外部相机、事件相机和任务板；没有 `hand` 相机位姿/手眼变换 | 获得作者补充标定后可升级为强候选 |
| **FurnitureBench** | 腕部 + 正面视频，原始采集另有后视；末端位置/四元数 | 发布数据/采集代码记录前后相机到 base 的外参，但没有真实腕部相机到末端的外参 | 只能使用末端 proxy；数据许可范围也需向作者确认 |
| **NavWareSet** | HSR 头部 RGB-D + Ground-Truth Recording Station 外部 RGB | 发布的机器人 pose 是 SLAM 提取的平面 `x,y,yaw`，不是直接测量的头部 6DoF | 适合社会导航，不适合本项目 GT |
| **PUT Messor II** | 机载 Xtion RGB-D + OptiTrack 相机轨迹，5 个序列 | 页面虽展示 external/on-board 预览，但下载结构只列一套 `images.zip`；未发现独立同步 exo RGB 发布 | 需联系作者确认；本次抽查中部分图像下载链接已失效 |

### 2.4 有真正机器人头部 ego，但没有第三人称视频

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

#### 3.1.1 公开样例与第三方镜像核验（2026-07-23）

公开内容需要分成“视觉预览”和“可运行的传感器样例”两类：

- **官方视觉预览存在**：[项目首页](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/)公开了场景合成图；[About 页](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/about/)另有场地布局、地图和 Vicon 标记布置图。这些图能确认采集场景和硬件，但不是头部/外部相机的同步原始帧。
- **官方补充视频存在**：[论文补充视频](https://youtu.be/gdC3mpZNjG4)公开展示了 HSR、行人、第三人称实验画面以及轨迹/代价场可视化。它适合判断场景与呈现效果，但不是 Oxford-IHM rosbag 的可复用小样：没有发布逐帧的两路原始 RGB-D、Vicon/TF 和相机标定。
- **官方代码仓库不含 bag**：[trajectory_prediction_ros](https://github.com/ori-arg/trajectory_prediction_ros)的 README 明确要求用户自行把 dataset bags 放入 `bags` 目录；仓库没有 release。仓库只提供代码及一份 `map.pgm`/`map.yaml`，完整 Git 历史中也未发现被删除的 bag 或视频。launch 文件泄露出的一个实际包名是 `merged_map4_run3.bag`，但该文件没有随代码发布。
- **第三方公开镜像未发现**：本轮用数据集全名、`Oxford-IHM`、内部目录名 `HumanTrajectoryPredictionDataset` 和实际包名 `merged_map4_run3.bag`，核验了通用网页搜索、GitHub 及其全部公开 forks、GitLab、Hugging Face Datasets、Zenodo、Internet Archive，以及 Google Drive/Dropbox/中文网盘相关索引；未找到可下载的 rosbag、裁剪片段或转换后的同步样例。
- **不建议使用来源不明的第三方 raw 上传**：[下载页](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/downloads/)按 GDPR 审批访问；其申请表条件还要求数据不得分享给第三方。因此，即使后续发现未经作者确认的 raw mirror，也应先向 Oxford 团队核实授权，不宜直接纳入项目。

当前能公开直取、对接有一点帮助的只有代码、二维地图、场景图和论文视频；它们都不能替代一个带以下内容的短 rosbag：

```text
/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed
/camera/color/image_raw/compressed
/tf
/tf/static
/vicon/d435_4_markers/d435_4_markers
/vicon/person_1/person_1
相机内参及 head-camera / marker 刚体标定
```

最省时间的获取方式不是继续找第三方镜像，而是在已提交申请的基础上邮件
`oxford-ihm-dataset@robots.ox.ac.uk`，明确只索要一个 30–60 秒、经过适当匿名化的
sample rosbag，或任意一条约 5 分钟 run；这样既降低传输成本，也仍走官方 GDPR
授权链路。

### 3.2 Vernissage Corpus（第二个严格头部候选）

[Vernissage 官方旧站](http://vernissage.humavips.eu/)和[论文](https://publications.idiap.ch/downloads/reports/2012/Jayagopi_Idiap-RR-33-2012.pdf)描述的是 NAO 与两名参与者进行讲解、问答的多模态 HRI 数据。它被上一版检索遗漏，但实际上非常贴合本项目：

- 13 场 session，每场约 11 分钟，总量约 143 分钟；
- NAO 头内单目 RGB，VGA、平均约 15 FPS，机器人会转头、点头；
- 三路外部 HD 相机，1920 × 1080、25 FPS；
- Vicon 以 100 Hz 记录人和 NAO 的 6DoF；
- NAO 的全部关节角、里程计与系统状态也被记录；
- [官方数据格式页](http://vernissage.humavips.eu/data.html)明确列出 `nao.csv: Nao's head pose`、`persons.csv`、Vicon/RSB 时间同步文件及外部视频。

同步链是有文档的：外部 HD 视频的音轨与 NAO 音频做互相关，Vicon 则通过带反光标记的 clapperboard 对齐到 RSB 时间；论文估计最远一台相机仅由 5 m 声传播造成的理论限制约为 14.6 ms。相机标定方面，作者向全部相机展示 checkerboard，也把 Vicon 标定杆展示给三台 HD 相机，因此原则上能求：

```text
T_exo_head(t) = inverse(T_vicon_exo) @ T_vicon_head(t)
T_exo_camera(t) = T_exo_head(t) @ T_head_camera
```

但必须保留两个警告：

1. 论文对 HD-camera-to-Vicon registration 明写了 **“not validated”**；拿到数据后应先用标定杆或场景已知点做重投影，而不是直接把它当可信外参。
2. `nao.csv` 是头部 rigid-body pose。若当前指标实际评估 optical camera center，还需从 NAO 模型或标定文件获得固定的 `T_head_camera`。

访问不是自动下载。[旧站联系页](http://vernissage.humavips.eu/contact.html)要求邮件申请，历史地址为 `{jwienke,swrede}@techfak.uni-bielefeld.de`；Sebastian Wrede 的[当前 Bielefeld 官方页面](https://www.uni-bielefeld.de/fakultaeten/technische-fakultaet/forschung/ag-ueberblick/cognitive-systems-enginee/)列出 `sebastian.wrede@uni-bielefeld.de`，建议发当前地址并抄送历史地址。旧站仍可访问但响应很慢，许可也没有在公开页清楚声明，应在邮件中一并确认。

**建议用途**：与 Oxford 并行申请。若重点是“头部旋转轨迹”而非移动导航，它甚至比 Oxford 有更长时长和更多 exo 视角；若需要明显的平移轨迹，则它不合适，因为 NAO 底座在场景中固定。

### 3.3 MuMMER（真头相机，运动学参考）

[MuMMER 官方数据页](https://www.idiap.ch/en/scientific-research/data/mummer/index_html)给出 33 段、1 h 29 min、28 名参与者的 Pepper 多人交互。其[论文](https://www.idiap.ch/~odobez/publications/CanevetHeMotlicekOdobez-ROMAN2020.pdf)确认：

- Pepper 头部前向 RGB 为 640 × 480、约 8 FPS，头部深度约 5 FPS；
- Intel D435 在第一天装于 tablet 顶部、第二天装于头顶，15 FPS；
- Kinect v2 固定在机器人后方，覆盖整个场景，960 × 540、15 FPS；
- 三路视频、音频、机器人 `joint_states` 和人物 mocap 位置均由同一台电脑写入 ROS bag，以同一机器时钟时间戳同步；
- 交互动作包含看向指定人物、点头等，头部相机会产生明确 ego-motion。

由头部 yaw/pitch 编码器和 Pepper 的固定相机几何可得到：

```text
T_base_camera(t) =
    T_base_head_yaw(q_yaw[t])
  @ T_head_yaw_head_pitch(q_pitch[t])
  @ T_head_pitch_camera
```

它的优点是真实 Pepper 头相机与静态第三人称视频确实同步。缺点是公开论文只说 mocap 记录“protagonists”的 3D 位置，没有说 mocap 追踪 Pepper；公开页面也没有 Kinect-to-base 外参。因此，访问前不能声称能把头相机姿态变换到 Kinect 坐标，更不能称为独立 GT。

正确入口是 [Zenodo record 3989642](https://zenodo.org/records/3989642)，DOI `10.34777/5p84-cq41`；它目前是 restricted access，元数据未声明公开许可。项目页上的旧跳转可能误指向另一个 ManiGaze record，申请时应核对标题为 **MuMMER dataset**。建议在申请备注里明确索要：

- raw ROS bags，而不只是模糊化后的三联视频；
- Pepper `joint_states`、`tf/tf_static`、URDF/robot description；
- 三台相机的 `camera_info`；
- Kinect/D435 相对 Pepper base 或共同 world 的外参；
- 数据使用许可。

**建议用途**：如果作者确认 rosbag 含 TF 与 Kinect 外参，可升级到 A-；若没有外参，它仍可作为 head-camera rotation 的 kinematic-reference demo。

### 3.4 RoboCup 2023–2024 ROSbag（可立即下载，但只能视觉用）

[数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615538/)与[Zenodo 总入口](https://zenodo.org/records/13838208)列出 TIAGo 在 RoboCup@Home 2023/2024 的 260 GB 以上记录。2024 按任务拆成多个仓库，例如 [Receptionist](https://zenodo.org/records/13902011) 与 [GPSR](https://zenodo.org/records/13902406)，无需审批，数据记录标注为 CC BY 4.0。

每个 rosbag 的核心内容包括：

- `/head_front_camera/rgb/image_raw`、depth 和 `camera_info`；
- `/joint_states`；
- `/tf`、`/tf_static` 和实际包中存在的 `/robot_description`；
- `/odom`、IMU、LiDAR 与任务日志；
- 独立的、已做人脸模糊化的手机第三人称 MP4。

所以可以从 TF 树提取：

```text
T_odom_head_camera(t)
```

但它只是机器人运动学 + 里程计参考；没有 Vicon/RTK 等外部测量。更严重的是，外部视频只是为了“与观众视角对照”，论文没有提供手机内外参、共同触发、时钟偏移或对齐脚本。本次还直接抽查了 `receptionist_try_2`：

- `metadata_recepcionist_try_2.yaml`：rosbag 时长 197.8551548 s；
- `receptionist_try_2_censored.mp4` 的 MP4 `mvhd`：27.136 s；
- 两者显然不是默认同起点、同终点的逐帧配对。

**建议用途**：若想今天就下载一套“真实 TIAGo 头部 + 第三人称画面”验证 UI，可用 Receptionist；但应显示 `odometry/kinematic pose`，关闭 ATE/RPE 和跨相机投影。除非作者补充手机视频的 offset 与外参，否则不要投入时间做人工硬对齐。

### 3.5 HARPER（机器人 GT 很强，但 exo 发布不完整）

[HARPER 项目页](https://intelligolabs.github.io/HARPER/)记录 Boston Dynamics Spot 与 17 名参与者的 15 类动作，共 607 段、6 万多帧：

- Spot 的 5 路灰度+深度相机及 gripper RGB-D，约 10 FPS；
- OptiTrack 120 Hz；
- Spot 背部有 4-marker rigid body，机器人关节由内部状态做正运动学，再放入 OptiTrack world；
- Spot 与 OptiTrack 的时间对齐误差低于 2 ms；
- 一台外部 RGB 覆盖全场。

这让 Spot 的 body/camera pose 很有潜力，且 robot skeleton 与 mocap world 对齐是真正的外部参考。但[官方仓库 README](https://github.com/intelligolabs/HARPER)明确写着：外部 RGB 视频“right now they are not aligned/synchronized with the rest of the data”，只能尝试利用文件名的录制时间自行对齐；它也没有给外部 RGB 到 OptiTrack 的已验证外参。此外 Spot 没有传统意义上的活动“头部”，移动 RGB-D 位于 gripper。

截至 2026-07-23，本次用官方 `harper_downloader.py` 中的 OneDrive tar 链接做无登录请求时返回 HTTP 403；外部视频又位于单独的 SharePoint 文件夹。因此它目前不比 Oxford 更快落地。

**建议用途**：暂不作为主线。若作者后续补齐同步 offset、exo 外参和稳定下载，可用于“带 mocap 的 quadruped ego”扩展实验。

### 3.6 RH20T（最推荐先做）

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

### 3.7 DROID（扩规模首选）

[DROID 官方项目页](https://droid-dataset.github.io/)给出 76k demonstrations / 350 h、564 scenes、86 tasks；标准硬件是 Franka、两个可调外部 ZED 2 和一个腕部 ZED Mini。官方[数据文档](https://droid-dataset.github.io/droid/the-droid-dataset)显示每步有 wrist RGB、两路 exterior RGB 和 6D Cartesian state；raw 版还包含三路 full-HD 视频、`trajectory.h5` 和相机原始信息。

对本项目最关键的是，官方采集代码中的 [`get_camera_extrinsics`](https://github.com/droid-dataset/droid/blob/main/droid/robot_env.py#L97-L105)会用当前 `cartesian_position` 更新手部相机外参，并将其写入 observation。raw HDF5 因而包含逐帧 wrist camera-to-base pose，而不只是末端 proxy。

但有三个实际风险：

1. 官方在 2025 年为 36k episodes 发布了[更高精度相机标定](https://huggingface.co/KarlP/droid)，并明确说明原始标定较 noisy。建议只从这 36k 中选 episode；该补丁同时提供 cam-to-base、cam-to-cam 和约 72k 条内参。
2. 1.7 TB RLDS 版的公开 schema 只列 RGB 和机器人状态，不暴露本任务需要的逐帧相机外参；要做严格 GT，应读取 raw `trajectory.h5` 并与 MP4/相机 capture timestamp 对齐。
3. raw 数据总量约 8.7 TB。应按 episode ID 选择性下载，而不是整桶同步。

[DROID 论文](https://autolab.berkeley.edu/assets/publications/media/2024-RSS-DROID.pdf)说明完整数据按 CC BY 4.0 发布。它比 RH20T 更有场景多样性，但首次接入的数据筛选和版本管理成本更高。

**建议用途**：RH20T adapter 跑通后复用相同的数据接口，扩大场景和任务覆盖。

### 3.8 HABIT（2026 年的新数据，展示强、GT 弱）

[HABIT 官方页面](https://habit-dataset.github.io/)和[数据卡](https://huggingface.co/datasets/configinc/HABIT)给出：10,563 episodes、164.19 h、60 tasks、5.91M frames、253 GB、10 FPS、CC BY 4.0。每个 episode 有五路同步 RGB：

- `front_view`：机器人中心前视；
- `left_wrist_view` / `right_wrist_view`：双腕；
- `human_front_view`：人类头戴 ego；
- `exo_view`：覆盖整个人机工作区的第三人称视角。

它在“ego + exo 的视觉呈现”上非常契合，而且 sample 配置覆盖全部 60 个任务、约 1 GB。问题是当前公开 `meta/info.json` / `meta/modality.json` 只包含双臂 Cartesian、joint、gripper 状态和视频键，没有相机内参、外参、腕部手眼变换或人类头部 SE(3)。机器人中心相机又是固定视角，不会产生当前 demo 想评估的移动 ego trajectory。

**建议用途**：可用于快速制作多视角展示、验证 UI 和视频同步；如需定量位姿，先向作者索要 wrist-to-EEF、exo-to-base 和内参。不要把公开的 EEF pose 直接标为 camera GT。

### 3.9 REASSEMBLE（轻量，但差一个关键标定）

[REASSEMBLE 项目页](https://tuwien-asl.github.io/REASSEMBLE_page/)给出 4,551 demonstrations（4,035 成功）、781 分钟、两路外部 HAMA RGB、腕部 RealSense D435i、末端位姿和独立传感器时间戳。其[正式数据发布页](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44)为 CC BY 4.0，总计 54.8 GiB，易于下载。

但正式发布页也说明 mocap 只在每次采集开始时测量相机与 board pose，避免持续 mocap 干扰 event camera；公开 JSON 只有 `Hama1`、`Hama2`、`DAVIS346` 和 `NIST_Board1`，没有腕部 `hand` 相机。HDF5 虽有逐帧末端 `pose`，但没有文档化的 hand-to-EEF 固定变换和相机内参。

**建议用途**：联系作者补齐两项标定后可成为很好的小规模 pilot；在此之前只能作为视觉 demo 或末端 proxy 实验。

### 3.10 RoboTwin 2.0（立即可生成的仿真 GT）

[RoboTwin 2.0 文档](https://robotwin-platform.github.io/doc/usage/configurations.html)支持 ALOHA 等双臂机器人、头/全局相机、双腕相机和 observer 视角。其[官方源码](https://github.com/RoboTwin-Platform/RoboTwin)在每个 observation 中已经返回：

```text
observation/<camera>/intrinsic_cv
observation/<camera>/extrinsic_cv
observation/<camera>/cam2world_gl
```

其中左右腕相机每帧跟随末端更新，因此可把任一腕相机作为移动 ego；把配置改为：

```yaml
data_type:
  rgb: true
  third_view: true
  endpose: true
  qpos: true
```

即可在同一仿真 step 保存 `third_view_rgb`。observer pose 固定为源码中的 `[0.0, 0.23, 1.33]` 及对应朝向，但当前 `get_config()` 没把 observer 参数写入 pkl；接入时应调用同样的 `get_intrinsic_matrix()`、`get_extrinsic_matrix()` 和 `get_model_matrix()` 存一次，而不是手抄矩阵。

需要准确命名：RoboTwin 的 `head_camera` 对 ALOHA 是静态/躯干全局观察，不会形成移动头部轨迹；真正适合当前几何 demo 的是 wrist ego。它的价值是能在 Oxford/Vernissage 到手前证明同步、坐标约定、Sim(3) 对齐和可视化都正确。

## 4. 不建议作为主数据的代表性近似集

- [Barcelona Robot Lab](https://www.iri.upc.edu/research/webprojects/pau/datasets/BRL/)：两次约 4 h 记录、17 路固定 IP 相机、机载双目、时间戳、相机标定和 CAD；[完整包 6.4 GB、相机网络与双目也可分开直下](https://www.iri.upc.edu/research/webprojects/pau/datasets/BRL/downloads.php)。但 `odometry.txt` 和 compass 只能给轮式平面运动，`robot_wrt_world.txt` 是初始固定变换而不是逐帧外部 GT。
- [HRI-SENSE](https://zenodo.org/records/14267885)：TIAGo ego RGB-D 与两路静态相机的采集设计很接近，146 sessions / 6 h 03 min，Zenodo 约 3.6 GB 可直接下载；但公开 release 主要给 ego depth、派生人物/人脸 CSV 和机械臂状态，没有活动头部轨迹及完成跨相机变换所需的原始 exo RGB/外参。
- [THÖR-MAGNI](https://zenodo.org/records/10407223)：Azure Kinect robot ego 与 100 Hz Qualisys robot/human trajectory 均公开、CC BY 4.0；但发布包没有同步固定第三人称 RGB，因此只有“ego + GT”，缺 exo。
- [FurnitureBench](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html)：5,100 条成功示范、219.6 h，腕部与前视图、EEF pose 齐全；但未公开真实腕部手眼外参，且数据页没有清晰声明数据本身的许可，仓库 MIT 许可不能自动等同于数据许可。
- [RoboSet](https://robopen.github.io/roboset/teleoperation.html)：约 30,050 trajectories、四视角和 EEF 状态，但当前页面仍只明确开放约 9,500 条 teleop 轨迹，固定腕部外参/统一相机标定文档不足。
- [NavWareSet](https://anr-navware.github.io/navwareset/)：有 HSR 头部视频与外部 Ground-Truth Recording Station RGB，但所谓 robot pose 由 SLAM 提取，最终 CSV 只有平面 `x,y,yaw`。
- [PUT Messor II](https://lrm.put.poznan.pl/put-messor-ii-state-estimation-dataset/)：OptiTrack 相机轨迹很合适，但第三人称录像是否实际发布不清楚，当前部分下载链接不可用。
- [HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500)、[AgiBot World](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)、[Humanoid Everyday](https://github.com/physical-superintelligence-lab/Humanoid-Everyday)：非常适合研究 robot-head ego，但需要自行补采同步 exo 相机。
- [Ego-Exo4D](https://docs.ego-exo4d-data.org/overview/)：同步 ego/exo 很强，但主体是人类，Aria trajectory 是 MPS/VIO 估计而非 mocap GT；它更适合当视觉域预训练数据，不是机器人 GT benchmark。

## 5. 与现有代码的接入映射和验收门槛

建议在现有 EgoBody I/O 上方定义一个与数据集无关的最小协议，再保留当前 DA3 推理、Sim(3) 对齐、轨迹绘制和第三人称合成。

### 5.1 真实头部候选

| 统一字段 | Oxford-IHM | Vernissage | MuMMER |
|---|---|---|---|
| `ego_rgb[t]` | HSR head RGB topic | NAO RSB/video | Pepper head RGB topic |
| `exo_rgb[t]` | static D435 RGB | 任一 HD camera | static Kinect RGB |
| `ego_timestamp[t]` | ROS message time | RSB time | ROS message time |
| `exo_timestamp[t]` | ROS message time | 音频互相关后的 RSB time | ROS message time |
| `T_world_head[t]` | Vicon + TF | Vicon `nao.csv` | 头部 joint + FK，仅运动学参考 |
| `T_world_exo` | Vicon / static TF | HD-to-Vicon calibration，论文称未验证 | 公开资料未发现 |
| `K_ego`, `K_exo` | `camera_info` | checkerboard calibration | 需核对 bag 的 `camera_info` |
| 接入状态 | 严格可用，待取得数据 | 需先验证外参 | 需作者补跨相机外参 |

### 5.2 可立即实施的替代

| 统一字段 | RH20T | DROID | RoboTwin 2.0 | RoboCup |
|---|---|---|---|---|
| `ego_rgb[t]` | in-hand MP4 | wrist MP4 | wrist RGB | TIAGo head RGB |
| `exo_rgb[t]` | global MP4 | exterior MP4 | `third_view_rgb` | smartphone MP4 |
| `T_world_ego[t]` | TCP + `tc_mat` | raw HDF5 camera extrinsic | 每帧 `cam2world_gl` | odom/TF，非 GT |
| `T_world_exo` | calibration | 2025 calibration patch | observer fixed pose | 无 |
| 时间关系 | 同平台时间戳 | camera capture time | 同一仿真 step | 未公开 offset，覆盖区间也不同 |
| 可否定量 | 是，运动学参考 | 是，运动学参考 | 是，仿真精确 GT | 否 |

实现时设置六个硬性验收：

1. **时间同步**：记录每个配对帧的 `Δt` 分布；对 10 Hz 数据建议 `|Δt| ≤ 50 ms`。超阈值帧应剔除，不要静默 nearest-neighbor。
2. **位姿对象**：区分 `head rigid body`、`camera optical frame`、`base` 和 `EEF`。若只给头 rigid body，必须显式乘固定 `T_head_camera`。
3. **跨相机闭环**：必须存在 `T_world_exo`。只有 `T_world_ego` 而没有 exo 外参时，不能把轨迹画进 exo 图像并称为 GT。
4. **变换方向**：用标定板、已知机器人关键点或 AprilTag 做像素重投影，确认 `camera-to-world` 与 `world-to-camera` 没有取反。
5. **GT 不泄漏**：GT 只用于既定的尺度/评估阶段；DA3 推理本身不读取 GT。
6. **文案随证据降级**：Oxford 可写 `head-camera mocap/TF GT`；Vernissage 在验证前写 `mocap head pose, exo calibration pending validation`；MuMMER/RH20T/DROID 写 `kinematic reference`；RoboCup 写 `odometry/TF visualization only`。

## 6. Oxford 等待期间的执行方案

### P0-A：并行申请两套数据

| 数据集 | 操作 | 申请时必须问清 |
|---|---|---|
| Vernissage | 发至 `sebastian.wrede@uni-bielefeld.de`，可抄送 `swrede@techfak.uni-bielefeld.de` | 是否包含 NAO RGB、三路 HD、`nao.csv`、同步 offset、HD-to-Vicon、`T_head_camera`；数据许可 |
| MuMMER | 在 [Zenodo 3989642](https://zenodo.org/records/3989642) 点击 Request access；无回复再联系论文通讯作者 `odobez@idiap.ch` | 是否提供 raw bags、joint/TF/URDF、相机内参、Kinect-to-base/world 外参；数据许可 |
| Oxford-IHM | 已提交，继续等待 | 获批后确认 head-camera rigid-body offset、static D435 Vicon pose 与 TF convention |

Vernissage 可直接使用下面的英文模板：

```text
Subject: Research access request for the Vernissage-HUMAVIPS corpus

Dear Dr. Wrede,

I am working on academic research on egocentric camera-pose estimation for
robots. We would like to evaluate a method using synchronized NAO egocentric
video, external HD video, and the Vicon-based NAO head pose from the
Vernissage corpus.

Could you please share the current access procedure? In particular, could you
confirm whether the release includes:
1) NAO video and all three external HD videos;
2) nao.csv and the Vicon/RSB synchronization files;
3) HD-camera-to-Vicon calibration and the head-to-optical-camera transform;
4) the current dataset license and citation requirements?

The data will be stored on access-controlled research machines, used only for
the stated research, and will not be redistributed.

Best regards,
[Name / Institution / Project]
```

MuMMER 的 Zenodo 申请可写：

```text
We request access for academic research on robot egocentric camera-pose
estimation. We need the raw ROS bags containing Pepper head RGB/depth,
the static Kinect stream, robot joint states/TF, camera_info, and timestamps.
We will keep the data access-controlled and will not redistribute it.

Could you also confirm whether Kinect-to-Pepper/world extrinsics, the Pepper
URDF/robot_description, and the data license are included?
```

### P0-B：不等审批，先把 adapter 跑通

1. **真实数据、可定量**：按原计划用 RH20T cfg5 的 320 × 180 RGB、LowDim、Calibration，约 14.6 GB；先选一个 in-hand camera 和一个无遮挡 global camera。
2. **仿真、几何完全确定**：RoboTwin 开 `third_view: true`，保存 2–3 段各 20–30 s 的 wrist ego、observer exo 和所有 camera matrices。先完成时间、坐标和重投影单测。
3. **真实机器人头部、仅看呈现**：如确实想先看 TIAGo 画面，可下 RoboCup Receptionist，但 UI 明确写 `no synchronized exo GT`，不计算误差。
4. **扩规模**：adapter 稳定后再接 DROID raw + 2025 高精度标定子集，不要先下载 8.7 TB 全量。

### P1：任一申请获批后的 30 分钟准入检查

先抽一个 episode，不要立即批量处理：

1. 列出所有视频、pose、calibration 和同步文件；
2. 找 20–30 s 同时包含明显头部旋转、ego 内容变化、exo 可见机器人的区间；
3. 画 `Δt` 直方图并检查丢帧；
4. 将头/相机轨迹变换到 exo frame；
5. 用至少 20 帧已知点重投影做方向与尺度检查；
6. 只有时间和重投影都通过后，才在界面中启用 `GT` 标签。

### P2：申请仍然很慢时的自采逃生路线

如果两周后 Oxford、Vernissage、MuMMER 都没有可用结果，最稳妥的办法不是继续降低“GT”标准，而是自采一小套 20–30 分钟 pilot：

- 固定一台已标定的 exo RGB，相机完整覆盖机器人；
- 在机器人头部放 Vicon/OptiTrack rigid body；没有 mocap 时至少使用刚性多 AprilTag 板；
- 标定 `T_marker_head`、`T_head_camera` 和 `T_world_exo`；
- 所有流进入同一 ROS clock，另做一次 LED/电子 clapper 作为独立同步校验；
- 采平移、yaw、pitch、复合 6DoF 和静止段，各重复 3 次。

数据量不大，但能得到完全匹配当前 demo 契约、许可清楚且误差来源可控的基准。

## 7. 最终决策

- **若“真实机器人头部 + GT + exo”不可妥协**：继续等移动底座的 Oxford-IHM，同时申请底座静止但头部活动的 Vernissage。两者是本轮找到的唯二严格候选。
- **若 MuMMER 作者能提供 Kinect 外参**：它可作为 Pepper 头部的运动学参考数据；否则只做同时间轴的视觉演示。
- **若现在就要真实机器人且要定量**：选 RH20T，接受 wrist-as-ego，并准确写 `kinematic reference`。
- **若现在就要真头部画面**：RoboCup 可直接下载，但只做视觉 demo，不做 GT 结论。
- **若现在就要验证全部几何代码**：用 RoboTwin 生成成对数据；它是仿真 wrist ego，不替代真实实验。
- **若目标是规模与多样性**：等 RH20T adapter 稳定后再接 DROID raw + 2025 calibration subset。

推荐路径如下：

```text
申请线：Oxford-IHM（移动 HSR） ─┐
       Vernissage（静止 NAO） ─┼─> 获批后做严格 head-camera benchmark
       MuMMER（Pepper FK）    ─┘   （先检查 exo 外参）

开发线：RoboTwin 几何单测 -> RH20T 真实 wrist pilot -> DROID 扩规模

展示线：RoboCup TIAGo head / HABIT multi-view
        只展示，不启用 GT 指标
```

核心结论是：**Oxford 有等待成本，但没有一套“公开直下、真实活动机器人头部、同步 exo、外部测量 6DoF、完整标定”可以无损替代它。** 最有价值的新发现是 Vernissage；最务实的过渡仍是 RH20T/RoboTwin 双线。
