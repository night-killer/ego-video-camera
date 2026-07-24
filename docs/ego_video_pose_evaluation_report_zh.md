# Ego Video 相机位姿评测：数据集、Baseline 与落地方案

> 调研日期：2026-07-23
> 目标：从单目 egocentric RGB 视频估计逐帧 6DoF 相机轨迹，并用可解释、可复现的协议评测。
> 范围：人类头戴、手持/移动相机、机器人头部与腕部视角；深度、IMU、外部相机和参考位姿只用于标注或独立扩展赛道，主赛道推理仅使用 RGB。

## 1. 结论先行

1. **真实头戴数据的主基准优先选 Aria Digital Twin（ADT）**。它直接提供 Aria 原始视频和由 motion capture 系统生成的 6DoF device trajectory；官方还明确区分了 `aria_trajectory.csv` 真值与 MPS 估计轨迹。这是本次查漏中最重要的数据集。[ADT 概览](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)；[ADT 数据格式](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)
2. **EgoBody 可以继续用，但不应单独承担“GT 榜”**。其逐帧 `pv2world_transform` 很适合工程开发和域内回归，但公开说明没有把它定义成对 RGB 相机逐帧独立测得的外部 mocap 真值；正式报告应标为“设备轨迹参考”。[EgoBody 官方仓库](https://github.com/sanweiliti/EgoBody)
3. **主榜和压力榜必须拆开**：ADT、TUM RGB-D、Bonn Dynamic、Oxford-IHM/M2DGR 等外部参考进入高置信度榜；HoloAssist、Stera-10M、EgoBody 进入设备轨迹榜；Ego-Exo4D、HD-EPIC、EPIC-Fields 进入视觉重建参考榜。三类结果不能混成一个总排名。
4. **Baseline 不能只有 DA3**。建议最低配置覆盖四种机制：
   `DA3-1.1 + DA3-Streaming`（当前同类主线）、`LingBot-Map`（长视频流式前馈）、`VGGT-SLAM 2.0` 或 `MASt3R-SLAM`（foundation model + SLAM）、`DROID-SLAM/DPV-SLAM + ORB-SLAM3`（成熟学习式与经典几何式对照）。动态短片再加 `MegaSaM` 或 `MonST3R`。
5. **仓库当前 checkpoint 需要更新**。当前 [README](../README_zh.md) 使用 `DA3NESTED-GIANT-LARGE`；DA3 官方已经把原始 Giant/Large/Nested Giant-Large 标为 deprecated，并建议使用修复训练 bug 后的 `-1.1` 权重。因此已有结果可保留作历史基线，正式比较应重跑 `DA3NESTED-GIANT-LARGE-1.1`。[DA3 官方仓库](https://github.com/ByteDance-Seed/Depth-Anything-3)
6. **20 秒、8 FPS、3 个 clip 只够验证代码，不够支撑论文结论**。现有 Easy clip 整段仅平移约 3.02 cm，已经被代码正确判为退化；下一轮必须包含 20–30 秒、2–3 分钟、10 分钟以上三个时长档，并单独统计初始化失败、失跟、重定位和输出覆盖率。
7. **全轨迹 Sim(3) 对齐只能作为 oracle 诊断项**。主结果应报告原始 metric SE(3)（声称 metric 的方法）、固定前缀 Sim(3)（尺度未知的方法）以及不依赖绝对尺度的多时间间隔 RPE。绝不能让整段未来 GT 帮助尺度和坐标系对齐后再把 ATE 当作部署性能。

## 2. 对仓库现有调研与实现的审计

仓库已有两项很扎实的基础工作：

- [机器人 Ego / 第三人称视频 / GT 位姿数据集调研](./robot_ego_exo_gt_pose_dataset_survey_zh.md) 对“机器人头部 ego + 同步 exo + 位姿”这个更严格的需求覆盖较好，Oxford-IHM、Vernissage、MuMMER、RH20T、DROID 等资料仍然有效。
- [EgoBody + DA3 demo](../README_zh.md) 已经实现时间同步、坐标变换、DA3 无 GT 泄漏、Sim(3)/SE(3)、退化检测、ATE/RPE、可视化和测试；`src/ego_video_camera/da3_adapter.py` 也已经接入官方 DA3-Streaming 的分块拼接结果。

当前工作与旧调研的范围不同：这次只要求 **ego RGB + 对应相机轨迹**，不强制有 exo 视频，也不限定机器人头部。因此旧调研不是错误，而是搜索条件过窄。扩展后暴露出五个缺口：

| 缺口 | 当前状态 | 对正式评测的影响 |
|---|---|---|
| 参考轨迹独立性 | EgoBody 的 HoloLens/PV 轨迹被口语化称作 GT | 可能把设备自身跟踪误差当成被测方法误差 |
| 数据域 | 单一室内、人类交互、3 个 clip | 无法覆盖长程、运动模糊、低光、动态遮挡、机器人域 |
| 时长与运动激励 | 20 秒、8 FPS；一段仅 3.02 cm 平移 | ATE/尺度拟合退化，不能测长程漂移和闭环能力 |
| 方法覆盖 | 一个 DA3 checkpoint | 无法判断收益来自模型类别、优化后端还是输入条件 |
| checkpoint 时效 | 原始 DA3 Nested Giant-Large | 不是官方当前推荐权重，横向结论会过时 |

已有三段结果仍有价值：

- Medium 的 full-oracle 旋转误差 median/P95 为 1.46°/4.40°，而 prefix 为 7.26°/9.19°，说明“未来全轨迹对齐”明显乐观。
- Hard 的 oracle 为 5.14°/7.25°，prefix median 达 34.62°，说明短前缀外推会暴露实际失败。
- Easy 被判为 `degenerate` 是正确行为；这类片段应转入“近静止稳定性”测试，而不是继续计算看似精确的尺度对齐 ATE。

## 3. 先定义什么叫“对应位姿”

### 3.1 参考轨迹分级

| 等级 | 参考来源 | 是否可称严格 GT | 用途 |
|---|---|---:|---|
| A1 | Vicon、OptiTrack 等外部 mocap，且有相机/设备刚体外参 | 是 | 真实数据精度主榜 |
| A2 | 仿真器直接输出相机状态 | 是，但只代表合成域 | 几何、坐标、尺度回归 |
| A3 | 外部激光跟踪、RTK/INS 等高精度定位，再经固定外参得到相机位姿 | 有条件 | 机器人/大场景榜，需单列参考系统 |
| B1 | HoloLens tracking、ARKit、设备 VIO、Aria MPS | 否，称 device/VIO reference | 长视频和真实域压力测试 |
| B2 | 机器人正运动学 + 手眼标定 | 否，称 kinematic reference | 腕部/机器人榜 |
| C | COLMAP、MPS closed-loop、离线 SfM/SLAM 恢复 | 否，称 reconstructed reference | 域覆盖、相对比较、人工质检 |

**硬规则**：A、B、C 分榜；合成与真实分榜；不同参考系统的误差不直接求一个总体平均数。

### 3.2 统一输出

所有 dataset adapter 和 method adapter 最终都输出：

```text
timestamp_ns
frame_id
T_world_camera[4,4]      # camera-to-world, meter
K[3,3]
distortion_model + coefficients
valid
confidence               # 可空
```

统一使用 OpenCV 光学相机局部轴 `+X right, +Y down, +Z forward`。若方法输出 world-to-camera，先求逆；例如 COLMAP 文档明确把 image pose 定义为 world-to-camera，DA3 也必须按其 API/保存格式核对后再转换。[COLMAP 输出格式](https://colmap.readthedocs.io/en/latest/format.html)；[DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)

设备轨迹通常是 `T_world_device`，不能直接当 RGB 相机位姿。应使用该时刻的标定：

```text
T_world_rgb(t) = T_world_device(t) @ T_device_rgb(t)
```

Project Aria 还可能提供随时间变化的 online calibration；需要记录到底用了 static 还是 online calibration。[Aria 坐标系说明](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention)

## 4. 数据集查漏补缺

### 4.1 第一优先级：可进入高置信度精度榜

| 数据集 | Ego 形态与规模 | 位姿来源 / 等级 | 优点 | 限制与建议 |
|---|---|---|---|---|
| **Aria Digital Twin (ADT)** | 真实 Aria 头戴；当前 V2 文档列 236 sequences，RGB 30 FPS | mocap 生成 `aria_trajectory.csv`，A1 | 最贴近目标；原始 VRS、标定、时间戳、GT 与 MPS 同时存在，可直接量化 MPS 偏差 | 仅 apartment + single-room office 两个空间；不能单独代表开放世界。作为**严格头戴主榜第一选择** |
| **Bonn RGB-D Dynamic** | 移动 RGB-D，相机周围有大量运动的人；24 dynamic + 2 static | OptiTrack，A1 | 动态前景、遮挡、搬箱子/气球等很适合检验 ego 动态场景鲁棒性；TUM 格式 | 不是头戴；主赛道只喂 RGB。作为动态压力子榜。[官方页](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) |
| **TUM RGB-D** | 手持/移动 RGB-D；640×480 @ 30 Hz | 8-camera mocap @ 100 Hz，A1 | 生态成熟、下载小、评测工具标准化，适合回归与排查坐标错误 | 不是 wearable，画质与运动模式偏旧；只做几何 sanity，不当 headline。[官方页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) |
| **Oxford-IHM** | Toyota HSR 头部移动 RGB-D + static RGB-D，约 60 min | mocap 对 sensors/obstacles 100 Hz，A1 | 严格机器人头部 ego，且场景含运动的人 | 数据较小，访问与 rosbag 解析需预审；优先纳入机器人子榜。[官方页](https://ori.ox.ac.uk/publications/datasets/oxford-indoor-human-motion-dataset-2024) |
| **M2DGR** | 地面机器人，多路鱼眼/前视 RGB；36 sequences，室内外约 1.2 TB | Vicon、Leica tracker、RTK/INS，A1/A3 | 室内、室外、暗室、电梯、长距离；同步和标定完整 | 不是人类佩戴式；不同 sequence 的参考系统精度不同，需分组报告。[项目页](https://sjtu-visys.github.io/M2DGR/) |
| **EuRoC MAV** | 飞行机器人双目灰度 | Vicon/Leica，A1/A3 | 快速运动、低纹理、工业环境，适合经典 VO/SLAM 回归 | 非 RGB、非头戴；仅作算法几何 sanity。[官方页](https://projects.asl.ethz.ch/datasets/euroc-mav/) |
| **TUM VI / TUM-VIE** | 手持或头戴鱼眼灰度/事件，快速运动 | mocap 只覆盖起止 mocap room 段，A1（局部） | 快速旋转、鱼眼、运动模糊很有诊断价值 | **不能对整条长走廊轨迹算全局 GT ATE**；只评有 GT 的片段或首尾漂移。[TUM VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)；[TUM-VIE 论文](https://arxiv.org/abs/2108.07329) |

ADT 官方明确说明 `aria_trajectory.csv` 虽与 MPS trajectory 使用相同结构，却由 ADT ground-truth system 而不是 MPS 生成；这是选择它而不是普通 Aria MPS 数据做主榜的核心原因。[ADT 数据格式](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)

### 4.2 精确合成真值：用于回归，不替代真实榜

| 数据集 | 内容 | 位姿 / 等级 | 建议 |
|---|---|---|---|
| **Aria Synthetic Environments (ASE)** | 100K 室内场景、模拟 Aria RGB 鱼眼，约 2 min/trajectory | 仿真逐帧 GT，10 FPS，A2 | 与 ADT 相机生态接近；选小而固定的 holdout 做坐标、鱼眼、尺度单测。[格式](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format) |
| **TartanAir V2** | 大量合成室内外、动态/天气、多相机模型 | raw camera pose，图像 10 Hz 且完美同步，A2 | 适合大运动和长程；但 DROID-SLAM 等方法明确用 TartanAir 训练，必须做训练泄漏审计，不能作为唯一结论。[官方文档](https://tartanair.org/modalities.html) |
| **MPI Sintel** | 动画电影渲染的动态视频，包含快速相机/物体运动、模糊和大遮挡 | 官方提供 GT depth 与 camera motion，A2 | 动态相机估计论文常用，适合与 MegaSaM、MonST3R、LEAP-VO 的公开协议对齐；不是 wearable，也不代表真实传感器噪声。[官方数据页](https://sintel.is.tue.mpg.de/downloads) |

### 4.3 第二优先级：有逐帧位姿，但属于设备/运动学参考

| 数据集 | Ego 视频与位姿 | 规模/特点 | 定位 |
|---|---|---|---|
| **EgoBody** | HoloLens PV RGB；`pv.txt` 每帧含 timestamp、fx/fy、`pv2world_transform`，并有 Kinect/HoloLens 标定 | 125 sequences，199,111 ego RGB frames，同步多 Kinect | B1；保留现有工程集，结果标题改为 “HoloLens/PV reference” |
| **HoloAssist** | HoloLens RGB、head pose、相机标定；`Head_sync.txt`、`Pose_sync.txt` 等同步文件 | 约 166 h，真实交互、手部和动态遮挡丰富 | B1；适合规模和域泛化，不进入独立 GT 主榜。[项目页](https://holoassist.github.io/)；[格式](https://holoassist.github.io/data_links/README.html) |
| **Stera-10M / MobileEgo Anywhere** | 头戴 iPhone Pro RGB 1280×720 @ 15 FPS；逐帧 ARKit 6DoF、LiDAR depth、IMU | 当前数据卡列约 10M frames、200 h、584 sessions，最长约 104 min | B1；目前最有吸引力的**长时头戴压力集**，但 ARKit 仍是被测视觉/惯性系统，不是独立 GT。[数据卡](https://huggingface.co/datasets/fpvlabs/stera-10m) |
| **RH20T** | 1–2 个 in-hand camera；gripper Cartesian pose 100 Hz + 内外参可推 camera pose | 110K+ robot manipulation sequences，多机器人、多视角 | B2；若“腕部视角也算 ego”则很有价值，否则只进机器人腕部子榜。[项目页](https://rh20t.github.io/) |

### 4.4 有轨迹但不适合作为主 GT

| 数据集 | 参考来源 | 为什么不进严格 GT 主榜 | 仍然适合什么 |
|---|---|---|---|
| **Ego-Exo4D** | Aria MPS `closed_loop_trajectory.csv`，1 kHz；exo 有 static calibrations | closed-loop trajectory 是 mapping/VIO/BA 的估计输出，不是独立传感器真值 | 大规模、多城市、同步 ego/exo 域压力和定性分析。[官方 MPS 文档](https://docs.ego-exo4d-data.org/data/mps/) |
| **HD-EPIC** | Aria 每帧 device-to-world + calibration，场景也使用 MPS/multi-video SLAM | 参考仍来自 MPS/视觉重建 | 41 h 厨房长时交互、强手部遮挡、真实任务压力。[项目页](https://hd-epic.github.io/site/) |
| **EPIC-Fields** | 对 EPIC-KITCHENS 用 COLMAP/photogrammetry 注册，提供 intrinsics/extrinsics | 用 SfM 生成的 pose 评 SfM/SLAM 方法会产生同源偏好 | 超大规模 in-the-wild 覆盖率与失败分析；不报“绝对 GT 精度”。[项目页](https://epic-kitchens.github.io/epic-fields/) |
| **HOT3D** | 头戴 Aria/Quest 多目；手/物体有 mocap 高质量标注，Aria 相机轨迹主要依赖 MPS/对齐 | “手/物体 mocap GT”不等于“RGB 相机轨迹独立 GT” | 动态手物遮挡、短程 HOI 压力测试。[官方页](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d) |

Aria MPS 文档把 closed-loop trajectory 定义为 mapping process 的 bundle-adjusted pose estimation，并指出 loop closure 甚至可能使局部短时精度变差。因此 MPS 轨迹应该叫 reference，而不是无条件叫 GT。[MPS trajectory 文档](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory)

### 4.5 暂不纳入

- **EgoXtreme（CVPR 2026）** 很适合极端照明、烟雾和运动模糊，但当前公开数据卡主要是 BOP 格式的相机内参与**物体相对相机的 6D pose**；没有清楚发布可直接用于本任务的逐帧 world-camera trajectory，不能因为论文使用 OptiTrack 就自动把它当相机轨迹集。[数据卡](https://huggingface.co/datasets/taegyoun88/egoxtreme)
- 只有头部姿态、IMU orientation、GPS 点或机器人 base odometry，而缺少相机完整 6DoF 与固定外参的数据，不进入本任务。
- 只有稀疏关键帧 pose 的数据可以进入“稀疏定位”赛道，但不能插值伪装成逐帧轨迹主榜。

## 5. 推荐的数据组合

### 5.1 最小可发表组合

建议最终报告至少有四个互补分区：

| 分区 | 推荐数据 | 回答的问题 |
|---|---|---|
| 严格头戴主榜 | ADT real | 在真正的 wearable RGB 上，姿态/轨迹有多准？ |
| 动态真实压力 | Bonn Dynamic（RGB only）+ ADT 的双人/动态物体片段 | 人、手和物体运动是否让相机运动估计崩溃？ |
| 经典几何回归 | TUM RGB-D（RGB only） | 实现、坐标和指标是否与社区标准一致？ |
| 长时与域外 | Stera-10M + HoloAssist/EgoBody（B 榜） | 2–100 分钟视频的存活率、漂移和资源成本如何？ |
| 机器人扩展 | Oxford-IHM 或 M2DGR；腕部可加 RH20T | 是否能迁移到机器人头部/腕部 ego？ |
| 精确合成 | ASE + 少量 TartanAir holdout | 尺度、鱼眼、极端运动的受控诊断 |

如果时间有限，第一批只做：**ADT + Bonn + TUM + 现有 EgoBody**。如果拿不到 ADT 或解析尚未完成，不能把 EgoBody 政名为严格 GT；应明确说这是 device-reference pilot。

### 5.2 建议起步配额

- P0 持续集成：TUM 4 条 + Bonn 4 条 + ADT 4 条 + EgoBody 现有 3 条。
- P1 正式短/中程：每个真实分区至少 15–20 条，按 sequence 而非 frame 加权。
- P2 长程：至少 10 条 10 min+，其中 3 条含 loop、3 条无 loop、3 条高动态/遮挡；Stera-10M 是优先来源。
- 每个数据集固定 manifest 和 checksum；不要运行时随机抽 clip。

## 6. Baseline 调研

### 6.1 第一层：与 DA3 最接近的前馈/基础模型

| 方法 | 内参 | 运行形态 | 长视频/动态 | 工程状态与建议 |
|---|---|---|---|---|
| **DA3NESTED-GIANT-LARGE-1.1** | 可未知 | 多帧 batch；输出 pose/intrinsics/depth | 短中片段；长视频需 streaming | **必做**。替换当前 deprecated 权重，并保留 checkpoint hash。[官方仓库](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| **DA3-Streaming** | 可未知 | chunk + overlap + alignment；近流式 | 官方给出数千帧实验，约 11.5–28.3 GB VRAM 取决于 chunk/分辨率 | **必做长程版本**。仓库已有 adapter；明确 chunk、overlap、loop setting。[官方说明](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) |
| **LingBot-Map** | 可未知 | streaming KV cache；也有 windowed mode | 官方展示约 25K frames；>3000 frames 推荐 windowed | **强烈建议必做**，它是最直接的长时新基线。记录 initial scale frames、balanced/long checkpoint、keyframe interval 和 window reset，并据实际读帧范围判定是否严格 causal。[官方仓库](https://github.com/Robbyant/lingbot-map) |
| **VGGT-Ω (Omega)** | 可未知 | 多帧前馈 batch | static + dynamic；显存随帧数上升，官方 A100 表中 100/200/500 帧约 13.37/20.82/43.15 GB | 作为当前新一代短/中 clip 强基线；checkpoint 访问受控且许可需核对。[官方仓库](https://github.com/facebookresearch/vggt-omega) |
| **VGGT** | 可未知 | 多帧前馈 batch | 数百视图，长视频需分块/SLAM 外壳 | 若已有环境可保留历史对照；新实验优先 Omega。[官方仓库](https://github.com/facebookresearch/vggt) |
| **VGGT-Long / Map-Long / Pi-Long** | 可未知或 metric 条件 | chunk、overlap alignment、loop closure | 可扩到公里级，但官方提示约 300/4500 frames 可产生约 5/50 GB 临时文件，且 motion blur 是明显风险 | 适合分离“长视频 wrapper”与“基础模型”贡献；DA3-Streaming 已沿用这一技术路线，故放扩展而非最小集合。[官方仓库](https://github.com/DengKaiCQ/VGGT-Long) |
| **MapAnything** | 可未知，也可条件输入 K/pose/depth | feed-forward metric 3D | 更像通用多视图模型，不是长时 SLAM | 可选 metric-scale 基线；只用 image-only 配置，防止把 GT K/pose 混入未知内参赛道。[项目页](https://map-anything.github.io/) |
| **Pi3 / Pi3X** | 可未知 | permutation-equivariant batch | Pi3X 支持近似 metric scale | 可选新架构消融，不是最小集合必需项。[官方仓库](https://github.com/yyfz/Pi3) |

注意：DA3-Streaming 官方明确说它“不是 SLAM system”；它与有闭环/全局图优化的方法应该同时报告结果，但运行类别必须标注，不能只按最终 ATE 宣称实时 SLAM 优劣。

### 6.2 第二层：可运行的 SLAM / VO 主干

| 方法 | 输入条件 | 优势 | 局限 | 推荐级别 |
|---|---|---|---|---|
| **VGGT-SLAM 2.0** | monocular RGB，可走 uncalibrated | submap/factor-graph + foundation prior，适合比纯 batch 更长的序列；当前仓库为 2.0 | 官方 real-time live code 仍列 TODO；以离线发布代码实测为准 | 核心候选。[仓库](https://github.com/MIT-SPARK/VGGT-SLAM) |
| **MASt3R-SLAM** | RGB；官方支持有/无 calibration | 实时 dense SLAM，可直接读 MP4/image folder；成熟度较好 | 安装和 checkpoint 依赖较重 | 核心候选。[仓库](https://github.com/rmurai0610/MASt3R-SLAM) |
| **DROID-SLAM** | 需要 calibration | 强而成熟的 learned BA；mono/stereo/RGB-D | 官方 demo 至少约 11 GB GPU；异步模式非确定 | **已知内参核心基线**，至少重复 3 次。[仓库](https://github.com/princeton-vl/DROID-SLAM) |
| **DPVO / DPV-SLAM** | 需要 calibration | 快速 patch VO；DPV-SLAM 带 loop closure，直接输出 TUM trajectory | 训练集包含 TartanAir，相关数据必须标泄漏 | 可与 DROID 二选一或都做。[仓库](https://github.com/princeton-vl/DPVO) |
| **ORB-SLAM3 monocular** | 需要 K/畸变；支持 pinhole/fisheye | 最重要的经典几何对照，实时、多地图、闭环 | 低纹理、模糊、强动态可能频繁初始化/失跟 | **必做经典基线**。[仓库](https://github.com/UZ-SLAMLab/ORB_SLAM3) |
| **LEAP-VO** | 需要 K | 长期点跟踪，对 occlusion/dynamic/low texture 有针对性 | 不是完整大规模闭环系统 | 动态场景已知 K 子榜优先。[仓库](https://github.com/wrchen530/leapvo) |
| **COLMAP sequential** | 可估 K，最好提供 camera model | 经典离线 SfM + BA，可作高计算预算控制组 | 非在线；动态 ego、模糊和长视频可能注册不全 | 离线 oracle/control，必须报 registered-frame coverage。[教程](https://colmap.github.io/tutorial.html) |

### 6.3 动态视频专项

| 方法 | 特点 | 资源/成熟度 | 建议 |
|---|---|---|---|
| **MegaSaM** | 面向 casual dynamic videos，联合 camera tracking 与 video depth，官方已发布代码和权重依赖 | 多阶段优化，速度和配置不宜与 causal 方法直接混比 | 动态短/中片段的核心专项基线。[仓库](https://github.com/mega-sam/mega-sam) |
| **MonST3R** | 输出动态点云、逐帧 camera pose 和 intrinsics | 官方说明 65 帧 16:9 全优化约 33 GB，non-batchified 约 23 GB；也有 window-wise/real-time 变体 | 只在短动态 clip 做，完整优化和实时模式分开记录。[仓库](https://github.com/Junyi42/monst3r) |
| **CUT3R** | online recurrent state、动态场景、逐帧 camera | 研究型前馈状态模型 | 作为附加消融，不取代成熟 SLAM 基线。[项目页](https://cut3r.github.io/) |
| **D4RT** | CVPR 2026，联合 depth、tracking、full camera parameters | 截至调研日官方项目页未提供可复现实验所需的正式代码/权重入口 | watchlist，不放进“已运行 baseline”表。[项目页](https://d4rt-paper.github.io/) |

### 6.4 推荐的最小 baseline 套件

算力和人力受限时，优先按以下顺序落地：

1. `DA3NESTED-GIANT-LARGE-1.1` + 官方 DA3-Streaming。
2. `LingBot-Map long`：流式/窗口模式。
3. `VGGT-SLAM 2.0` 或 `MASt3R-SLAM`：至少一个 foundation-SLAM。
4. `DROID-SLAM` 或 `DPV-SLAM`：已知内参 learned SLAM。
5. `ORB-SLAM3 monocular`：已知内参经典基线。
6. `MegaSaM`：动态短片；资源允许再加 MonST3R。
7. `VGGT-Ω`：短/中 clip 的高容量 batch 模型。
8. `COLMAP sequential`：离线控制组。

这套组合比“把所有新模型都跑一遍”更有解释力：每一项代表不同机制，而且代码/权重当前可获得。

## 7. 公平的评测协议

### 7.1 三个轴、最多十二种组合，不做混榜

| 轴 | 赛道 |
|---|---|
| 内参 | U：RGB only / unknown intrinsics；K：RGB + 官方 intrinsics/distortion |
| 时序 | ON：严格 causal/online；OFF：offline/batch，可访问未来帧 |
| 参考 | A：外部/仿真 GT；B：device/kinematic reference；C：vision-reconstructed reference |

结果名称例如 `U-ON-A1`、`K-OFF-A1`。方法若内部自估 K，属于 U；若使用数据集 K 初始化、固定或优化，都属于 K。batch 模型用滑窗拼接后仍属于 OFF，除非严格证明输出某帧时从未读未来帧。默认 DA3-Streaming 使用带 overlap 的完整 chunk 和事后对齐，应归入 OFF；只有改成每个输出时刻严格不访问未来图像的版本才能进入 ON。

### 7.2 公共输入

- 所有 RGB-only 方法获得完全相同的 frame manifest、时间戳、方向和 crop。
- 公共频率主榜建议 10 Hz；另报 native FPS 榜。不能只在 8 FPS 抽帧后推断原生视频性能。
- 对 Aria/鱼眼同时建立：
  - `rectified-common`：统一投影成共同 pinhole FoV，所有方法可参加；
  - `raw-native`：只允许原生支持对应鱼眼模型的方法参加。
- 数据集提供的 depth、IMU、exo 视频、body/object annotations 均不可进入 RGB 主赛道推理。
- rolling shutter 数据要按曝光时间关联轨迹，并在局限中说明把整帧近似成单 pose；不能静默当 global shutter。

### 7.3 时间同步与插值

1. 以图像 capture timestamp 为主键，不以解码后的均匀帧号替代真实时间。
2. GT 在 SE(3) 上插值：translation 线性插值，rotation 用 SLERP。
3. 设定数据集级最大时间差，例如不超过半个图像周期或经校验的固定阈值；超阈值帧丢弃并计入 coverage。
4. 方法只输出 keyframe pose 时，主结果只在实际输出帧上评分并报告 coverage；插值结果只能作为独立 ablation。
5. 相机刚体外参、时间偏移和坐标转换必须用重投影/静态标定 sanity check，而不是看轨迹“像不像”。

### 7.4 对齐协议

每条轨迹同时输出三组结果：

1. **Raw metric / SE(3)**：对声称 metric scale 的方法，只允许用首帧或固定刚体 SE(3) 对齐，不缩放。报告 scale ratio 与随时间 scale drift。
2. **Prefix Sim(3)**：尺度未知方法用最初固定 5 s 和 10 s 两个版本估计一次 Sim(3)，随后冻结。若前缀平移/空间秩不足，标记 `scale_unobservable`，不能自动扩展到整段 30% 后仍当主结果。
3. **Full Sim(3) oracle**：使用整条 GT 拟合，仅回答“轨迹形状在最佳事后对齐下如何”，表格必须带 `oracle`，不能作为主排名。

现有实现允许 prefix 从 3 s 回退到 5 s，再扩展到 clip 前 30%。这对 demo 可用，但正式评测应改成固定 5 s/10 s 报告；否则不同方法或 clip 实际偷看的 GT 时长不同。

### 7.5 指标

每个 sequence 至少报告：

- **输出与鲁棒性**：pose coverage、首次初始化时间、最长连续跟踪时长、lost 次数、relocalization/reset 次数、整段成功率。
- **绝对轨迹**：ATE RMSE / median / P95，位置误差时间曲线。
- **旋转**：逐帧 geodesic rotation error mean / median / P95。
- **局部漂移**：RPE translation/rotation @ 1 s、5 s、10 s；另可加固定路程 1 m/5 m。
- **长程**：final drift、每分钟漂移、loop closure 前后误差、滑窗 scale ratio。
- **尺度**：prefix scale、raw metric scale error、不同时间窗的 scale drift。
- **效率**：端到端 FPS、首帧/首个 pose latency、peak GPU VRAM、peak CPU RAM、临时磁盘、模型加载时间与纯推理时间。
- **置信度**：若模型给 confidence，报告 error-confidence Spearman、risk-coverage curve 和 AUSE，而不只做可视化。

聚合时先算每条 sequence 的指标，再取 sequence-level median/mean 与 bootstrap 95% CI；不要把所有 frame 混在一起，否则长视频会支配结果。对 DROID/ORB 等可能非确定的方法至少跑 3 个 seed/run，并报告成功次数。

### 7.6 退化与失败不是缺失值

- GT translation span 小于阈值、纯旋转或空间秩不足：进入 `low-excitation / zero-motion stability` 分区。
- 方法未初始化或中途崩溃：coverage 和 success 记 0/失败，不能从 ATE 均值中静默删除。
- 近静止片段应评“预测抖动、虚假平移、虚假旋转”，而不是做 Sim(3) 后的 ATE。
- COLMAP 只注册部分帧、模型只输出 keyframes、OOM、超时都必须进入汇总表。

### 7.7 Clip taxonomy

manifest 至少标注以下属性，便于按失败模式切片：

- low/high translation、pure/rapid rotation、回环/无回环；
- 运动模糊、rolling shutter、低光、低纹理、反光；
- 手/物体占据大面积、多人动态前景、完全遮挡；
- 室内/室外、头戴/手持/机器人头/腕；
- 20–30 s、2–3 min、10 min+。

## 8. 训练泄漏与版本控制

每个 `(method, checkpoint, dataset)` 需要一张 model card：

```yaml
method:
checkpoint:
checkpoint_sha256:
code_commit:
license:
training_datasets:
known_overlap_with_eval:
input_intrinsics: unknown | provided
causal: true | false
resolution:
frame_rate:
window:
overlap:
loop_closure:
seed:
```

尤其需要注意：

- DROID-SLAM 官方说明使用 TartanAir 训练；TartanAir 结果不能称 zero-shot。
- 3D foundation models常使用大量公开重建数据；若官方没有列清训练集，应标 `overlap_unknown`，不能写“无泄漏”。
- DA3 原始和 `-1.1` 是不同 checkpoint；当前 repo 的历史结果与重跑结果不能共用同一方法名。
- LingBot-Map 官方提示 KV cache 超过训练的 320 views 后性能下降，长距离可能 pose collapse，需要 window reset；这些设置是算法配置，必须写入结果。
- 2026 年的新仓库更新较快，应 pin commit。LingBot-Map、VGGT-SLAM 2.0、VGGT-Ω 等都不能只写 `main`。

## 9. 对当前代码库的落地改造

现有代码不用推倒重来。建议按以下顺序扩展：

### P0：两周内可形成可信 pilot

1. 下载并登记 `DA3NESTED-GIANT-LARGE-1.1`，用现有 `da3_adapter.py` 重跑 3 个 EgoBody clip。
2. 新增通用 `DatasetSequence` schema，把 EgoBody adapter 从主 pipeline 中解耦。
3. 先加 TUM RGB-D、Bonn、ADT 三个 adapter；每个 adapter 写：
   - pose convention inverse test；
   - timestamp association test；
   - `T_world_device @ T_device_camera` test；
   - 3D 点投影到 RGB 的可视化验收。
4. 新增固定 5 s / 10 s prefix alignment；保留当前 full oracle 但从默认主表移到 diagnostic。
5. 指标补齐 RPE@1/5/10s、coverage、失败状态、runtime、VRAM、sequence-level bootstrap。

### P1：形成 baseline 矩阵

1. 定义统一 method output（TUM trajectory + metadata JSON 即可）。
2. 先接 DA3-1.1、LingBot-Map、DROID-SLAM、ORB-SLAM3、VGGT-SLAM 2.0/MASt3R-SLAM。
3. 同一 manifest 同时生成：
   - unknown-K rectified input；
   - known-K rectified input；
   - raw fisheye input。
4. 用容器或独立环境锁住 CUDA/PyTorch 依赖，主进程只消费标准化 trajectory。

### P2：论文级长时与动态实验

1. 加 Stera-10M/HoloAssist 长视频；加 MegaSaM/MonST3R 动态专项。
2. 建 10 min+ 任务队列、超时/断点恢复和 GPU memory telemetry。
3. 输出按 dataset、GT tier、camera type、duration、failure tag 分层的表和图。
4. 人工抽查每个失败类别，不用单一 ATE 数字替代 failure analysis。

## 10. 最终建议的论文表格

主文最多放三张表：

1. **ADT A1 主榜**：`U-ON`、`U-OFF`、`K-ON` 分块，列 coverage、prefix-ATE、RPE@1/5s、rotation、FPS/VRAM。
2. **动态与长时榜**：Bonn A1 + Stera/HoloAssist B1，突出 survival、final drift、reset、每分钟漂移。
3. **域泛化与失败分组**：wearable/robot、blur/dynamic/low-light/loop，报 sequence median 与 95% CI。

附录再放：

- full Sim(3) oracle；
- TUM/ASE sanity；
- raw fisheye vs rectified；
- frame-rate、window、prefix、checkpoint 消融；
- B/C reference 数据结果；
- 每条 sequence 的完整结果和失败日志。

## 11. 一页执行清单

- [ ] 严格 GT、设备参考、视觉重建参考已分榜。
- [ ] ADT 已进入真实 wearable 主榜。
- [ ] EgoBody 结果不再无条件标成独立 GT。
- [ ] DA3 已换 `-1.1`，旧结果保留但单独命名。
- [ ] 至少有 foundation feed-forward、foundation-SLAM、learned SLAM、classic SLAM 四类 baseline。
- [ ] unknown-K / known-K、causal / offline 已分赛道。
- [ ] 10 Hz 公共帧与 native FPS 都有结果。
- [ ] 5 s / 10 s prefix 固定，不按方法或 clip 偷看更长 GT。
- [ ] full Sim(3) 只叫 oracle。
- [ ] coverage、初始化失败、失跟、OOM、超时进入总表。
- [ ] 近静止/纯旋转片段单独评稳定性。
- [ ] 结果按 sequence 聚合，并有 bootstrap CI。
- [ ] 记录 code commit、checkpoint hash、训练数据重合、许可、硬件和峰值资源。

## 12. 推荐决策

如果现在就启动实现，最稳妥的路线是：

> **ADT 做严格 wearable 主榜；Bonn/TUM 做动态与几何回归；EgoBody 保留作现有工程回归；Stera-10M 做长时 device-reference 压力榜。Baseline 先跑 DA3-1.1/Streaming、LingBot-Map、VGGT-SLAM 2.0 或 MASt3R-SLAM、DROID/DPV-SLAM、ORB-SLAM3，再给动态片段加 MegaSaM。**

这个组合已经覆盖本任务最关键的四个变量：真实头戴、独立位姿参考、动态遮挡、长程漂移；同时不会把不同输入条件、不同参考质量和不同时间因果性混为一个排行榜。
