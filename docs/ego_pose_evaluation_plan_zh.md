# Ego Video 6DoF 头部姿态 Evaluation 计划

> 版本：2026-07-23
>
> 固定任务：输入头戴单目 RGB 视频，输出逐帧相机 \(SE(3)\)，经固定相机—头部外参转换为头部 \(SE(3)\)，用于机器人仿真与 IK。主赛道允许使用相机标定内参，但不允许使用测试视频的 IMU、深度、设备 VIO/MPS 或未来 GT。
>
> 整份计划只包含三项：模型、数据集、指标。

## 1. 模型

正式主表跑 7 个模型；动态遮挡子表再跑 2 个专项模型。这样同时覆盖最新前馈几何模型、长视频流式模型、优化式视频系统、ego 专项模型和公共 SLAM 锚点。

| 优先级 | 模型与固定版本 | 类型 | 输入、输出与尺度 | 在本评测中的角色 | 运行范围 |
|---|---|---|---|---|---|
| P0 | [DA3NESTED-GIANT-LARGE-1.1 + DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3) | 多视图 geometry foundation model | RGB → W2C 外参、内参、depth；Nested 模型声称 metric；短片用 `use_ray_pose=True`，长片用官方 Streaming | 当前项目主线；检验 DA3 的 metric scale 和分块连续性 | 全部数据 |
| P0 | [VGGT-Ω-1B-512](https://github.com/facebookresearch/vggt-omega) | CVPR 2026 通用前馈几何模型 | RGB frames → extrinsics、intrinsics、depth；不把平移尺度预设为 metric | 最新通用 SOTA；论文覆盖静态和动态场景，项目页为 CVPR 2026 Best Paper Finalist；权重需申请但由自动流程审核 | 短片全跑；长片做无 GT overlap stitching |
| P0 | [LingBot-Map-long](https://github.com/robbyant/lingbot-map) | 2026 流式 3D foundation model | 连续 RGB → camera pose、depth；单目尺度视为 scale-free | 代表因果长视频模型；官方报告约 20 FPS、可处理超过 10k 帧，并公开长视频 checkpoint 和 benchmark | 全部数据，尤其 3 min 长片 |
| P0 | [VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM) | RSS 2026，feed-forward submap + SLAM 后端 | RGB → 全局轨迹和稠密地图；单目尺度视为 scale-free | 检验 foundation model 加全局优化、回环和在线重建是否优于纯前馈 | 全部数据 |
| P0 | [ViPE 1.2.0](https://github.com/nv-tlabs/vipe) | 通用视频几何系统 | raw video → 内参、camera motion、near-metric depth；使用官方 `dav3` pipeline | 动态视频和广角场景强基线；当前代码为 2026-06 的 1.2.0，官方称相对旧版加速 2.7×；因内部也用 DA3 depth，它测的是“DA3 + BA/SLAM 约束”的系统增益 | 全部数据 |
| P0 | [EgoM2P base](https://github.com/ligengen/EgoM2P) | ICCV 2025 ego-specific 前馈模型 | 2 s、16 帧 RGB@8 FPS → 60 poses@30 FPS，C2W、第一帧参考；学习到的平移尺度需验证 | 唯一直接为 egocentric camera tracking 训练且当前代码、权重可访问的主表模型 | 原生 2 s 表 + overlap stitching 后的全数据表 |
| P0 | [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | 学习式 dense SLAM | RGB + 已知内参 → camera trajectory；单目 scale-free | EgoEgo、HaWoR、EgoM2P 等论文的公共锚点，判断新 foundation model 是否真的优于成熟 learned SLAM | 全部数据 |
| P1 | [MegaSaM](https://github.com/mega-sam/mega-sam) | CVPR 2025 动态 casual video 专项 | 动态 RGB → camera parameters + depth；离线，项目页约 0.7 FPS | 测大量人体/物体运动时的 camera-motion 分离能力 | ADT 动态、HOT3D、InCrowd 子集 |
| P1 | [HaWoR camera module](https://github.com/ThunderVVV/HaWoR) | CVPR 2025 ego 手遮挡专项 | 官方推理流程生成 hand mask，再用 masked DROID + Metric3D → world camera trajectory；不得输入 GT hand mask | 检验显式剔除第一视角手部区域和 metric depth 的收益 | ADT manipulation + HOT3D |

主结论使用 P0 七模型。P1 两模型只进入动态遮挡子表，避免它们因专用输入或较高运行成本影响整个评测进度。

所有模型执行以下统一口径：

1. **主输入为 `RGB + 标定 K`**。机器人/头戴相机通常可预先标定，这能把任务聚焦在 pose；模型可以忽略 K 并自行预测，但不得读取 GT pose、GT depth 或 IMU。另开一个小型 `RGB-only/unknown-K` 消融，只比较原生支持自标定的方法。
2. **公平精度输入统一为 10 FPS**，使用相同时间戳和相同官方去畸变图像；模型可按其架构继续内部重采样，但必须记录实际输入帧。流式模型再在原生帧率上跑因果延迟与高频头动子表。
3. 所有输出统一为 `timestamp, T_world_camera, valid, confidence`；W2C 必须先求逆为 C2W。定义 `T_camera_head` 为“head 坐标到 camera 坐标”的固定变换，再统一计算 `T_world_head = T_world_camera @ T_camera_head`。
4. 原生短窗口模型不得使用 GT 拼接。VGGT-Ω 使用 200 帧窗口、20% overlap（官方 A100 测量约 20.8 GB 峰值，较适合 24 GB 级显卡）；EgoM2P 使用 2 s 窗口、1 s overlap。overlap 只根据两段预测 pose 做 SE(3)/Sim(3) 配准，拼接参数固定后再看 GT。
5. DA3、VGGT-Ω、LingBot-Map、VGGT-SLAM、ViPE、EgoM2P 分别锁定 checkpoint SHA/下载哈希；DROID 锁定官方 `droid.pth`。每个模型独立容器，记录 CUDA、PyTorch、显卡、分辨率、FPS 与峰值显存。
6. EgoM2P 主结果使用未在 ADT post-train 的 base checkpoint；ADT post-trained checkpoint只能列为 in-domain upper bound。其 base pretraining 已包含 HOT3D，因此 HOT3D 结果必须标为 in-domain，不能宣称 zero-shot。其他模型也要公开已知的训练重叠。
7. [ReViV](https://arxiv.org/abs/2607.17790)只列论文参考：作者声称代码开源，但截至本计划日期，其项目页代码链接返回 404。[Map-Mono-Ego](https://arxiv.org/abs/2605.20889)需要预扫描 3D map，项目页数据仍为 “Coming soon”，因此都不进入当前可运行主表。
8. 研究评测前仍要保存许可证快照：DA3 大模型权重为 CC BY-NC 4.0，EgoM2P 权重使用其 Sample Code License，ViPE 的第三方 depth 组件也可能带非商业条款。商业 demo 不能只依据代码仓库本身的 Apache/BSD 许可证。

如果资源不足，最小但仍有解释力的 6 模型组合为：

```text
DA3NESTED-GIANT-LARGE-1.1 + VGGT-Ω + LingBot-Map + ViPE + EgoM2P + DROID-SLAM
```

## 2. 数据集

核心评测固定为 112 个片段、约 65 分钟视频。它不是挑战赛规模，但能够让每种重要 ego failure mode 至少有 4–6 个独立序列实例，并能在合理成本内跑完所有 P0 模型。

| 等级 | 数据集与参考 pose | 固定抽样 | 覆盖的 ego 条件 | 进入哪张表 |
|---|---|---:|---|---|
| A：独立真值、真头戴 RGB | [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)；mocap ground-truth device trajectory，和 MPS 分开提供 | 24 × 30 s = 12 min | 6 个静止/慢动、6 个行走转头、6 个快速转头、6 个手部交互或双人动态片段；apartment/office 均覆盖；只选 Aria 确实由目标人物佩戴且 device association 有效的序列 | **主精度榜** |
| B：真实头戴、设备轨迹参考 | [EgoBody](https://github.com/sanweiliti/EgoBody)；HoloLens PV RGB 的逐帧 `pv2world_transform`，另有 head tracking、同步 Kinect exo 与标定 | 20 × 20 s ≈ 6.7 min；9 条官方 val + 11 条跨场景完整帧补充记录 | `low_motion / moderate_motion / locomotion / fast_turn` 各 5 条，覆盖 11 个场景；每条严格 200 张 PV 输入，且 20 条原始 HoloLens stream 互异；可选同步 exo | **device-reference + robot demo 回归榜**；不得把 HoloLens 自身跟踪写成独立 mocap GT |
| A：独立头显参考、强头动 | [Monado SLAM Dataset](https://arxiv.org/abs/2508.00088)；SteamVR Lighthouse 外部参考 | 16 × 30 s = 8 min | 快速旋转、动态遮挡、低光/过曝、长时 VR gameplay 各 4 段 | **头动/失跟/因果压力榜**；单目灰度复制成 3 通道，不能与普通 RGB 域混平均 |
| A：高精度 RGB、非严格头戴 | [Princeton365](https://princeton365.cs.princeton.edu/)；隐藏标定板 + 360 相机的独立 pose，论文另有 Vicon 验证 | 18 × 30 s = 9 min | object scanning、indoor、outdoor 各 6 段；纹理、反光、尺度和开放空间；每段 GT pose coverage ≥95% | **RGB 几何泛化榜**；不冒充 head-worn |
| B：真实 ego、位姿非完全独立 | [HOT3D Aria](https://facebookresearch.github.io/hot3d/)；Aria RGB，MPS trajectory 经外部系统对齐 | 16 × 20 s ≈ 5.3 min | 四类日常手—物交互各 4 段；由官方 hand GT 投影生成评测 mask，按 `<10%`、`10–30%`、`>30%` 图像占比分桶 | **手遮挡压力榜** |
| B：真实头戴人群、MPS reference | [InCrowd-VI](https://arxiv.org/abs/2411.14358)；Meta Aria，参考轨迹来自离线 MPS | 12 × 30 s = 6 min | 数据均为室内；high/medium/low crowd density 各 4 段，并跨车站、机场、商场、校园和博物馆 | **动态人群 coverage 榜** |
| B/A 混合：城市尺度长程 | [LaMAria](https://www.lamaria.ethz.ch/)；独立测量稀疏控制点 + 联合 BA 稠密 pseudo-GT | 从公开训练/附加部分取 6 × 180 s = 18 min | `1_19, 2_11, 3_17, 3_18, 4_10, 4_11`，覆盖短/中/长与低光；轻量 ASL cam0 为灰度，复制成 3 通道 | **长时漂移/窗口拼接榜**；moving-platform 公开序列只有 sparse control points，不混入逐帧稠密表 |

核心量合计：

```text
短片：106 个，共约 47 分钟
长片：  6 个，共 18 分钟
总计：112 个，共约 65 分钟
```

这里的 65 分钟是下载 profile 的固定输入量。112 条均已固定到官方
sequence/window；EgoBody 还保存 recording frame bounds 与首帧 timestamp，
下载时若发布内容与冻结清单不一致会直接失败。固定清单与下载方法见
[Evaluation 数据下载说明](./eval_dataset_download_zh.md)。

另设一个不进入 6DoF GT 主榜的野外审计：

- 从 [EgoStatic / ORE](https://papers.nips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Paper-Datasets_and_Benchmarks.pdf)固定抽 10 条约 6 min 序列，覆盖 10 种日常活动，仅对主表前三名和 DROID 计算 ORE。它回答“在自然 ego 视频中是否仍然稳定”，不产生 meter/degree 排名。

抽样规则在运行任何模型之前固化为 manifest：

1. `dataset_version + sequence_id + start/end timestamp + camera_id + GT source + clip tag + SHA256` 全部入库，随机种子固定；每个原始 sequence 最多抽一个 clip，确保 112 个样本来自不同记录。
2. 不按任何待测模型的表现挑片。只允许依据 GT 运动强度、官方场景标签、图像亮度/模糊度和 hand-mask 占比分层。
3. 除专门的 stationary bin 外，片段至少满足 `平移 ≥ 0.5 m` 或 `累计旋转 ≥ 45°`，避免 Sim(3) 和尺度评测退化。
4. stationary bin 定义为 GT 线速度 `< 0.02 m/s` 且角速度 `< 2°/s` 连续至少 3 s，用于抖动测试。
5. 只因 GT 缺失、时间戳损坏或视频不可解码而排除片段；方法失跟、崩溃、OOM 和空输出必须保留。
6. A、B、ORE 三种参考等级分别出表，不能计算一个混合总平均。严格
   wearable headline 只来自 ADT；EgoBody、HOT3D、InCrowd 等 device/reference
   结果必须另表展示，Monado与 Princeton365 用于解释 failure mode。
7. 每个数据集按**序列**而不是按帧 bootstrap；同一个原始序列的多个 clip 不得跨统计折，以免把相邻帧当独立样本。

## 3. 指标

所有几何指标最终作用于头部 frame，而不是直接作用于模型各自定义的 camera frame。令 \(T^W_{H,t}\) 为预测头部 C2W，\(T^{W*}_{H,t}\) 为 GT；没有解剖学 head frame 的数据集使用固定 device frame 作为 head proxy，并在结果中注明。

时间同步统一在 10 Hz GT 网格完成：GT translation 线性插值、rotation 用 SLERP；预测 pose 只允许在相邻有效输出间插值，若缺口超过 0.2 s 则整段记为 invalid，不能靠长距离插值提高 coverage。

**对齐设置必须同时保留三套，且不能混列：**

| 设置 | 做法 | 用途 |
|---|---|---|
| `Raw-Metric` | 只用第一帧 \(SE(3)\) 设置世界原点和初始朝向，不允许缩放 | 主测 DA3 Nested、ViPE、HaWoR 等方法的真实单位与长期 metric 能力 |
| `Prefix-Sim3` | 只用开头 2 s GT 估计一次 Sim(3)，随后冻结到整段 | **所有方法的主要公平排名**和 drift 诊断；它仍使用少量 GT，只有部署时确实存在已知 2 s 标定运动/外部参考才可视为可部署 |
| `Oracle-Sim3` | 使用整段 GT 做全轨迹 Sim(3) | 仅诊断几何形状上限；不得被称为部署成绩 |

如果一段视频不足以让前 2 s 的尺度拟合稳定，则标记 `prefix-degenerate`，不偷偷改用整段 GT。还要额外报告 first-frame-only \(SE(3)\) 结果，因为机器人通常可以给定初始头部 pose。

EgoM2P 的原生 2 s 文献对照表只能用前 0.5 s 做 prefix alignment；若用完整 2 s 对齐，则必须放入 `Oracle-Sim3` 列。其 20 s 以上拼接轨迹仍统一使用前 2 s 初始化。

**一级指标：直接衡量 ego 头部 6DoF 是否准确。**

| 指标 | 定义与单位 | 为什么是 ego pose 必需 |
|---|---|---|
| `H-ATE-pos` | 每帧 \(\|\hat p_t-p_t^*\|_2\)，单位 cm；每序列报 median/P95 | 头部平移是否能直接驱动机器人 root/head target |
| `H-ARE-rot` | \(d_R=\cos^{-1}(\mathrm{clamp}((\mathrm{tr}(R_t^{*T}\hat R_t)-1)/2,-1,1))\)，单位 degree；报 median/P95 | 完整 3DoF 头朝向误差，避免使用不可直读的矩阵 Frobenius norm |
| `H-ATE-horizontal / vertical` | 将平移误差按重力方向拆为水平 cm 与高度 cm | 区分走动漂移和坐下、弯腰、头部高度恢复错误 |
| `Tilt Error` | 预测与 GT 头部 up-axis 的夹角，degree | 直接测重力/roll-pitch 对齐；错误会让机器人歪头或身体倾斜 |
| `Heading Drift` | 先对齐 up-axis，再测残余 yaw；报 degree/min | 单目长视频最常见的头朝向漂移 |
| `Metric Scale Error` | 只在 `Raw-Metric` 下计算 `abs(log(L_pred/L_gt))`，另报路径长度比例；stationary/退化段不计算 | 确认“metric”方法的平移是否真能按米传给机器人；Prefix-Sim3 已强制拟合尺度，不能再用它证明 metric |
| `Accurate Coverage@5/5` | `有效输出且 position<5 cm 且 rotation<5°` 的 GT 时间戳比例 | 同时惩罚错误 pose 和无输出，不能通过丢弃困难帧提高均值 |
| `Accurate Coverage@10/10` | 同上，阈值为 10 cm / 10° | 更宽松的 demo 可用率 |

**二级指标：衡量 ego 头动是否被动态地复现。**

- `Head-RPE-trans/rot@0.1, 1, 5, 30 s`：比较相同时间间隔的相对 \(SE(3)\)。0.1 s 测快速转头，1 s 测局部头动，5/30 s 测行走和长漂移；只有序列时长覆盖该 \(\Delta\) 时才计算。
- `Angular-Velocity RMSE` 与 `Linear-Velocity RMSE`：在统一时间网格上由 \(SE(3)\) log map 求速度，分别使用 degree/s 和 cm/s。
- `Fast-Turn Peak Error`：在 GT 角速度 `>90°/s` 的事件中测峰值幅度误差；`Turn Lag` 通过预测和 GT 角速度互相关得到毫秒延迟。
- `Motion Excursion Ratio`：每个连续头动事件中预测/GT 的累计旋转角与位移比，防止轨迹看起来平滑但把真实转头幅度压小。
- 所有上述结果分别对 `stationary / locomotion / fast-turn / hand-occlusion / crowd / low-light` 六类 macro-average，不能让大量普通行走帧淹没困难事件。

**三级指标：衡量机器人最敏感的稳定性与连续性。**

| 指标 | 计算方式 |
|---|---|
| `Stationary Pos Jitter` | stationary bin 内，去掉真实头动后的预测位置标准差，mm |
| `Stationary Rot Jitter` | stationary bin 内相邻预测旋转的 RMS，degree/frame 和 degree/s |
| `Velocity/Acceleration/Jerk Error` | 与 GT 对比头部线/角速度、加速度和 jerk；同时报 P95 |
| `Boundary Excess RPE` | 短窗口或 streaming 边界处的 RPE 减去相邻非边界 RPE，平移 cm、旋转 degree |
| `Discontinuity Rate` | 边界/相邻帧 RPE 超过 `5 cm` 或 `5°` 的次数 / min，而不是直接把真实快速头动算作跳变 |
| `Scale Jump` | 相邻窗口独立估计尺度之比的 `abs(log ratio)`；只针对需要拼接的方法 |

**四级指标：失跟与因果可用性。**

- `Output Coverage`：有有限、时间戳合法 pose 的帧比例；与 Accurate Coverage 分开。
- `Sequence Success`：全段无崩溃，且 Output Coverage ≥95%。
- `Time-to-Failure`：首次连续 1 s 无 pose，或 1 s RPE 超过 50 cm / 30° 的时间。
- `Relocalization Time`：失败后重新连续 2 s 满足 10 cm / 10° 所需时间；从未恢复记为序列剩余时长。
- `Causal Latency`：输入帧到对应 pose 可用的 wall-clock P50/P95；离线方法另报 time-to-first-pose、总 wall time、FPS 和峰值 VRAM，但不与 causal latency 混为一谈。
- 对可随机或依赖初始化的方法运行 3 次，报告成功率和误差方差；确定性大模型只跑 1 次。

**五级指标：直接验证机器人 retargeting。**

在 ADT 与 EgoBody 每个 clip 上，将同一条预测 \(T^W_H\) 输入同一个
IK/仿真控制器，不更改滤波参数。ADT 给出严格 GT 机器人结果；EgoBody
只给 device-reference/demo 回归结果。声称 metric 的模型以 `Raw-Metric`
作为部署主结果；scale-free 模型的 `Prefix-Sim3` 仅作为“存在一次外部
标定时”的 calibrated upper bound；绝不把 `Oracle-Sim3` 输入机器人主榜。

- `IK Success Rate`：求解成功且残差低于控制器阈值的帧比例；
- `Head Target Error`：仿真机器人实际头部与目标头部的 position cm / rotation degree；
- `Joint-limit Violation`、`Self/Scene Collision Rate`；
- `Joint Velocity/Acceleration/Jerk P95`；
- `Control Lag`：预测头动峰值到机器人实际头动峰值的延迟；
- `Emergency Hold Rate`：因 pose 无效、跳变或 IK 失败而触发 hold-last-pose 的次数 / min。

最终主结果不合成一个任意权重的总分。每个模型至少展示以下 headline：

```text
Raw-Metric H-ATE-pos / Metric Scale Error ↓（声称 metric 的模型）
Prefix H-ATE-pos ↓
Prefix H-ARE-rot ↓
Head-RPE-rot@1s ↓
Heading Drift ↓
Stationary Rot Jitter ↓
Accurate Coverage@5cm/5° ↑
IK Success Rate ↑
Causal P95 Latency ↓（仅因果模型）
```

统计以 sequence 为单位：先求每序列指标，再对数据集取 median 和 bootstrap 95% CI；最后只对同级数据做 macro-average。所有失败序列都进入 coverage/survival 统计，不能像部分既有 ego 工作那样在计算精度前删除 SLAM 失败样本。
