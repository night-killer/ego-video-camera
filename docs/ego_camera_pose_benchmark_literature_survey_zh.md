# Ego 视角 6DoF 相机/头部姿态 Benchmark 与 Evaluation 文献调研

> 调研日期：2026-07-23
>
> 目标任务：单目 egocentric RGB 视频 → 逐帧 6DoF 相机轨迹 → 固定相机—头部外参 → 机器人头部目标姿态/逆运动学。
>
> 本文重点：已有工作是否真的评测了这条链路、它们用了什么数据和指标，以及应该怎样建立一个不会被对齐方式“美化”的 evaluation。

## 1. 结论

### 1.1 DA3 能不能做这件事？

**能，但“能输出位姿”和“位姿已被证明适合机器人使用”是两个不同结论。**

[Depth Anything 3（DA3）](https://arxiv.org/abs/2511.10647)可以为一段图像输出逐帧 world-to-camera 外参。若第 \(i\) 帧输出

\[
T^{C_i}_{W}=
\begin{bmatrix}
R_i&t_i\\
0&1
\end{bmatrix},
\]

则 camera-to-world 为

\[
T^{W}_{C_i}=(T^{C_i}_{W})^{-1}=
\begin{bmatrix}
R_i^\top&-R_i^\top t_i\\
0&1
\end{bmatrix}.
\]

假设 \(T^{H}_{C}\) 是相机坐标到人头坐标的固定标定，则

\[
T^{W}_{H_i}=T^{W}_{C_i}(T^{H}_{C})^{-1}.
\]

只要机器人侧能够把 \(T^{W}_{H_i}\) 反解为关节目标，DA3、VGGT、ViPE、DROID-SLAM 一类输出相机 \(SE(3)\) 轨迹的方法都可以接入。还需要一次性处理：

- 相机与头部/机器人目标坐标系的刚性外参；
- 世界坐标原点和初始朝向；
- 单目方法的尺度；
- 时间戳、坐标轴方向和单位；
- 失跟、抖动、窗口边界跳变以及机器人关节约束。

不过，[DA3 官方 pose benchmark](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md)主要评估所有帧对之间的**相对旋转角误差和归一化平移方向角误差**。实现中计算 3°/5°/15°/30° 阈值下的 AUC，官方汇总表主要展示 AUC@3° 和 AUC@30°。它不直接评估：

- 平移长度和 metric scale；
- 连续轨迹的 ATE、长时漂移与闭环；
- 分块推理后的跨窗口连续性；
- 静止抖动、速度和 jerk；
- 失跟率、有效输出覆盖率；
- 因果延迟与机器人 IK 成功率。

所以 DA3 是合理 baseline，但不能用 DA3 官方 pose AUC 直接证明其输出已经是“机器人可用的头部轨迹”。

对当前项目，建议用官方修复训练问题后的
`DA3NESTED-GIANT-LARGE-1.1` 做 metric 主版本，并增加
`DA3-GIANT-1.1` 作为 scale-free 消融。官方 [API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)将外参明确为 OpenCV/Colmap 格式的 W2C；`use_ray_pose=True` 通常比默认 camera head 更准，但更慢。Nested 模型被定位为 metric reconstruction，仍应在 ADT/KinPoly 上单独验证其绝对尺度。上述大模型权重为 CC BY-NC 4.0，若 demo 涉及商业用途还需另行核对授权。

### 1.2 有没有直接相关的 benchmark/evaluation 论文？

有，而且可以分成三类：

| 类别 | 最相关工作 | 它真正评测了什么 |
|---|---|---|
| 直接从 ego RGB 估计头部/相机轨迹 | EgoEgo、EgoM2P、ReViV、HaWoR | 6DoF 头部或相机轨迹，通常是短片段或特定场景 |
| 专门评测 ego 相机轨迹 | ORE/EgoStatic、LaMAria、Monado、InCrowd-VI | 野外代理误差、长程 SLAM、覆盖率、运行时间和失跟 |
| 可复用的带位姿数据/服务器 | ADT、Kin-Poly、HOT3D、Princeton365 | 严格或近似的相机轨迹参考，但不一定原生提供 RGB-only 方法排行榜 |

最直接的五篇工作是：

1. [EgoEgo（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Ego-Body_Pose_Estimation_via_Ego-Head_Pose_Estimation_CVPR_2023_paper.html)：明确研究“头戴单目视频 → 6DoF 头部轨迹 → 全身动作”。
2. [ORE / EgoStatic（NeurIPS 2023 D&B）](https://papers.nips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Paper-Datasets_and_Benchmarks.pdf)：在 600 小时左右的 Ego4D 自然视频上，用静态物体重投影代理误差评测相机轨迹。
3. [HaWoR（CVPR 2025 Highlight）](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.html)：针对手部遮挡改造 DROID-SLAM，并在 HOT3D 上单独报告 ego 相机轨迹 ATE。
4. [EgoM2P（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Li_EgoM2P_Egocentric_Multimodal_Multitask_Pretraining_ICCV_2025_paper.html)：直接从 2 秒 RGB clip 回归 60 帧相机轨迹，并与 DROID-SLAM、ACE-Zero、Align3R 比较。
5. [ReViV（作者声明已接收 ECCV 2026）](https://arxiv.org/abs/2607.17790)：从单目 ego RGB 联合预测相机、深度、注视、身体和手，在 ADT 上比较 EgoM2P、EgoMono4D、ViPE。

但截至本次调研，**没有一个统一公开榜单同时满足**：

> 普通单目 RGB + 真实头戴运动 + 独立且稠密的 metric 6DoF GT + 分钟级长轨迹 + 因果评测 + 失败覆盖率 + 机器人下游效用。

这正是现有工作的主要缺口。

## 2. 任务边界：这里的“头部姿态”是什么

本文不是从第三人称画面做人脸 yaw/pitch/roll，也不是预测未来头部动作。目标是估计**固定在佩戴者头上的相机在世界中的 6DoF 刚体轨迹**。经过固定外参，它就是头部根节点的 6DoF 轨迹。

第三视角 exo 视频不是必要输入。它只在以下情况下有帮助：

- 构造外部真值或校验头部/身体运动；
- 标定相机到头部刚体的外参；
- 评测机器人复现后的空间一致性。

如果 demo 使用预录视频，离线方法可以先完成完整视频重建再驱动仿真；如果未来要实时遥操作，则必须另外建立 causal/streaming 赛道。

## 3. 最直接的方法与评测工作

### 3.1 EgoEgo：最贴近“ego 视频 → 头部 6DoF → 身体/机器人”

[EgoEgo 项目页](https://lijiaman.github.io/projects/egoego/)和[官方代码](https://github.com/lijiaman/egoego_release)给出了目前最直接的任务定义。其流程是：

```text
head-mounted monocular RGB
    → DROID-SLAM 初始相机轨迹
    → GravityNet 对齐重力
    → HeadNet 修正头部旋转并预测平移距离/尺度
    → 条件扩散模型恢复全身动作
```

它在 ARES、KinPoly-MoCap 和 GIMO 上报告头部指标：

- \(O_\text{head}\)：\(\lVert R_\text{pred}R_\text{gt}^{-1}-I\rVert_F\)，不是角度制；
- \(T_\text{head}\)：预测与真值轨迹的平均欧氏距离，单位 mm。

| 数据集 | DROID-SLAM \(O/T\) | EgoEgo \(O/T\) |
|---|---:|---:|
| ARES | 0.62 / 411.3 mm | **0.23 / 176.5 mm** |
| KinPoly-MoCap | **0.55** / 1290.8 mm | 0.58 / **487.8 mm** |
| GIMO | **0.67** / 865.4 mm | 0.68 / **304.7 mm** |

这篇工作证明了相机轨迹经尺度和重力修正后可以直接服务于人体/机器人运动恢复。但有四个重要限制：

1. 在两个真实数据集上，旋转误差并未优于原始 DROID-SLAM；
2. 推理时假设第一帧头部朝向已知；
3. DROID 的旋转比较先把第一帧预测对齐到真值；
4. 官方定量评测排除了 DROID-SLAM 无法产生合理结果的序列，因此没有体现失败率和覆盖率。

结论：EgoEgo 应作为**任务定义和直接 head-pose baseline**，但其筛选协议不能原样用作唯一主榜。

### 3.2 EgoM2P：短片段直接回归相机轨迹

[EgoM2P 项目页](https://egom2p.github.io/)与[代码](https://github.com/ligengen/EgoM2P)将 RGB、深度、注视和相机轨迹统一预训练。相机分支的输入/输出为：

- 输入 2 秒 clip，16 个 RGB 帧，8 FPS，分辨率 \(256^2\)；
- 输出 60 个相机 pose，30 FPS；
- 每个 pose 用 6D rotation representation + 3D translation；
- 坐标统一为 OpenCV camera-to-world，并以第一帧为参考。

论文在 EgoExo4D 和 ADT 各抽取 200 个验证 clip，报告 ATE/RTE/RRE；表中 ATE/RTE 单位为 m，RRE 单位为 degree：

| 方法 | EgoExo4D ATE / RTE / RRE | ADT ATE / RTE / RRE |
|---|---:|---:|
| DROID-SLAM | 0.018 / 0.005 / 0.506 | 0.034 / 0.010 / 0.316 |
| ACE-Zero | 0.028 / 0.007 / 0.672 | 0.049 / 0.011 / 0.333 |
| Align3R | 0.019 / 0.006 / 0.762 | 0.028 / 0.010 / **0.276** |
| EgoM2P | **0.017 / 0.004 / 0.429** | 0.032 / 0.006 / 0.490 |
| EgoM2P（ADT post-train） | — | **0.026 / 0.005** / 0.480 |

论文还报告每个 60 帧序列的平均运行时间：DROID 2.7 s、ACE-Zero 426 s、Align3R 372 s、EgoM2P 0.18 s。

需要注意：

- 论文没有充分展开 ATE/RTE/RRE 的对齐公式，复现时不能仅凭名称假设协议相同；
- 2 秒片段无法测长时漂移和窗口拼接；
- clip 级批处理速度不能等同于在线因果延迟；
- EgoExo4D 的参考轨迹来自设备 MPS，不属于独立外部 mocap 真值。

结论：EgoM2P 是很合适的**短 clip、ego-specific feed-forward baseline**。

### 3.3 ReViV：当前最新的直接 ego camera baseline

[ReViV](https://reviv4d.github.io/)用一个前馈模型从单目 ego RGB 联合预测相机轨迹、深度、注视、身体和手。作者在 arXiv v1 中声明论文已接收 ECCV 2026，并声称代码和模型完全开源；但截至 2026-07-23，项目页指向的 GitHub 仓库仍返回 404。因此它目前适合作为论文结果对照，不能算“已验证可运行”的 baseline。

其 ADT 相机评测协议是：

- 将未见过的 ADT 序列全局切成 2 秒 clip；
- RGB 从 30 FPS 降至 8 FPS，其他输出保持 30 FPS；
- 每个 clip 单独用**完整 clip 真值做 Sim(3) 对齐**；
- 报告 ATE、RTE、RRE 和平均 clip 运行时间。

| 方法 | 时间/clip | ATE (m) | RTE (m) | RRE (°) |
|---|---:|---:|---:|---:|
| EgoM2P | 0.7 s | 0.030 | 0.009 | 1.290 |
| EgoMono4D | 14.2 s | 0.051 | 0.015 | 1.307 |
| VIPE | 25.8 s | **0.005** | 0.009 | 1.307 |
| ReViV | **0.7 s** | 0.015 | **0.009** | **1.279** |

ReViV 是值得纳入的最新直接 baseline，但这个结果不能被解读为长轨迹或 metric scale 榜：每个 2 秒 clip 都使用了整段未来真值做 Sim(3) 对齐，长时漂移、跨 clip 跳变、因果性和失败覆盖率均未被测试。

### 3.4 HaWoR：手部动态遮挡下的 ego 相机评测

[HaWoR 项目页](https://hawor-project.github.io/)针对第一视角中大量近景手部运动改造 DROID-SLAM：

- 从图像和 bundle adjustment 中屏蔽手区域；
- 按置信度和重投影误差调整优化；
- 结合 Metric3Dv2 估计 metric scale。

在 HOT3D 的 27 个验证视频上，论文报告：

| 方法 | ATE，经尺度对齐 (mm) | ATE-S，使用估计尺度 (mm) |
|---|---:|---:|
| DROID-SLAM | 3.80 | — |
| HaWoR camera module | **3.36** | — |
| DROID + Metric3Dv2 | — | 21.07 |
| HaWoR metric module | — | **14.61** |

它的价值在于明确评测了“手遮挡如何影响 ego 相机轨迹”和“预测尺度是否真的可用”。不足是场景较窄，只报告平移 ATE，没有旋转、失败覆盖率和在线延迟。

### 3.5 EgoMono4D：ego 4D 重建中的相机分支

[EgoMono4D](https://egomono4d.github.io/)从 ego RGB 预测内参、视频深度和置信度，再通过跨帧点云对齐得到相机 pose。原论文主评测是动态点云的 Chamfer/F-score，而不是完整的相机轨迹榜；ReViV 后续才在 ADT 上将其作为 camera baseline。它适合作为“ego 4D reconstruction”类别的代表，但相机 evaluation 证据弱于前四项。

## 4. 专门的 benchmark 与评测协议

### 4.1 ORE / EgoStatic：没有 pose GT 也能评测野外 ego 视频

[Object Reprojection Error（ORE）论文](https://papers.nips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Paper-Datasets_and_Benchmarks.pdf)从 Ego4D/EgoTracks 构建 EgoStatic：

- 5,708 段、每段约 6 分钟；
- 总计约 600 小时、超过 900 万帧；
- 约 22,000 个静态物体 tracklet；
- 覆盖 137 类日常活动。

对每个静态物体 tracklet，ORE 优化一个深度，将其中心点经待测相机轨迹重投影到其他帧；若重投影落在标注框内则误差为 0，否则计算到框的归一化 L1 距离。它在 ScanNet 上与 GT 指标具有较强相关性：

- ORE 与平移 ATE 的 Spearman 相关系数为 0.716；
- ORE 与旋转 ATE 的 Spearman 相关系数为 0.800。

EgoStatic 平均 ORE：

| DROID | COLMAP | ORB-SLAM2 | ORB-SLAM3 | ParticleSfM | MonoDepth2 | TartanVO |
|---:|---:|---:|---:|---:|---:|---:|
| 0.185 | **0.066** | 0.161 | 0.134 | 0.225 | 0.349 | 0.383 |

ORE 很适合验证方法在大量自然 ego 视频上的相对稳定性，补充材料也提供 `egostatic.zip` 示例和代码片段。但它只是 proxy：

- 不提供 6DoF GT，无法直接得到 meter/degree 误差；
- 对视线方向平移、roll 等运动存在弱约束或零空间；
- 依赖静态 tracklet 和相机内参；
- 不能替代严格的 GT 主榜。

### 4.2 LaMAria：城市尺度头戴 VIO/SLAM

[LaMAria（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Krishnan_Benchmarking_Egocentric_Visual-Inertial_SLAM_at_City_Scale_ICCV_2025_paper.html)是目前最接近“分钟级、城市尺度、真实头戴运动”的公开 benchmark：

- 63 条主测试序列，平均约 1.5 km / 26 min；
- 最长约 2.87 km / 48 min；
- 覆盖室内外、低光、移动平台和标定变化；
- 设备为 Project Aria，含双灰度相机、RGB 和 IMU。

它使用厘米级测量的稀疏控制点作为独立约束，并由视觉—惯性—控制点联合 BA 生成稠密 pseudo-GT。官方协议关注控制点 recall、pseudo-GT pose recall、ATE 和部分轨迹处理，基线包括 DSO、DM-VIO、DPVO、DPV-SLAM、ORB-SLAM3、OpenVINS、OKVIS2、Kimera 和 Aria SLAM。

局限：

- 官方主设置偏双灰度 + IMU，不是标准单目 RGB-only；
- 稠密轨迹是带控制点约束的离线 pseudo-GT，而非全程独立 mocap；
- 更适合作为长程 stress test，不应与 ADT 外部 mocap 结果混榜。

官方还提供[数据说明](https://lamaria.inf.ethz.ch/slam_documentation)和[排行榜](https://www.lamaria.ethz.ch/leaderboard)。

### 4.3 Monado SLAM Dataset：真实头部快速运动、因果性和失败率

[Monado SLAM Dataset（IROS 2025）](https://arxiv.org/abs/2508.00088)包含：

- 64 条真实 VR 头戴记录，总计约 5 h 15 min；
- 单条最长约 40 min；
- 快速转头、遮挡、闪烁、低光和游戏过程；
- SteamVR Lighthouse 提供约厘米级外部参考；
- 报告 SE(3)/Umeyama ATE、固定间隔 RTE、成功完成帧比例和相对 33 ms 帧预算的耗时。

它还区分 causal 与使用未来帧的 non-causal 设置，并完整保留崩溃/失跟结果。这种协议非常适合机器人在线控制。

主要问题是传感器以 VR 头显的双目/多目灰度或 luma + IMU 为主，不是普通前向 RGB。它应作为**头部运动与因果压力榜**，而非主 RGB 精度榜。数据以 [CC BY 4.0 发布](https://huggingface.co/datasets/collabora/monado-slam-datasets)。

### 4.4 InCrowd-VI：人群中的覆盖率和实时性

[InCrowd-VI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679079/)使用头戴 Meta Aria 采集 58 段、约 5 km / 1.5 h 的拥挤人群数据，对 DROID-SLAM、DPV-SLAM、SVO 和 ORB-SLAM3 报告：

- ATE；
- 路径漂移百分比；
- Pose Estimation Coverage；
- FPS 和 real-time factor。

它很好地补上了“动态人群下到底输出了多少有效 pose”这一维度。但参考轨迹来自 Aria MPS 离线多传感器 SLAM，虽经人工检查且官方称约 2 cm 精度，仍不是独立 GT；用它评价视觉方法存在一定同源性风险。

### 4.5 Princeton365：最成熟的 RGB 6DoF 在线评测协议之一

[Princeton365（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Kayan_Princeton365_A_Diverse_Dataset_with_Accurate_Camera_Pose_ICCV_2025_paper.html)包含 365 条 user-view RGB 视频。其独立 360° 相机能看到标志板，而被测 user view 看不到，使用 Bundle-PnP / Bundle Rig PnP 获得相机 GT；Vicon 验证中的平均 ATE 为 2.88 mm。

官方 [SLAM evaluation](https://princeton365.cs.princeton.edu/evaluation/slam/)报告：

- ATE；
- 平均旋转误差；
- 轨迹诱导 optical-flow error 的 AUC；
- pose coverage；
- 综合 Flow AUC 与 coverage 的分数。

它还提供[在线提交入口](https://princeton365.cs.princeton.edu/request_submit)。这套协议对“失败时不能静默丢帧”处理得很好。

但 Princeton365 是便携 user-view rig，不是严格头戴；长室内外视频只有进入 GT 区域的部分帧有 pose，不能把整段都当稠密长时 GT。官方还会做 Sim(3) 对齐，并在 Flow AUC 前额外做全局 SO(3) 对齐，解释旋转结果时必须注明。

## 5. 可复用数据集：参考轨迹的可信度

| 数据集 | Ego/RGB | 参考位姿来源 | 适合评什么 | 主要限制 |
|---|---|---|---|---|
| [Aria Digital Twin](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Aria_Digital_Twin_A_New_Benchmark_Dataset_for_Egocentric_3D_ICCV_2023_paper.html) | 真头戴，RGB + mono + IMU | 外部 mocap 系统生成连续 device 6DoF | 主精度榜、短 clip、手和物体交互 | 原论文 200 条；当前发布版本可能扩充，实验须锁版本 |
| [Kin-Poly / EgoMoCap](https://proceedings.neurips.cc/paper/2021/file/d1fe173d08e959397adf34b1d77e88d7-Paper.pdf) | GoPro 头戴 RGB | mocap studio，头部/身体全局运动 | 头部轨迹到机器人/身体 retarget | 266 条、约 148k 帧，场景和动作较有限 |
| [HOT3D](https://arxiv.org/abs/2411.19167) | Aria RGB；Quest 多目 | Quest 由 OptiTrack 跟踪；Aria 为 MPS 轨迹经 7DoF 对齐 OptiTrack | 手遮挡、近景交互、metric scale | Quest 严格但没有普通前向 RGB；Aria RGB 的稠密 pose 非完全独立 |
| [GIMO](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730675.pdf) | HoloLens ego RGB | HoloLens 设备跟踪 | 与 EgoEgo 对齐的真实域回归 | device-SLAM reference，不是独立 GT |
| [EgoExo4D](https://ego-exo4d-data.org/) | Aria ego RGB | Aria MPS | 大规模真实活动、EgoM2P 协议复现 | MPS reference；不应称外部 GT |
| HOT3D Quest 子集 | 真头戴，多目灰度 | 外部 OptiTrack + camera-to-head calibration | 严格头部运动/遮挡压力榜 | 模态与 RGB-only 主任务不一致 |
| LaMAria | 真头戴，RGB/mono/IMU | 稀疏测量控制点 + 稠密优化轨迹 | 长程、低光、失跟/重定位 | 稠密部分是 pseudo-GT |
| Princeton365 | user-view RGB | 隐藏标志板 + 360° 相机 | RGB 6DoF、coverage、在线提交 | 非严格头戴；长视频非全程有 pose |
| EgoStatic | 野外 ego RGB | 无 pose GT；静态物体 proxy | 大规模自然域泛化 | 不能给出完整 6DoF 绝对误差 |

ADT 是本项目最合适的主数据集。其[数据格式说明](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)明确区分 `aria_trajectory.csv` 真值与 MPS 输出。Kin-Poly 则最贴近“相机/头部轨迹最终驱动身体或机器人”的下游关系。

## 6. 各论文的数字为什么不能直接横向排名

| 工作 | 对齐方式/尺度处理 | 主要指标 | 没有测到的关键问题 |
|---|---|---|---|
| DA3 | 首帧归一；两两相对旋转与归一化平移方向 | pairwise pose AUC | 平移长度、长漂移、连续性、coverage |
| EgoEgo | 第一帧朝向给定/对齐；学习距离与重力 | \(O_\text{head}\)、\(T_\text{head}\) | 失败序列、统一模型泛化、因果延迟 |
| EgoM2P | 第一帧参考；论文未充分说明 metric alignment | ATE/RTE/RRE | 2 秒以上轨迹、窗口拼接、coverage |
| ReViV | 每个 2 秒 clip 用全 clip GT 做 Sim(3) | ATE/RTE/RRE | metric scale、长漂移、因果性、跨 clip 连续性 |
| HaWoR | ATE 做尺度对齐；ATE-S 使用估计尺度 | 平移 ATE | 旋转、失败率、长时行为 |
| ORE | 每个静态物体轨迹优化一个深度 | proxy reprojection error | 完整 6DoF GT 和 metric 误差 |
| Princeton365 | Sim(3)；Flow AUC 另做全局 SO(3) | ATE、rotation、flow AUC、coverage | 原始 metric 输出误差 |
| Monado | SE(3) 或 Umeyama；保留失败 | ATE、RTE、coverage、runtime | 普通单目 RGB 模态 |

因此正式比较必须复用同一份轨迹转换、时间同步和评测代码，而不是直接抄论文表格。论文数字只用于确认某方法是否值得纳入 baseline。

## 7. 相邻但有用的通用视频方法

这些方法不是专为 ego 设计，但能补齐方法机制：

- [ViPE](https://research.nvidia.com/labs/toronto-ai/vipe/)：原始视频 → 内参、相机轨迹、近 metric 深度；结合 dense flow、稀疏 SLAM 和 metric depth。其[代码](https://github.com/nv-tlabs/vipe)已支持广角并在 2026 版本接入 DA3 depth pipeline。ReViV 的 ADT 表中它取得最低 ATE，值得优先跑。
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)：EgoEgo、HaWoR、EgoM2P 等多篇论文共同使用的学习式 SLAM 基线，最适合作为可解释的轨迹参考。
- [DPVO / DPV-SLAM](https://github.com/princeton-vl/DPVO)：轻量 VO 与带全局后端版本，LaMAria、Princeton365、InCrowd-VI 等协议均有覆盖。
- [VGGT](https://github.com/facebookresearch/vggt) / [VGGT-Ω](https://arxiv.org/abs/2605.15195)：前馈几何基础模型类别，用于判断 DA3 的收益是否来自特定训练或一般多视图先验。
- [MegaSaM](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MegaSaM_Accurate_Fast_and_Robust_Structure_and_Motion_from_Casual_CVPR_2025_paper.pdf)与 [MonST3R](https://openreview.net/forum?id=lJpqxFgWCM)：面向动态 casual video，可作为大量人体运动场景的补充。
- [WildGS-SLAM / Wild-SLAM MoCap](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.pdf)：不是头戴数据，但提供动态人物/物体环境中的 OptiTrack RGB-D 轨迹，可用来隔离“动态遮挡”这一变量。

### 7.1 搜索到但不应作为主 6DoF 榜的近邻工作

- [BPOD](https://arxiv.org/abs/2112.13018)确实是头戴快速运动数据，也评测 ORB-SLAM3、DSO 和 TrianFlow，但其真值主要是行人在地面的 2D 位置，不包含完整头部朝向，不能承担 6DoF 主榜。
- [Human POSEitioning System（HPS）](https://openaccess.thecvf.com/content/CVPR2021/papers/Guzov_Human_POSEitioning_System_HPS_3D_Human_Pose_Estimation_and_Self-Localization_CVPR_2021_paper.pdf)从头戴视频和身体 IMU 在预扫描场景中恢复人体位姿与位置，任务相邻，但不是未知场景下的 RGB-only 相机轨迹 benchmark。
- HoloLens/ARKit 数据集中的系统 pose，包括 GIMO、HoloSet 等，适合做工程参考和真实域测试；它们通常没有独立外部轨迹真值，不能用来证明方法优于设备自身 VIO。
- 从第三视角做 face head-pose estimation、从历史帧预测未来 head pose、或只预测重力/yaw-pitch-roll 的工作，均没有输出本项目需要的完整世界坐标 6DoF 轨迹，因此未纳入 baseline 主表。

## 8. 推荐 baseline 组合

### 8.1 最小可执行版本

如果当前 demo 只跑 5 个方法，建议：

| 方法 | 代表机制 | 纳入原因 |
|---|---|---|
| DA3NESTED-GIANT-LARGE-1.1 | 多视图 foundation model + metric depth | 当前项目主线；直接输出相机外参并尝试恢复 metric scale |
| EgoM2P | ego-specific 前馈模型 | 已有公开代码；直接从 2 秒 RGB clip 输出 camera trajectory |
| ViPE | 视频几何系统 | ReViV 表中 ADT ATE 最强，支持真实动态视频和近 metric 深度 |
| DROID-SLAM | 学习式 SLAM | 多篇直接相关论文的共同基线，能测出 foundation model 是否真正更好 |
| DPV-SLAM | VO + 全局优化 | 补充长轨迹、回环和 coverage 能力 |

若只能跑 4 个，优先 `DA3NESTED-GIANT-LARGE-1.1 + EgoM2P + ViPE + DROID-SLAM`。ReViV 的公开仓库恢复可访问后，再将它加入。

若手部经常遮挡相机，再增加 HaWoR camera module；若需要最直接的历史 head-pose 对照，再增加 EgoEgo。

### 8.2 完整论文版本

建议按机制分组，避免堆很多相似模型：

- Ego-specific：EgoEgo、EgoM2P、EgoMono4D、HaWoR，以及代码可访问后的 ReViV；
- Foundation/video geometry：DA3NESTED-GIANT-LARGE-1.1、VGGT 或 VGGT-Ω、ViPE；
- 学习式轨迹估计：DROID-SLAM、DPVO/DPV-SLAM；
- 经典几何控制组：ORB-SLAM3；
- 动态场景补充：MegaSaM 或 MonST3R。

所有方法必须只读取赛道允许的输入。RGB-only、RGB+IMU、使用预测深度、使用真值内参应分开报告。

## 9. 推荐 evaluation 设计

### 9.1 数据赛道

| 赛道 | 推荐数据 | 目的 |
|---|---|---|
| A. 严格 ego RGB 主榜 | ADT + KinPoly-MoCap | 外部参考下的头戴 6DoF 精度 |
| B. 手/物遮挡 | HOT3D Aria/Quest 分开报告 | 近景动态遮挡和 metric scale |
| C. 长程压力榜 | LaMAria + Monado | 漂移、重定位、因果性和覆盖率 |
| D. 人群压力榜 | InCrowd-VI | 动态人群、失跟和实时性；标为 MPS reference |
| E. 野外泛化榜 | EgoStatic ORE | 数百小时自然 ego 视频；只作 proxy |
| F. RGB 协议复核 | Princeton365 | 独立位姿、在线服务器和 coverage |
| G. 机器人下游榜 | 从 A/B 中挑选有代表性的连续片段 | 直接测 IK 与仿真复现效果 |

### 9.2 对齐协议

每种方法至少报告三种设置：

1. **Raw/metric SE(3)**：方法声称 metric translation 时，不允许缩放；只做坐标系约定所需的固定变换。
2. **Prefix alignment**：尺度不确定时，只用开头 1–2 秒估计 Sim(3)，然后固定到全段。它更接近 demo 中一次初始化后运行。
3. **Full-trajectory oracle**：允许整段 Sim(3) 对齐，仅作为算法几何质量诊断，不能当部署主结果。

还应增加 first-frame-only 对齐，模拟已知初始头部姿态。禁止把不同对齐方式的 ATE 放在同一列比较。

### 9.3 核心轨迹指标

- 平移 ATE：meter，报告 median / mean / P90 / P95；
- 旋转 geodesic error：degree，而不是旋转矩阵 Frobenius norm；
- 多间隔 RPE：\(\Delta=1\) frame、1 s、5 s、30 s；
- path-length drift 和 scale ratio/error；
- 不同时长/路径长度分桶：2 s、20–30 s、2–3 min、10 min+；
- pose coverage、完整成功序列比例、首次失败时间、重定位时间；
- time-to-first-pose、平均/尾部延迟、FPS、峰值显存；
- 静止片段的平移/旋转 jitter、速度、加速度和 jerk；
- 固定窗口推理时的边界平移跳变和边界旋转跳变。

对单目方法，尺度未知不是“失败”，但必须明确标记为 scale-free，并与真正 metric 方法分列。

### 9.4 机器人下游指标

轨迹误差之外，再直接测机器人是否能用：

- IK 成功帧比例；
- 关节限位违反率和碰撞率；
- 目标头部位置/朝向跟踪误差；
- 关节速度、加速度、jerk 超限率；
- 因滤波造成的相位延迟；
- 丢 pose 时 hold-last-pose、插值、重定位策略的恢复时间。

这组指标能揭示一个常见现象：ATE 相近的两条轨迹，抖动较大的那条会给机器人带来完全不同的控制效果。

## 10. 建议的实验顺序

1. 先在 ADT 和 KinPoly 上统一 `timestamp + T_world_camera + valid` 数据接口，并验证 W2C/C2W、米/毫米和坐标轴。
2. 跑 DA3、DROID-SLAM、ViPE、EgoM2P 四个核心 baseline；同一序列同时生成 raw、prefix、oracle 三套结果。
3. 增加静止片段、快速转头、手遮挡和跨窗口边界诊断。
4. 将预测相机轨迹通过固定外参转换成头部轨迹，接入机器人 IK；报告轨迹指标和下游指标。
5. 再扩展 DPV-SLAM、HaWoR、EgoEgo，以及代码开放后的 ReViV，并加入 LaMAria/Monado/EgoStatic 压力赛道。

这样第一轮就能回答三个关键问题：

- DA3 是否比标准 SLAM 更准；
- DA3 是否比直接为 ego 训练的 EgoM2P 更稳，并在 ReViV 代码公开后复核最新方法；
- 离线几何误差的提升是否真的转化成更高的机器人 IK 成功率和更低抖动。

## 11. 可以形成的研究缺口

现有文献已经分别覆盖了短 clip 精度、野外 proxy、长程 SLAM、metric scale 和下游人体恢复，但没有把它们统一起来。一个有价值的 benchmark/evaluation 贡献可以定位为：

> **面向机器人 retargeting 的 egocentric RGB 6DoF head-trajectory benchmark**：统一外参、尺度、因果性、失跟覆盖率、跨窗口连续性与机器人下游指标，并比较 ego-specific、foundation model、学习式 SLAM 和经典几何方法。

要让这一定位成立，最重要的不是再增加一个 ATE 表，而是提供：

- 独立且可追溯的 reference-pose 分级；
- 不偷看未来的 prefix/causal 对齐协议；
- 失败序列不被删除的 coverage 统计；
- metric scale 与 scale-free 结果分榜；
- 相机轨迹到机器人 IK 的闭环下游评测。

## 12. 最终建议

对于当前 demo，可以直接把 DA3 输出的 W2C 外参求逆，并通过固定相机—头部外参转换为机器人目标 6DoF。工程上先按离线轨迹使用是合理的。

对于正式 evaluation，建议以 **ADT + KinPoly** 为严格主榜，**EgoStatic + LaMAria/Monado + InCrowd-VI** 为泛化和长程压力榜；baseline 首轮使用 **DA3NESTED-GIANT-LARGE-1.1、EgoM2P、ViPE、DROID-SLAM、DPV-SLAM**，ReViV 暂作论文对照并等待其仓库开放。报告 raw/prefix/oracle 三种对齐，并把 coverage、尺度、抖动、跨窗口跳变和机器人 IK 成功率列为一等指标。

这能避免把“DA3 在两两相机相对方向上得分高”误解成“DA3 已经稳定恢复了可直接驱动机器人的连续 metric 头部轨迹”。
