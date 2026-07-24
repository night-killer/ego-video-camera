# 彩色透视 Ego Video 位姿估计：项目审计、模型微调与 Evaluation 实验报告

> 审计日期：2026-07-24
>
> 任务：输入单目彩色、非鱼眼 ego RGB 视频，估计逐帧 6DoF 相机位姿；机器人头部或腕部视角优先，真人头戴或手持数据可作为补充。
>
> 文档状态：这是资源、协议与可执行性的实验报告。本轮运行环境是 CPU，未安装模型环境、未编译 CUDA、未执行任何新模型推理；只有第 2.4 节的 EgoBody + 旧 DA3 数字是历史 pilot，其他数值均为数据完整性或论文引用，不是本项目新跑分。

## 1. 最终结论

1. `core65` 的设置偏差已经确认：112 个 clip 中有 52 个 Aria 鱼眼输入和 22 个灰度输入；只有 EgoBody 20 个与 Princeton365 18 个同时是彩色透视，且机器人 ego 为 0。它现在只作为 `legacy_diagnostic`，不再承担用户要求的主榜。
2. 新的 A 级 pilot 已真实落盘并严格校验：TUM、Bonn Dynamic、OpenLORIS Office D435i color 各 4 段，共 **12 clips / 180 s / 1800 RGB frames / 907,006,401 bytes**。它满足原生彩色和 pinhole 输入，但只有 OpenLORIS 是机器人平台；规模只够 adapter/CI 回归，不能据此做统计性模型排名。
3. 交互/机器人 B 级 pilot 已升级为 v2：DROID wrist、HoloAssist、RH20T cfg3 wrist、Stera-10M 各 4 段，共 **16 clips / 200 s / 2000 frames / 1,242,321,622 bytes**。DROID/RH20T 是机器人运动学 B2，HoloAssist/Stera 是设备跟踪 B1；四者不得混入外部 mocap A 榜。
4. Oxford-IHM 仍是最值得补的“机器人 ego + 动态人 + 外部 mocap”数据，GDPR 申请已经提交、仍待审批。Stera-10M 和 NVIDIA Cosmos 的 gated access 已获批：Stera 只抽取 4 段 evaluation，Cosmos 缺失的 5 个 tokenizer 文件已补齐。
5. 模型资源准备已完成：`thirdparty/` 有 **13 个顶层 git submodule**，清理下载 metadata/cache 后的 `ckpts` 为 **57,991,051,668 bytes**，验证结果是 **88 complete / 0 missing**。ReViV 的 256/512、camera 与 metric-depth 所需权重现在都已存在。
6. “资源完整”仍不等于“evaluation 完成”。本轮 CPU 机器没有运行 DA3-1.1、ReViV、EgoM2P、DROID、ViPE、VGGT-SLAM 等推理；方法层目前仍只有已有 DA3/DA3-Streaming adapter，缺少统一 method adapter、失败协议和跨模型结果表。
7. 主模型矩阵应覆盖通用几何、经典/学习式 SLAM、动态视频与 ego 专项四类。最低组合建议为 `DA3-1.1/Streaming`、`DROID-SLAM`、`ORB-SLAM3`、`ViPE`、`VGGT-SLAM`、`ReViV`、`EgoM2P`；动态和手遮挡子表增加 `MegaSaM` 与 `HaWoR`。
8. **ReViV 是第一微调对象，EgoM2P 是直接对照。** 但 HoloAssist 已在 ReViV 预训练域内，所以相应结果必须标为 `pretrained-domain`，不能称 zero-shot；RH20T/DROID held-out task 才能回答机器人腕部 adaptation 的问题。
9. 论文中的 ReViV/EgoM2P 数字不能当作本项目结论。ReViV 在 ADT 上的 `0.015/0.009/1.279°` 使用每个 2 秒 clip 的整段 GT 做 Sim(3) 对齐，只是短窗口 oracle，不衡量 metric scale、长时漂移、拼接跳变或部署失败率。

## 2. 对当前项目的审计

### 2.1 已有资产与真正可运行的范围

| 项目 | 当前状态 | 审计判断 |
|---|---|---|
| 原始 demo | EgoBody 3 个 20 秒 clip，8 FPS，DA3 推理和可视化已完成 | 可用于坐标、同步、投影和对齐回归，不足以支持模型优劣结论 |
| Legacy 数据 | `core65` 为 112/112；清除仓库外 177,344,813,322-byte 中间目录（含 153 GB 下载缓存和重复输出）后，唯一保留副本为 13,968,075,942 bytes | 覆盖广，但混合鱼眼、灰度、preview、手持和设备参考，只作诊断 |
| Native A pilot | 12/12 clips，180 s，1800 RGB frames，严格 verify 通过 | TUM/Bonn/OpenLORIS 已统一输出帧、时间、相机信息与 reference；样本量仍不足以排名 |
| Robot B pilot | 16/16 clips，200 s，2000 RGB frames，严格 verify 通过 | DROID/HoloAssist/RH20T/Stera 均有动态 reference；B1/B2 与 A 榜隔离 |
| Checkpoint | `ckpts` 57,991,051,668 bytes；88 complete / 0 missing | 所有声明权重均通过大小、格式与哈希检查；Hugging Face metadata/cache 已清理 |
| ReViV | `thirdparty/ReViV` 与两条推理权重路径完整 | camera 与 metric-depth tokenizer 均已齐；尚未在本机创建模型环境或运行推理 |
| Dataset adapter | 两个新 profile 已统一 RGB、frame manifest、相机 metadata 与轨迹 reference | 已解决下载和输入契约；还没有统一模型输出 `T_world_camera` adapter |
| Method adapter | 只有 DA3/DA3-Streaming 接入现有评测流程 | 其余已下载模型还不能在同一协议下一键比较 |
| 指标 | 已有 Sim(3)/SE(3)、ATE、约 1 秒 RPE、旋转误差、coverage、退化检测 | 缺多时间尺度 RPE、尺度漂移、reset/失跟、生存率、bootstrap CI 和统一失败计分 |

`scripts/verify_eval_checkpoints.py --hash` 的本次退出码为 `0`，88 个声明文件
全部通过大小、格式与哈希检查；退出码 `1` 才表示文件缺失或校验失败。

当前实验仍加载 `/data/aigc/cyb/zxgu/ckpt/DA3NESTED-GIANT-LARGE`，不是新下载的 `DA3NESTED-GIANT-LARGE-1.1`。因此 DA3-1.1 也属于“资源准备完成、正式结果未跑”。

### 2.2 `core65` 的硬条件偏差

| 数据集 | clip 数 | 颜色与相机 | 位姿参考 | 新定位 |
|---|---:|---|---|---|
| ADT | 24 | 彩色 Aria RGB，明显鱼眼；原始模型为 `Fisheye624` | 外部 GT system/mocap | 只进 `rectified-common/A`，不进原生透视榜 |
| HOT3D | 16 | 彩色 Aria RGB；本地 `camera_models.json` 明确为 `FISHEYE624` | camera trajectory 主要为 MPS/device reference | `rectified-common/B` 手物遮挡诊断 |
| InCrowd-VI | 12 | 彩色 Aria RGB；官方 calibration 明确为 `FisheyeRadTanThinPrism` | offline MPS reference | `rectified-common/B` 人群压力诊断 |
| Monado | 16 | 灰度 | Lighthouse 外部参考 | 灰度快速头动诊断，不进入彩色榜 |
| LaMAria | 6 | 已矫正为 pinhole，但 ASL cam0 为灰度 | control points + joint BA/pseudo-GT | 灰度长程诊断，不进入彩色榜 |
| EgoBody | 20 | HoloLens PV 彩色透视 | HoloLens `pv2world` device reference | 原生透视/B；保留作真人交互和工程回归 |
| Princeton365 | 18 | 3840x2160 彩色透视，普通径向/切向畸变 | 隐藏标志板 + 360 相机 rig，另有 Vicon 验证 | 原生透视/A；保留作高精度泛化集 |

按互斥的输入类别统计：

```text
原生彩色透视： 38 / 112 = 33.9%
彩色鱼眼：     52 / 112 = 46.4%
灰度：         22 / 112 = 19.6%
机器人 ego：    0 / 112 = 0.0%
```

这里的“原生彩色透视”只判断图像条件，尚未把位姿参考等级计算进去。EgoBody 的 20 个 clip 仍然不能进入外部真值主榜。

另一个容易被忽略的问题是 ADT/HOT3D 当前保存的是官方 H.264 preview，而不是 raw VRS。实际抽帧仍可看到明显桶形畸变，且 preview 不能替代逐帧 online calibration。若要做严格矫正赛道，应从 raw VRS 按时间戳读取相机标定并记录虚拟 pinhole 参数。

### 2.3 本轮实际完成的数据实验

本轮完成的是数据与 reference pipeline 实验，不是模型精度实验。两个固定 manifest
均通过逐帧图像和 reference 文件 SHA-256、RGB mode、分辨率、pinhole 标志与
固定帧数检查：

| Profile / 数据 | Clip | 秒 | 输出帧 | 关键设置 |
|---|---:|---:|---:|---|
| Native / TUM | 4 | 60 | 600 | 原生 RGB；官方 mocap trajectory |
| Native / Bonn Dynamic | 4 | 60 | 600 | 原生 RGB；动态人/物体 + OptiTrack |
| Native / OpenLORIS Office | 4 | 60 | 600 | 只选 D435i color，排除 T265 fisheye；OptiTrack |
| Robot / DROID wrist | 4 | 20 | 200 | 4 个实验室；按 H5 `estimated_capture` 真实时间，再按 H5 index 抽 MP4；动态 C2Base |
| Robot / HoloAssist | 4 | 60 | 600 | 官方 test-v1_2；GoPro/Switch/DSLR 分层；动态 C2World |
| Robot / RH20T cfg3 wrist | 4 | 60 | 600 | 4 个 task/scene；真实毫秒时间、TCP 和手眼标定；动态 C2AlignedBase |

RH20T 的 pose 不是把 gripper pose 直接改名。按照官方 API 的列向量约定，
cfg3 `tc_mat` 是 `T_camera_tcp`，而 `transformed/tcp_base.npy` 已包含 base/TCP
坐标对齐，因此导出公式固定为：

```text
T_aligned_base_camera = T_aligned_base_aligned_tcp
                      @ inv(align_tcp_matrix)
                      @ inv(T_camera_tcp)
```

实现还在每个 calibration 上用官方 `extrinsics_base_aligned` 矩阵链做方向闭环，
残差门槛为 `1e-8`。四段 RH20T 的 150 个输出都能找到同时间戳 TCP；由于
原生采集约 8--9 Hz，统一 10 Hz 后分别有 129/129/126/121 个唯一源帧，最大
最近帧时间误差为 67/68/68/70 ms。`task_0004` 原计划的 2--17 秒窗口跨过
665 ms 采集空洞，最终固定为 1--16 秒，而不是放宽 250 ms 质量门。

HoloAssist 必须标成 `pretrained-domain/device-reference`，因为 ReViV released
weights 已使用 HoloAssist。DROID 和 RH20T 必须标成 `robot-kinematic/B2`。
本轮没有产生 ATE、RPE、旋转误差、FPS 或显存数字。

资源清理也已完成：RH20T 27,399,012,782-byte 归档通过整包 SHA-256
`b49b2970...e38b6d3` 后只提取固定子集并删除；`core65` 的仓库外
177,344,813,322-byte 中间目录（含 153 GB 可重建 remote-ZIP 缓存和重复
evaluation 输出）也在两份数据分别通过 112/112 verify 后删除。完整状态见
[`ego_pose_eval_resource_status.yaml`](../configs/ego_pose_eval_resource_status.yaml)。

### 2.4 历史结果能说明什么

已有三段 DA3 结果应继续保留，但标题应写为“EgoBody device-reference pilot”：

- Easy 的参考相机中心在 20 秒内只移动 3.02 cm，低于现有 10 cm 可观测性门槛；所有尺度对齐被正确标记为 `degenerate`。
- Medium 的 full-clip oracle 旋转误差 median/P95 为 `1.46°/4.40°`，而 prefix 为 `7.26°/9.19°`。
- Hard 的 oracle 为 `5.14°/7.25°`，prefix median 达 `34.62°`。

这三点主要证明当前实现能识别退化，以及 full-clip oracle 会显著美化结果。它们不能回答 DA3-1.1 是否优于 ego 专项模型，也不能代表机器人、动态物体、手遮挡或长视频表现。

## 3. 重新筛选数据集

### 3.1 A 级：原生彩色透视 + 外部真值

| 优先级 | 数据集 | Ego 形态 | RGB 与镜头 | 位姿来源 | 用法与准入条件 |
|---|---|---|---|---|---|
| P0 | [Oxford-IHM](https://ori.ox.ac.uk/publications/datasets/oxford-indoor-human-motion-dataset-2024) | Toyota HSR 移动机器人视角，另有静态相机；约 60 min | robot-mounted RGB-D 彩色流 | 传感器、障碍和人体由 mocap 以 100 Hz 跟踪，官方称平均残差为亚毫米级 | 最贴近“机器人 ego + 动态人”。GDPR 申请已提交、待审批；收到 rosbag 后必须先检查 `camera_info.distortion_model`、RGB frame 与 mocap rigid-body 外参，合格后才升为 headline |
| P0 | [OpenLORIS-Scene](https://lifelong-robotic-vision.github.io/dataset/scene.html) `D435i color` | 约 1 m 高的轮式机器人，以人类步速或更慢移动 | 彩色 848x480 @ 30 Hz，FOV 69°x42°；不要误用 T265 双鱼眼 | office 为 OptiTrack；其他场景为 offline LiDAR SLAM | 立即可用的机器人原生透视候选。严格 A 表只用 office；其他场景单列为 LiDAR-SLAM reference |
| P0 | [Bonn RGB-D Dynamic](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | 移动/手持 RGB-D，相机周围有人和物体运动 | ASUS Xtion Pro LIVE 彩色流，论文采用 pinhole 模型 | OptiTrack Prime 13 | 24 dynamic + 2 static；箱子、气球、人群和遮挡非常适合动态压力榜。不是机器人头戴，主任务只输入 RGB |
| P0 | [Princeton365](https://princeton365.cs.princeton.edu/) | user-view 手持 rig | 彩色透视，普通 radtan 畸变 | 隐藏标志板 + 360 相机辅助 rig；Vicon 验证 | 已下载，适合高精度 RGB 泛化和 coverage；不得标成 head-mounted |
| P1 | [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | 手持 Kinect | 彩色 640x480 @ 30 Hz，透视 | 8 相机 mocap @ 100 Hz | 生态成熟、下载小；用于坐标、尺度和 evaluator 回归，不作为新 ego 结论 |

Oxford-IHM 目前标记为“条件准入”而非无条件合格，是因为公开网页说明了 robot-mounted RGB-D 和 mocap，却没有公开交付 rosbag 中 RGB 镜头的具体畸变模型。实验报告不能在拿到标定前替数据集补写 `pinhole`。

### 3.2 B 级：彩色 Ego + 设备或运动学参考

| 数据集 | Ego 形态和规模 | 轨迹 | 最适合的角色 | 关键限制 |
|---|---|---|---|---|
| [EgoBody](https://github.com/sanweiliti/EgoBody) | HoloLens PV，125 sequences、199,111 ego RGB frames | HoloLens PV/device tracking | 复用现有工程；真人面对面交互、头动和遮挡 | device reference，不是独立 GT |
| [HoloAssist](https://holoassist.github.io/) | HoloLens 彩色 ego，约 166 h；手、头、物体与动作信息丰富 | `Video/Pose_sync.txt`、`Head_sync.txt` 与标定 | 已落盘 4 个 test-v1_2 pilot；ReViV 适配训练和手部动作压力集 | B1 设备轨迹；ReViV 预训练已经使用该数据，不能把其测试集称为严格 zero-shot |
| [Stera-10M](https://huggingface.co/datasets/fpvlabs/stera-10m) | 头戴 iPhone Pro；README 称 584 sessions，固定 revision 实际发布 575 个完整目录；200 h、约 10M RGB frames | 每帧 ARKit 6DoF | 已落盘 4 段、60 s、600 帧；覆盖走动取物、重物操作、人与人交接、近距离双手精细操作 | B1；access 已获批但全量约 1.6 TB，当前只保留 816,291,198-byte evaluation；ARKit 不能证明厘米级绝对精度 |
| [RH20T](https://rh20t.github.io/) | 1-2 个 robot in-hand RGB-D camera；110K+ 操作序列 | 100 Hz gripper Cartesian pose + camera/robot calibration | 已落盘 cfg3 的 4 task/scene pilot；最直接的机器人腕部适配集 | 已导出动态 C2AlignedBase；B2 误差包含机械臂与手眼标定误差 |
| [DROID](https://droid-dataset.github.io/) | 大规模真实机器人操作，多视角含腕部相机 | robot state + calibration | 已落盘 4 个跨实验室 wrist pilot；机器人视觉域适配 | 已导出动态 C2Base，但仍是 B2 运动学参考，不是外部真值 |

Stera-10M 数据卡统计是 200 h、584 sessions、约 10M 帧、最长 104 min；当前
固定 revision 的文件树实际包含 575 个完整 session。四段 pilot 的 MP4 均为
1280x720 @ 15 FPS，标定为 `plumb_bob`。MP4 与 HDF5 pose 按 frame index 一一
对应；其中一个 session 的绝对 ARKit 时间存在 pause，因此窗口按 MP4 index
抽取，并通过 250 ms pose-gap gate，不按绝对时间误采样。

本表只说明彩色 ego 形态和参考类型，不自动授予 `native-perspective-color` 标签。Stera 等数据仍要从 `rgb_K/rgb_D` 或等价 calibration 确认投影模型；无法确认或实际使用鱼眼镜头的 session 转入 rectified/diagnostic 赛道。

### 3.3 C 级：保留但必须换赛道

| 类别 | 数据 | 处理 |
|---|---|---|
| 鱼眼但有重要外部 GT | ADT | 从 raw VRS 使用官方标定生成固定虚拟 pinhole；记录 `virtual_K/output_size/FOV/valid_mask`，只进 `rectified-common/A` |
| 鱼眼、强手部/人群动态 | HOT3D、InCrowd-VI | 同样矫正，但参考仍是 B；用于 HaWoR、ReViV、动态 mask 消融，不与 ADT A 级合并 |
| 灰度 | Monado、LaMAria | 只进入 grayscale diagnostic；不能通过 RGB channel replication 改名为彩色 |
| 视觉重建参考 | Ego-Exo4D、EPIC-Fields、HD-EPIC | 用于 coverage 和定性压力测试，不用来宣称绝对相机精度 |
| 精确合成 | Isaac Sim/其他机器人仿真或 EgoGen 类数据 | 明确设为 perspective camera、关闭鱼眼畸变并直接记录 camera C2W；只作训练和受控回归，真实榜独立报告 |

### 3.4 推荐的冻结测试集结构

本轮 adapter/CI pilot 的 sequence ID 已固定在两个 YAML manifest 中。下面是扩大到
论文级统计时的目标配额；新增 sequence 必须先按 subject/scene/task 冻结，再运行
任何模型：

| 层级 | 数据 | 建议首轮配额 | 目的 |
|---|---|---:|---|
| CI sanity | TUM 4 + Bonn 4 + OpenLORIS-office 4 + Princeton365 4 | 16 clips x 20-30 s | 坐标、时间同步、普通运动和动态场景回归 |
| 机器人 A 榜 | OpenLORIS-office + 条件准入后的 Oxford-IHM | 每个来源至少 12 个独立 sequence/window | 机器人 ego headline |
| 动态 A 榜 | Bonn dynamic + static control | 12 dynamic + 4 static | 分离动态前景造成的性能损失 |
| 人类透视 A 榜 | Princeton365 + TUM | 12 + 8 | 人类手持、室内外和几何泛化 |
| B 榜 | EgoBody + HoloAssist + RH20T + Stera 长视频 | 每来源至少 10 个独立 subject/session/task | 手部、操作、机器人腕部和长时存活率 |
| Rectified appendix | ADT + HOT3D + InCrowd-VI | 每来源 8-12 个 | 与 ego 文献对齐，但不污染原生透视结论 |

每个原始 sequence 最多进入一个统计折；训练、验证、测试按 subject、scene、task、robot configuration 分组，而不是随机拆帧。

## 4. 模型查漏与优先级

### 4.1 主模型矩阵

| 优先级 | 方法 | 类型 | 动态/手部机制 | 可微调性与当前项目状态 | 实验角色 |
|---|---|---|---|---|---|
| P0 | [ReViV](https://github.com/lvsean/reviv4d) | ego 专项 400M masked multimodal transformer；2 s RGB 直接输出 camera/body/hands/gaze/depth | 联合学习 viewer 与 view，利用人体和手运动先验，不依赖预计算 SLAM | 本地 submodule、256/512 推理权重、Cosmos metric-depth tokenizer 和训练代码均完整；尚未建环境/推理 | **第一微调对象**和最新直接 ego baseline |
| P0 | [EgoM2P](https://github.com/ligengen/EgoM2P) | ego 专项多模态前馈模型；2 s RGB -> 60 camera poses | 从多个 ego 数据学习相机和视觉相关性 | 训练、post-training 和本地权重已有 | ReViV 的直接架构/预训练对照 |
| P0 | [DA3-1.1 + DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3) | 通用多视图 foundation geometry | 没有显式手 mask；通过几何与多帧先验处理动态 | 新权重已下载，现有 adapter 可复用；正式 1.1 结果未跑 | 延续当前主线，短片与长片都要跑 |
| P0 | [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | learned dense SLAM | 鲁棒优化但默认静态世界 | 权重已下载；尚无统一 method adapter | EgoEgo、HaWoR 等工作的公共锚点 |
| P0 | [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | 经典几何 SLAM | RANSAC/关键点外点抑制，无语义动态处理 | 源码与 42,527,984-byte vocabulary 已校验；仍需固定参数和失败规则 | 必须有的非 foundation 基线 |
| P0 | [ViPE](https://github.com/nv-tlabs/vipe) | 通用动态视频几何系统；flow/tracks/metric depth + BA | 显式融合稠密 flow、稀疏轨迹和优化，适合真实动态视频 | 1.2 依赖与权重已下载；尚无 adapter | 动态和近 metric 通用强基线 |
| P0 | [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) | foundation submap + SLAM 后端 | 局部 foundation geometry + 全局图优化 | 权重已下载；尚无 adapter | 检验全局后端和回环的收益 |
| P1 | [LingBot-Map](https://github.com/Robbyant/lingbot-map) | 长视频 streaming foundation model | 长时 cache/window，不专门建模手 | long checkpoint 已下载 | 3 min/10 min+ 长视频子表 |
| P1 | [MegaSaM](https://github.com/mega-sam/mega-sam) | dynamic casual video camera/depth optimizer | 联合动态场景的 camera motion 与几何 | 权重已下载；离线且较慢 | Bonn/人群动态专项 |
| P1 | [HaWoR](https://github.com/ThunderVVV/HaWoR) camera module | ego 手遮挡专项；hand mask + masked DROID + metric depth | 显式移除第一视角手区域 | 推理权重已下载，但 README 仍写 training code 待发布 | 手遮挡消融和 inference baseline，不作为主微调对象 |
| P2 | [EgoEgo](https://github.com/lijiaman/egoego_release) | DROID/flow 后的 ego head-motion correction | 学习头动与视觉运动关系 | 可训练，但依赖 SLAM 成功 | 历史 head-pose 对照；失败样本不得先删除 |

如果算力只允许 6 个模型，先跑：

```text
DA3-1.1/Streaming + DROID-SLAM + ViPE + VGGT-SLAM + ReViV + EgoM2P
```

ORB-SLAM3 计算成本较低，应尽量补上。HaWoR 和 MegaSaM 只在动态/手遮挡子集跑即可。

### 4.2 ReViV 的时效性修正

官方 GitHub 仓库创建于 `2026-07-21T01:59:57Z`，已发布：

- 约 400M 参数的 12 encoder + 12 decoder 主模型；
- 256 路径：16 帧、256x256、8 FPS；
- 512 路径：32 帧、512x512、16 FPS；
- 每个 2 秒 clip 输出 `[60, 9]` camera trajectory，30 FPS；
- 完整的 tokenizer、WebDataset、训练和推理说明；
- code 为 Apache-2.0，released weights 为限制非商业用途的 Sample Code License。

因此旧计划的 ReViV 404 结论只代表 2026-07-23 当时状态，不能继续用于模型排期。
本地固定 commit 为 `de23a67009685e3878e4bad49d33f023d4b7a085`，ReViV/EgoM2P
共享的 256 路径、`reviv_500b` 组权重及 Cosmos-1.0 metric-depth tokenizer
均通过文件级检查。资源层已不再阻塞 camera 或 metric-depth 分支；尚未完成的
是 GPU 环境、method adapter 和正式推理。

### 4.3 论文数字的正确解读

ReViV 论文在 unseen ADT 上报告：

| 方法 | time/2 s clip | ATE | RTE | RRE |
|---|---:|---:|---:|---:|
| EgoM2P | 0.7 s | 0.030 | 0.009 | 1.290° |
| EgoMono4D | 14.2 s | 0.051 | 0.015 | 1.307° |
| ViPE | 25.8 s | **0.005** | 0.009 | 1.307° |
| ReViV | **0.7 s** | 0.015 | **0.009** | **1.279°** |

论文明确写明所有预测轨迹都用**整个 clip**的 GT 估计 Sim(3)。这张表只能作为论文协议复现表；本项目的部署导向主表必须另报 raw metric、固定 prefix 和长时 stitching。

## 5. ReViV 优先的微调方案

### 5.1 为什么先调 ReViV

- 任务最匹配：RGB 直接预测 camera trajectory，同时建模 body、hands 和 gaze，正面覆盖 ego 视频中的移动人体和手动作。
- 代码可训练：官方支持从 checkpoint `--finetune`，也支持 `frozen_model_epochs/tokens`，先只训练 modality embeddings，再自动解冻 shared encoder/decoder。
- 相机表示明确：官方 `CamTrajDataset.canonicalize()` 使用 `inv(T0) @ Ti`，再把旋转前两列 6D representation 与 translation 拼成每帧 9D。
- 可测清楚瓶颈：相机 VQ-VAE、主 transformer、窗口拼接可以分层评估，不必把所有误差归因于一个黑盒。

EgoM2P 使用同类 60x9 camera tokenizer，适合作为同数据、同窗口、同坐标预处理下的公平对照。

### 5.2 数据统一与防泄漏

所有训练样本先转为：

```text
timestamp_ns
image_rgb                 # 真彩色，不接受灰度复制
T_world_camera[4,4]       # OpenCV camera-to-world，单位 meter
K[3,3]
distortion_model + coeffs
reference_grade           # A / B_device / B_kinematic / C
subject scene task robot_config session_id
```

相机标签变换固定为：

```text
T_rel[i] = inv(T_world_camera[0]) @ T_world_camera[i]
camera_9d = concat(rotation_6d(T_rel), translation(T_rel))
```

转换前先把 source 的 W2C/C2W、OpenGL/OpenCV 轴和 device-camera 外参写入 adapter metadata。机器人腕部必须使用每台 robot/camera 的手眼标定，不能把 gripper pose 原样当 camera pose。

建议训练来源：

- HoloAssist 与 EgoBody：真人头戴、手和交互；按 subject/scene 留出验证集；
- RH20T：机器人腕部、操作和近景手/夹爪；按 robot config/task 留出；
- Stera-10M：access 已获批；当前 4 个 evaluation session 封存不用作训练，后续训练只能从其余 session 按 contributor/environment 单独划分；
- perspective robot simulator：直接输出精确 C2W，覆盖头部、腕部、快速转动、移动人和物体；相机配置必须锁定为非鱼眼。

严格最终测试的 Oxford-IHM、OpenLORIS-office、Bonn、Princeton365 和 TUM 不进入微调。若未来决定用其中一部分训练，必须重建未接触的 final test split，并在表中标明 in-domain。

ReViV released weights 已使用 Ego-Exo4D、HoloAssist、HOT3D、ARCTIC、TACO、H2O、EgoGen 和 Nymeria。相应结果只能写 `pretrained-domain`，不能写 zero-shot；ADT 被论文明确排除在训练之外，但它仍是鱼眼矫正赛道。

### 5.3 先做 camera tokenizer ceiling

ReViV 不直接回归连续 pose，而是先把 camera trajectory 量化成 30 个离散 token，codebook size 为 256。微调主模型前必须回答“目标机器人轨迹是否已经超出 camera tokenizer 的表示能力”：

1. 用官方 camera tokenizer 编码并立即解码 HoloAssist、RH20T、OpenLORIS/Oxford 的独立 validation sequence 标签；strict final-test sequence 始终封存。
2. 在不看 RGB 的情况下计算 raw/RPE/rotation reconstruction error。
3. 这就是主模型不可能超过的量化上限。
4. 若快速机器人转动或腕部小范围平移的重建误差已经过大，先微调 camera VQ-VAE、重新计算 mean/std 和 tokens，再训练主模型。

不做这一步，主模型 fine-tune 失败时无法区分是视觉预测不准，还是 256-entry camera codebook 本身表达不了目标分布。

### 5.4 分阶段训练

| 阶段 | 设置 | 目的 |
|---|---|---|
| FT-0 | released ReViV 256/512、未做目标域微调；RGB -> camera only | 建立真实起点；逐数据集标记 `zero-shot` 或 `pretrained-domain`，并比较两个 checkpoint 的 camera 输出 |
| FT-1 | tokenizer encode-decode oracle | 测 camera representation ceiling |
| FT-2 | `--finetune` + 只含 RGB/camera 的 data config；设置 `frozen_model_epochs`，冻结 shared transformer | 先适配 RGB/camera embeddings，降低小数据灾难性遗忘 |
| FT-3 | 低学习率解冻 shared parameters；保存每个 epoch 的 strict validation | 学习机器人与透视镜头域差异 |
| FT-4 | 若 FT-1 不合格，微调 camera tokenizer 后重做 tokens，再重复 FT-2/3 | 修正轨迹 codebook 和归一化分布 |
| FT-5 | 相同数据与 split 对 EgoM2P post-train | 判断收益来自 ReViV 架构还是数据适配 |

官方 reference pretraining 使用 256 GPUs 和 500B tokens，不应从头复现。这里做 checkpoint adaptation；先以单节点 smoke 确认数据和 loss，再按显存扩展。每次实验记录有效 batch、gradient accumulation、学习率、训练 tokens、checkpoint SHA、CUDA/PyTorch 和峰值显存。

### 5.5 长视频处理

ReViV/EgoM2P 原生只验证 2 秒局部轨迹，不能直接宣称长时 SLAM。长视频实验需要单列：

1. 使用 2 秒窗口、1 秒 overlap；每个窗口内部保持第一帧 canonicalization。
2. 只利用 overlap 内两段**预测轨迹**估计 SE(3)/Sim(3) 关系，不读取 GT 拼接。
3. 把相邻窗口约束加入 pose graph；无可靠 overlap 时输出断链，不用 GT 或线性插值补齐。
4. 分别报告 native 2 s、20-30 s、2-3 min 和 10 min+；记录 boundary RPE、scale jump 和 time-to-failure。
5. ReViV 使用完整 2 秒上下文，属于固定窗口/有 look-ahead 的方法；不能与 causal streaming 方法混报 latency。

## 6. 统一 Evaluation 协议

### 6.1 三个正交分组

每个结果必须同时带三个标签：

| 轴 | 取值 |
|---|---|
| 图像 | `native-perspective-color` / `rectified-fisheye-color` / `grayscale-diagnostic` |
| 参考 | `A-external` / `B-device` / `B-kinematic` / `C-reconstructed` / `A2-synthetic` |
| 运行 | `causal` / `fixed-window` / `offline-global` |

只在同一个单元内做排名。不得把 A/B/C、原生/矫正或 causal/offline 的误差平均成一个总分。

### 6.2 输入公平性

- 主赛道只给 RGB；数据集自带 depth、IMU、device pose 和 robot state 只作 GT/reference，不得进入方法输入。
- `RGB + calibrated K` 与 `RGB-only/unknown-K` 分赛道。需要 K 的 ORB/DROID 与自行估 K 的方法可以同表，但输入条件必须有列。
- 同一 sequence 的所有方法读取同一 timestamp manifest；不强制所有方法使用相同分辨率，但报告实际 resize/crop、有效 FOV 和帧率。
- 标准 radtan 镜头使用官方标定去畸变；Aria 鱼眼只在单独 rectified track 使用同一固定 remap。任何方法不得单独选择更有利的 crop/FOV。
- rectified 图像保存 `virtual_K` 和 valid-pixel mask。全黑无效边界不得参与 hand/dynamic occupancy 统计。

### 6.3 Pose 与对齐

统一输出 OpenCV C2W：

```text
timestamp_ns, frame_id, T_world_camera[4,4], valid, confidence
```

同时保留四种结果：

| 协议 | 做法 | 定位 |
|---|---|---|
| `Initial-SE3/Raw-Metric` | 只统一第一帧世界原点和朝向，不缩放 | metric 方法的主结果 |
| `Prefix-Sim3@5s` | 只用前 5 s GT 估计一次 Sim(3)，之后冻结 | scale-free 方法主要 calibrated 结果；仍需外部初始化 |
| `Prefix-Sim3@10s` | 同上，前 10 s | 检查 prefix 长度敏感性 |
| `Oracle-Sim3` | 用整段 GT | 仅作轨迹形状上限和论文协议复现 |

prefix 运动不足时标记 `prefix-degenerate`，不能自动改用整段 GT。2 秒 ReViV 文献复现只报 `Oracle-Sim3`；长视频主表才使用 5/10 秒 prefix。

### 6.4 指标与失败计分

每条 sequence 至少输出：

- translation ATE median/P95/RMSE 与逐帧 rotation geodesic error；
- RPE translation/rotation @ 1 s、5 s、10 s；
- raw path-length scale ratio、`abs(log scale ratio)`、scale drift/min；
- final drift、heading drift、stationary position/rotation jitter；
- output coverage、accurate coverage @ `5 cm/5°` 和 `10 cm/10°`；
- initialization time、首次失跟时间、reset/loss count、relocalization time、sequence success；
- wall time、time-to-first-pose、FPS、峰值 VRAM；causal 方法另报 P50/P95 latency；
- 窗口方法的 boundary excess RPE、discontinuity/min 和 scale jump。

崩溃、OOM、空轨迹和失跟必须保留在 coverage/failure 表。精度统计不能先删除失败序列。

### 6.5 动态与手部切片

动态标签只用于分层统计，不作为主模型输入：

- `hand occupancy`：可见手 mask 占有效图像面积 `<10% / 10-30% / >30%`；
- `moving foreground ratio`：移动人/物 mask 或相对背景运动的面积比例；
- `occlusion burst`：连续高占比动态前景持续时间；
- blur、low texture、low light、rapid rotation、stationary、locomotion；
- duration：`2 s / 20-30 s / 2-3 min / 10 min+`。

HOT3D/HoloAssist 有官方手部信息时优先使用；其他数据可用冻结的 segmentation/flow pipeline 离线打标签。用于打标签的 mask 不能偷偷喂给普通 baseline；HaWoR 的 predicted hand mask 是其方法组成部分，应在输入列注明。

### 6.6 统计

- 先在每条 sequence 内聚合，再按数据集取 median/mean；不按帧数给长视频更大权重。
- 以原始 sequence 为 bootstrap 单位，报告 95% confidence interval。
- subject/session/scene 不得跨 bootstrap group 或训练/测试折。
- 除每个数据集结果外，只对相同 image/reference/runtime 级别做 macro-average。

## 7. 实验矩阵

### 7.1 执行顺序

| 实验 | 内容 | 通过条件 |
|---|---|---|
| E0 Adapter gate | TUM、Bonn、OpenLORIS、Princeton365 各 1 条；检查图像颜色、镜头模型、时间戳、C2W 与相机中心轨迹 | GT 投影/坐标单测通过；pose coverage >=99%；无 W2C/C2W 或轴翻转 |
| E1 Zero-shot core | 6 个最小模型跑 CI 16 clips | 所有失败均可复现并入表；无方法读取 GT/depth/IMU |
| E2 Dynamic/hand | Bonn dynamic/static + HoloAssist/RH20T held-out；DROID、HaWoR、MegaSaM、ViPE、ReViV | 产出 hand/dynamic 分桶和相对 static degradation |
| E3 ReViV ceiling | camera tokenizer 在各目标域 encode-decode | 决定只调主模型还是连 camera tokenizer 一起调 |
| E4 Fine-tune | FT-0 至 FT-5，按 subject/scene/task/robot split | 验证集选择 checkpoint，final test 始终封存 |
| E5 Long horizon | DA3-Streaming、LingBot、VGGT-SLAM、ViPE、ReViV/EgoM2P stitching | 同时报 drift、survival、boundary 和资源成本 |
| E6 Locked final | A/B/rectified 三套冻结 manifest 上只运行一次最终 checkpoint | 生成最终表、CI、失败清单和版本清单 |

### 7.2 必做消融

1. ReViV released/no-target-finetune vs frozen-shared adaptation vs full adaptation；HoloAssist 必须标 `pretrained-domain`。
2. ReViV 256 vs 512 checkpoint。
3. `human only / robot only / human + robot / human + robot + synthetic` 训练来源。
4. 原 camera tokenizer vs target-adapted tokenizer。
5. native 2 s vs prediction-only overlap stitching。
6. DROID vs DROID + predicted hand mask（HaWoR camera module）。
7. dynamic vs matched static control，避免把数据集差异误解成动态物体影响。
8. raw metric vs prefix-5s vs prefix-10s vs oracle，直接量化对齐带来的乐观程度。

### 7.3 建议验收门槛

以下是项目 go/no-go 门槛，不是社区统一标准：

- 数据 adapter：时间匹配 coverage >=99%，所有相机模型和外参来源可追溯；不合格数据不得进入主榜。
- Fine-tune：在 held-out robot/device B 表的 rotation RPE@1s 或 ATE median 至少改善 10%，output coverage 下降不超过 2 percentage points。
- 泛化保护：在完全未训练的 A 级 native-perspective 数据上，核心误差退化不超过 5%；否则把模型标成 domain-specific，而不是通用提升。
- 长视频：不能只给成功片段平均值；必须报告 100% sequence 的 success/coverage/time-to-failure。
- 论文主张：至少两个外部真值来源、一个机器人来源、一个强动态来源都支持同一趋势后，才写“对机器人 ego 有效”。

## 8. 最终结果表模板

### 表 A：原生彩色透视 + 外部真值

| Method | Input/K | Runtime class | Raw ATE cm | Raw rot ° | RPE-t/r @1s | Prefix-5s ATE | Coverage | Fail % | FPS/VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 待运行 | | | | | | | | | |

Oxford-IHM、OpenLORIS-office、Bonn、Princeton365 和 TUM 分数据集出行；机器人 macro-average 只含通过准入的 Oxford-IHM 与 OpenLORIS-office。

### 表 B：设备/运动学参考

| Method | Dataset/ref | ATE cm | rot ° | RPE @1/5/10s | Coverage | Drift/min | Fail % |
|---|---|---:|---:|---:|---:|---:|---:|
| 待运行 | | | | | | | |

表头必须写 `device-reference` 或 `kinematic-reference`，不能缩写成 GT。

### 表 C：动态和手遮挡

| Method | Static | Dynamic-low | Dynamic-high | Hand <10% | Hand 10-30% | Hand >30% | Coverage delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 待运行 | | | | | | | |

每格报告 rotation RPE@1s 与 accurate coverage，另附相对 matched-static degradation。

### 表 D：ReViV 微调

| Variant | Camera tokenizer | Train sources | A-native | B-head | B-robot wrist | Long stitching | Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| released | original | released pretrain | | | | | |
| frozen shared | original | | | | | | |
| full FT | original | | | | | | |
| tokenizer + full FT | adapted | | | | | | |

## 9. 项目落地修改清单

### 已完成

1. 固定 `native_rgb` 与 `robot_interaction_rgb` 两个 manifest；真实 sequence、相机流、窗口、reference grade 和来源 revision 均已写入。
2. TUM、Bonn、OpenLORIS、DROID、HoloAssist、RH20T、Stera 共 28 个 clip 已统一生成 RGB frames、时间戳、相机 metadata、动态 reference 和逐文件 hash，并通过 strict verify。
3. `core65` 已在统一资源清单中标为 legacy mixed-input；仓库内保留 112 个评测 clip，删除仓库外 177,344,813,322-byte 中间目录（含 153 GB 可重建下载缓存和重复输出）。
4. 13 个源码仓库已经成为 `thirdparty/` submodule；88 个 checkpoint 文件和 ORB vocabulary 已验证，checkpoint 缺失数为 0。
5. RH20T 已实现真实时间采样、MP4 frame-index 解码、TCP/手眼变换、标定方向闭环和整包 SHA-256 gate。
6. Stera 已固定 4 个 session：按 MP4 frame index 对齐同 index ARKit pose，组合 optical-to-link 外参，要求 `normal` tracking 并避开 timestamp pause；完整 MP4/HDF5 已删除。

### P0：下一轮 GPU evaluation

1. 新增统一 method output adapter：先 DA3-1.1、DROID、ViPE、ReViV、EgoM2P，再接 VGGT-SLAM/LingBot/MegaSaM/HaWoR/ORB-SLAM3。
2. 把 evaluator 从 EgoBody 专用 pipeline 中拆出，统一 raw/prefix/oracle、多尺度 RPE、失败和资源记录。
3. 在 GPU 机建立彼此隔离的模型环境；不要在当前 CPU 机编译 CUDA 或把依赖混进一个环境。
4. 先跑 adapter smoke test，再跑 28-clip pilot；所有失败都进入 denominator，禁止只保留成功序列。

### P1：微调与长视频

1. 在 GPU 机验证已补齐的 Cosmos tokenizer 与 ReViV metric-depth 512 路径，资源完整不能替代运行验证。
2. 实现 camera tokenizer ceiling 测试和 RH20T/HoloAssist/Stera tokenization。
3. 实现 prediction-only overlap stitching 与 boundary metrics。
4. 固化 train/val/final-test manifest 和训练 provenance；本轮 28 个 pilot 默认封存为 evaluation，不进入微调训练。

### P2：论文级报告

1. 完成 dynamic/hand/blur/texture/duration 分层。
2. 对每条 sequence 生成 failure trace、轨迹图和可复现命令。
3. 使用 sequence-level bootstrap CI，生成 A/B/rectified 三套表。
4. 最终报告同时发布 checkpoint hash、代码 commit、数据版本、镜头/矫正参数和所有失败样本。

## 10. 当前最合理的决策

近期不要在 112 条 mixed-input profile 上直接跑完整模型矩阵，也不要用 28 个
pilot clip 宣称统计排名。当前合理顺序是：

1. 在 GPU 机先接 DA3-1.1、DROID、ViPE、ReViV、EgoM2P 的统一输出和失败协议，用两个新 profile 做 adapter smoke/CI。
2. Oxford-IHM 已申请并等待审批；收到数据后以 RGB calibration 与 mocap 外参 gate 决定是否进入机器人 headline。Stera-10M 的 4 段短窗已冻结，长时 survival 子集另行设计，不能直接把全量拉入当前 profile。
3. 扩充 A/B profile 到按 sequence/subject/task 可 bootstrap 的规模，再冻结 final test；现有 pilot 不作为训练数据回流。
4. 先做 ReViV released checkpoint 与 camera tokenizer ceiling，再开始 fine-tune；EgoM2P 使用相同 train/val split 作直接对照。HoloAssist 结果始终标记 pretrained-domain。
5. ADT/HOT3D/InCrowd 统一矫正后进入 appendix，Monado/LaMAria 保留为灰度诊断；最终主结论只来自外部真值、原生彩色透视数据。

## 11. 一手资料

- [ReViV official code](https://github.com/lvsean/reviv4d)；[paper](https://arxiv.org/abs/2607.17790)
- [EgoM2P official code](https://github.com/ligengen/EgoM2P)
- [HaWoR official code](https://github.com/ThunderVVV/HaWoR)
- [Oxford-IHM official page](https://ori.ox.ac.uk/publications/datasets/oxford-indoor-human-motion-dataset-2024)；[access page](https://ori-arg.github.io/oxford-indoor-human-motion-dataset/downloads/)
- [OpenLORIS-Scene official page](https://lifelong-robotic-vision.github.io/dataset/scene.html)
- [Bonn RGB-D Dynamic official page](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)；[ReFusion paper](https://arxiv.org/abs/1905.02082)
- [TUM RGB-D official page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [Princeton365 project](https://princeton365.cs.princeton.edu/)
- [HoloAssist project](https://holoassist.github.io/)；[data format](https://holoassist.github.io/data_links/README.html)
- [Stera-10M data card](https://huggingface.co/datasets/fpvlabs/stera-10m)
- [RH20T project](https://rh20t.github.io/)
- [Project Aria camera models](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models)
