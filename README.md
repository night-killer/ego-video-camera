# ego-video-camera

从 SuperSplat Gaussian 场景手工标定 egocentric 相机轨迹，渲染 GT 视频，使用
Depth Anything 3 预测相机，再将预测轨迹对齐到场景中进行可视化和回渲染比较。

## 环境与初始化

项目默认使用已有环境和模型：

```bash
export PY=/data/aigc/cyb/zxgu/env/worldsearcher/bin/python
export DA3_MODEL=/data/aigc/cyb/zxgu/ckpt/DA3NESTED-GIANT-LARGE
git submodule update --init --recursive
$PY -m pip install -e . --no-deps
```

三个 third-party 源码均为固定 submodule：Depth Anything 3
`41736238f5bced4debf3f2a12375d2466874866d`、Spark v2.1.0 和 Three.js r180。
标定网页直接由本地文件提供 Spark/Three，不访问 CDN。

默认视频规格为 896×504、15 FPS。忠实 Gaussian 渲染使用 `gsplat`，第 2、4、5
步以及 DA3 实推均需要 NVIDIA GPU；当前 CPU 机器可运行标定服务、轨迹插值、
DA3 dry-run 和所有非 GPU 测试。

## 1. 网页标定关键帧

```bash
$PY scripts/01_annotate_camera.py \
  --ply /data/aigc/cyb/zxgu/code/WorldSearcher/data/supersplat_scene_spz/ply/0a1a3fd8.ply \
  --camera-json /data/aigc/cyb/zxgu/code/WorldSearcher/data/supersplat_scene_spz/camera/0a1a3fd8.json \
  --output outputs/0a1a3fd8/keyframes.json \
  --host 127.0.0.1 --port 7860
```

远程服务器建议通过端口转发访问：

```bash
ssh -L 7860:127.0.0.1:7860 USER@SERVER
```

浏览器打开 `http://127.0.0.1:7860`。鼠标和 WASD 调整第一视角，点击“添加当前
视角”记录关键帧。关键帧时间会吸附到 FPS 网格；可以替换、删除、调序、编辑
帧号/时间以及全局 FOV。至少保存两个关键帧，且第一个关键帧必须为第 0 帧。

脚本会优先发现同目录层级下的 `spz/0a1a3fd8.spz` 以减少浏览器下载和内存；若
不存在则直接加载给定 PLY。无论页面显示哪种格式，保存结果都转换回原始 PLY
世界坐标。

## 2. 插值并渲染 GT 视频

先在 CPU 机检查插值结果：

```bash
$PY scripts/02_interpolate_and_render.py \
  --keyframes outputs/0a1a3fd8/keyframes.json \
  --output-dir outputs/0a1a3fd8/gt \
  --trajectory-only
```

在 GPU 机渲染完整 GT 视频：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/02_interpolate_and_render.py \
  --keyframes outputs/0a1a3fd8/keyframes.json \
  --output-dir outputs/0a1a3fd8/gt \
  --device cuda:0 --overwrite
```

位置使用 cubic spline，旋转使用 `RotationSpline`；只有两个关键帧时使用线性位置
加 quaternion SLERP。输出 `gt_trajectory.json`、`gt_video.mp4` 和
`render_manifest.json`。

## 3. DA3 相机预测与 Sim(3) 对齐

CPU dry-run 会完整解码视频、核对帧数/尺寸、检查 submodule 和权重，但不加载模型：

```bash
$PY scripts/03_predict_da3_camera.py \
  --video outputs/0a1a3fd8/gt/gt_video.mp4 \
  --gt-trajectory outputs/0a1a3fd8/gt/gt_trajectory.json \
  --output-dir outputs/0a1a3fd8/da3 \
  --model-dir "$DA3_MODEL" --dry-run
```

GPU 正式命令：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/03_predict_da3_camera.py \
  --video outputs/0a1a3fd8/gt/gt_video.mp4 \
  --gt-trajectory outputs/0a1a3fd8/gt/gt_trajectory.json \
  --output-dir outputs/0a1a3fd8/da3 \
  --da3-root third_party/depth-anything-3 \
  --model-dir "$DA3_MODEL" \
  --device cuda --process-res 504 --max-frames 180
```

该步骤逐帧输入视频，不静默抽帧，调用 camera decoder、`infer_gs=False`、
`ref_view_strategy=middle`。输出原始相机 NPZ、`raw_trajectory.json`、
`aligned_trajectory.json`、Sim(3) 参数和误差报告。第一版明确限制最多 180 帧；
更长视频需要后续单独接入 DA3-Streaming。

## 4. 带 Gaussian 背景的轨迹视频

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/04_visualize_trajectories.py \
  --gt-trajectory outputs/0a1a3fd8/gt/gt_trajectory.json \
  --predicted-trajectory outputs/0a1a3fd8/da3/aligned_trajectory.json \
  --output-dir outputs/0a1a3fd8/trajectory_visualization \
  --device cuda:0
```

脚本从 GT 水平轨迹主方向自动建立两个相差 90° 的透视观察相机。每个固定背景只
做一次 Gaussian rasterization，随后降低亮度和饱和度，再累计叠加始终可见的
绿色 GT、红色 DA3 四棱锥。输出 `trajectory_principal.mp4`、
`trajectory_orthogonal.mp4`、两张背景图和观察相机 manifest。

## 5. GT / DA3 回渲染比较

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/05_render_comparison.py \
  --gt-trajectory outputs/0a1a3fd8/gt/gt_trajectory.json \
  --predicted-trajectory outputs/0a1a3fd8/da3/aligned_trajectory.json \
  --output-dir outputs/0a1a3fd8/comparison \
  --device cuda:0
```

输出包括：

- `gt.mp4`
- `predicted_full_camera.mp4`：预测外参与 DA3 预测内参
- `predicted_pose_only.mp4`：预测外参与 GT 内参
- `comparison_full_camera.mp4`：`GT | DA3 full camera`
- `comparison_pose_only.mp4`：`GT | DA3 pose only (GT intrinsics)`

单路视频为 896×504，双栏视频为 1792×504，均为 H.264/yuv420p。

## 相机格式

所有 JSON 使用 `camera_trajectory.v1`：

- 世界坐标：原始 SuperSplat PLY 世界；DA3 raw 文件会明确标记独立坐标系。
- 相机坐标：OpenCV RDF，`+X` 右、`+Y` 下、`+Z` 前。
- `camera_to_world`：4×4 row-major。
- `K`：3×3 像素内参。
- 每帧时间严格满足 `timestamp_seconds = frame_index / fps`。

DA3 的 world-to-camera extrinsics 会先求逆，再用全部帧的相机中心执行 Umeyama
Sim(3)。对于直线或静止轨迹，使用平均相机姿态确定旋转，并以最小二乘求尺度与
平移，避免未定义的绕轨迹轴旋转。

## 测试

CPU 测试：

```bash
$PY -m pytest -q
```

测试覆盖坐标往返、schema、FOV/K、插值、Sim(3) 与退化 fallback、PLY 参数转换、
H.264 编解码、网页保存 API、DA3 相机格式转换，以及不依赖 CUDA 的轨迹投影合成。
GPU rasterization 和真实 DA3 推理由上述命令完成 smoke test；`gsplat` 第一次运行
可能需要 JIT 编译 CUDA extension。

已有手工关键帧后，也可在 GPU 机一条命令执行第 2--5 步的 opt-in 集成 smoke test：

```bash
EGO_CAMERA_GPU_SMOKE_KEYFRAMES=outputs/0a1a3fd8/keyframes.json \
  $PY -m pytest -q -m "gpu and integration" tests/test_gpu_end_to_end.py
```
