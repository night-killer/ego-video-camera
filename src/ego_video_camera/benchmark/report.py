from __future__ import annotations

import csv
import json
import struct
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..serialization import read_json, to_jsonable, write_json
from .schema import RunSpec, RunStatus


CORE_METRICS = (
    "primary_ate_m_rmse",
    "primary_rotation_deg_rmse",
    "primary_rpe_translation_m_1p0s_rmse",
    "primary_rpe_rotation_deg_1p0s_rmse",
    "primary_final_position_drift_m",
    "primary_final_rotation_drift_deg",
    "primary_accurate_coverage_10cm_10deg",
    "robustness_output_coverage",
    "robustness_scorable_coverage",
    "scale_abs_log_scale_error",
    "scale_scale_drift_abs_log",
    "telemetry_wall_time_sec",
    "telemetry_time_to_first_prediction_sec",
    "telemetry_peak_cpu_ram_mb",
    "telemetry_peak_gpu_vram_mb",
    "telemetry_peak_temporary_disk_mb",
)


PLOT_SPECS = (
    {
        "artifact": "png",
        "filename": "benchmark_metrics.png",
        "metric": "primary_ate_m_rmse",
        "xlabel": "ATE RMSE (m, lower is better)",
        "title": "Absolute Trajectory Error",
    },
    {
        "artifact": "rotation_plot",
        "filename": "benchmark_rotation_rmse.png",
        "metric": "primary_rotation_deg_rmse",
        "xlabel": "Rotation RMSE (deg, lower is better)",
        "title": "Rotation Error",
    },
    {
        "artifact": "rpe_1s_plot",
        "filename": "benchmark_rpe_1s.png",
        "metric": "primary_rpe_translation_m_1p0s_rmse",
        "xlabel": "RPE 1 s RMSE (m, lower is better)",
        "title": "1-second Relative Translation Error",
    },
)


def _read_dict(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _add_scalars(target: dict[str, Any], prefix: str, source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        number = _number(value)
        if number is not None:
            target[f"{prefix}_{key}"] = number


def canonical_grade(config: dict[str, Any], grade: str) -> str:
    aliases = config["datasets"].get("grade_aliases", {})
    return str(aliases.get(grade, grade))


def run_metric_rows(
    config: dict[str, Any], runs: Iterable[RunSpec]
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        state = _read_dict(run.output_dir / "run.json") or {}
        evaluation = _read_dict(run.output_dir / "evaluation.json")
        telemetry = _read_dict(run.output_dir / "telemetry.json") or state.get(
            "telemetry", {}
        )
        run_status = str(state.get("status", RunStatus.PENDING.value))
        evaluation_state = state.get("evaluation") or {}
        evaluation_status = "pending"
        if (
            evaluation_state.get("status") == "failed"
            or run_status == RunStatus.EVALUATION_FAILED.value
        ):
            evaluation_status = "failed"
        elif (
            run_status == RunStatus.SUCCESS.value
            and evaluation_state.get("status") == "success"
            and evaluation is not None
        ):
            evaluation_status = "success"
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "method_id": run.method.method_id,
            "method_display_name": run.method.display_name,
            "canonical": run.method.canonical,
            "dataset_id": run.sequence.dataset_id,
            "sequence_id": run.sequence.sequence_id,
            "sequence_key": run.sequence.key,
            "reference_grade": canonical_grade(config, run.sequence.reference_grade),
            "seed": run.seed,
            "run_status": run_status,
            "evaluation_status": evaluation_status,
            "primary_protocol": None,
        }
        if evaluation_status == "success" and evaluation is not None:
            primary = evaluation.get("primary_protocol")
            row["primary_protocol"] = primary
            protocol = evaluation.get("protocols", {}).get(str(primary), {})
            _add_scalars(row, "primary", protocol.get("metrics"))
            _add_scalars(row, "robustness", evaluation.get("robustness"))
            _add_scalars(row, "scale", evaluation.get("scale"))
            _add_scalars(row, "confidence", evaluation.get("confidence"))
        _add_scalars(row, "telemetry", telemetry)
        rows.append(row)
    return rows


def _metric_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    prefixes = ("primary_", "robustness_", "scale_", "confidence_", "telemetry_")
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith(prefixes) and _number(value) is not None
        }
    )


def sequence_metric_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["method_id"]), str(row["sequence_key"]))].append(row)
    metrics = _metric_columns(run_rows)
    output = []
    for key in sorted(grouped):
        rows = grouped[key]
        successful = [row for row in rows if row["evaluation_status"] == "success"]
        failures = [
            row
            for row in rows
            if row["run_status"]
            not in {RunStatus.PENDING.value, RunStatus.SUCCESS.value}
        ]
        if len(successful) == len(rows):
            status = "complete"
        elif successful:
            status = "partial"
        elif failures:
            status = "failed"
        else:
            status = "pending"
        first = rows[0]
        protocols = sorted(
            {str(row["primary_protocol"]) for row in successful if row["primary_protocol"]}
        )
        aggregate: dict[str, Any] = {
            "method_id": first["method_id"],
            "method_display_name": first["method_display_name"],
            "canonical": first["canonical"],
            "dataset_id": first["dataset_id"],
            "sequence_id": first["sequence_id"],
            "sequence_key": first["sequence_key"],
            "reference_grade": first["reference_grade"],
            "status": status,
            "planned_seed_count": len(rows),
            "successful_seed_count": len(successful),
            "primary_protocol": ",".join(protocols) if protocols else None,
        }
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in successful if _number(row.get(metric)) is not None],
                dtype=np.float64,
            )
            aggregate[metric] = float(values.mean()) if len(values) else None
            aggregate[f"{metric}_seed_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
            )
        output.append(aggregate)
    return output


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    *,
    samples: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float | None, float | None, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None, None, None
    mean = float(array.mean())
    if len(array) == 1 or samples <= 0:
        return mean, mean, mean
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def leaderboard_rows(
    sequence_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped_sequences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_runs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sequence_rows:
        grouped_sequences[(str(row["method_id"]), str(row["reference_grade"]))].append(row)
    for row in run_rows:
        grouped_runs[(str(row["method_id"]), str(row["reference_grade"]))].append(row)
    metrics = _metric_columns(run_rows)
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    for key in sorted(grouped_sequences):
        sequences = grouped_sequences[key]
        runs = grouped_runs[key]
        successful_sequences = [row for row in sequences if row["successful_seed_count"] > 0]
        complete_sequences = [row for row in sequences if row["status"] == "complete"]
        successful_runs = [row for row in runs if row["evaluation_status"] == "success"]
        if len(complete_sequences) == len(sequences):
            status = "complete"
        elif successful_sequences:
            status = "partial"
        elif any(row["status"] == "failed" for row in sequences):
            status = "failed"
        else:
            status = "pending"
        first = sequences[0]
        protocols = sorted(
            {
                str(row["primary_protocol"])
                for row in successful_sequences
                if row["primary_protocol"]
            }
        )
        aggregate: dict[str, Any] = {
            "method_id": first["method_id"],
            "method_display_name": first["method_display_name"],
            "canonical": first["canonical"],
            "reference_grade": first["reference_grade"],
            "status": status,
            "planned_sequence_count": len(sequences),
            "successful_sequence_count": len(successful_sequences),
            "planned_run_count": len(runs),
            "successful_run_count": len(successful_runs),
            "run_success_rate": len(successful_runs) / len(runs) if runs else 0.0,
            "primary_protocol": ",".join(protocols) if protocols else None,
            "rank": None,
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in successful_sequences
                if _number(row.get(metric)) is not None
            ]
            mean, low, high = bootstrap_mean_ci(
                values, samples=bootstrap_samples, rng=rng
            )
            aggregate[metric] = mean
            aggregate[f"{metric}_ci_low"] = low
            aggregate[f"{metric}_ci_high"] = high
            seed_stds = [
                float(row[f"{metric}_seed_std"])
                for row in successful_sequences
                if _number(row.get(f"{metric}_seed_std")) is not None
            ]
            aggregate[f"{metric}_mean_seed_std"] = (
                float(np.mean(seed_stds)) if seed_stds else None
            )
        output.append(aggregate)

    for grade in sorted({str(row["reference_grade"]) for row in output}):
        rankable = [
            row
            for row in output
            if row["reference_grade"] == grade
            and row["canonical"]
            and row["status"] == "complete"
            and _number(row.get("primary_ate_m_rmse")) is not None
        ]
        rankable.sort(key=lambda row: float(row["primary_ate_m_rmse"]))
        for rank, row in enumerate(rankable, 1):
            row["rank"] = rank
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return "pending"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _format_value(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "pending" if number is None else f"{number:.{digits}f}"


def _format_ci(row: dict[str, Any], metric: str, digits: int = 3) -> str:
    value = _number(row.get(metric))
    low = _number(row.get(f"{metric}_ci_low"))
    high = _number(row.get(f"{metric}_ci_high"))
    if value is None:
        return "pending"
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _markdown(
    payload: dict[str, Any], leaderboard: list[dict[str, Any]], status_counts: Counter[str]
) -> str:
    lines = [
        "# Ego RGB 相机位姿 Benchmark 指标报告",
        "",
        f"生成时间：`{payload['generated_at']}`",
        "",
        f"计划运行 `{payload['planned_run_count']}` 项，已完成评测 `{payload['evaluated_run_count']}` 项。"
        " 指标先在同一 sequence 内聚合 seed，再跨 sequence 求均值；置信区间采用 sequence bootstrap。",
        "",
        "## 运行状态",
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    if not status_counts:
        lines.append("| pending | 0 |")

    lines.extend(
        [
            "",
            "## 参考轨迹等级",
            "",
            "A、B1、B2 表示参考轨迹的来源和可信度，不表示片段难度；不同等级分别统计，不混合排名。",
            "",
            "| 等级 | 参考来源 | 是否严格 GT | 本 benchmark 数据集 |",
            "|---|---|---|---|",
            "| A | 外部 mocap、高精度外部定位或公开数据集提供的独立真值 | 是或可作为高质量独立真值 | Princeton365、TUM RGB-D、Bonn RGB-D Dynamic、OpenLORIS Office |",
            "| B1 | HoloLens tracking、ARKit 等设备自身的 VIO/跟踪轨迹 | 否，属于 device/VIO reference | EgoBody、HoloAssist、Stera10M |",
            "| B2 | 机器人正运动学结合手眼标定得到的相机轨迹 | 否，属于 kinematic reference | DROID wrist、RH20T wrist |",
            "",
            "## 三项主要误差",
            "",
            "三项均为误差，都是越低越好。误差棒是按 sequence bootstrap 得到的 95% 置信区间；图中只包含结果完整的正式方法。",
            "",
            "- **ATE RMSE**：全局绝对位置误差，反映整体轨迹精度和累计漂移。",
            "- **旋转 RMSE**：逐帧相机朝向的角度误差。",
            "- **RPE 1s RMSE**：相隔 1 秒两帧间的相对平移误差，反映短期运动估计和局部漂移。",
            "",
            "### ATE RMSE",
            "",
            "![ATE RMSE by reference grade](benchmark_metrics.png)",
            "",
            "### 旋转 RMSE",
            "",
            "![Rotation RMSE by reference grade](benchmark_rotation_rmse.png)",
            "",
            "### RPE 1s RMSE",
            "",
            "![RPE 1s RMSE by reference grade](benchmark_rpe_1s.png)",
        ]
    )

    grade_order = ["A", "B1", "B2"]
    extras = sorted({str(row["reference_grade"]) for row in leaderboard} - set(grade_order))
    for grade in grade_order + extras:
        grade_rows = [row for row in leaderboard if row["reference_grade"] == grade]
        if not grade_rows:
            continue
        for canonical, label in ((True, "正式方法"), (False, "消融方法")):
            rows = [row for row in grade_rows if bool(row["canonical"]) == canonical]
            if not rows:
                continue
            rows.sort(
                key=lambda row: (
                    row.get("rank") is None,
                    row.get("rank") or 999,
                    str(row["method_id"]),
                )
            )
            lines.extend(
                [
                    "",
                    f"## {grade} 榜单：{label}",
                    "",
                    "| 排名 | 方法 | 状态 | 序列 | 成功 run | Primary | ATE RMSE m (95% CI) | 旋转 RMSE deg | RPE 1s m | Coverage | Wall s | VRAM MB | Seed ATE std |",
                    "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows:
                rank = row.get("rank") if canonical else "-"
                sequence_count = (
                    f"{row['successful_sequence_count']}/{row['planned_sequence_count']}"
                )
                run_count = f"{row['successful_run_count']}/{row['planned_run_count']}"
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(rank if rank is not None else "pending"),
                            str(row["method_display_name"]).replace("|", "\\|"),
                            str(row["status"]),
                            sequence_count,
                            run_count,
                            str(row.get("primary_protocol") or "pending"),
                            _format_ci(row, "primary_ate_m_rmse"),
                            _format_value(row.get("primary_rotation_deg_rmse")),
                            _format_value(row.get("primary_rpe_translation_m_1p0s_rmse")),
                            _format_value(row.get("robustness_output_coverage")),
                            _format_value(row.get("telemetry_wall_time_sec"), 1),
                            _format_value(row.get("telemetry_peak_gpu_vram_mb"), 1),
                            _format_value(
                                row.get("primary_ate_m_rmse_mean_seed_std")
                            ),
                        )
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- Metric-scale 方法的 primary protocol 为 initial-SE3；非 metric-scale 方法为 5 秒 Prefix-Sim3。",
            "- `pending` 表示尚无真实评测结果；失败、超时和 OOM 不会被当作零误差参与均值。",
            "- CSV 中保留逐 run、逐 sequence 和榜单三级数据；JSON 保留完整结构与 bootstrap 配置。",
            "",
        ]
    )
    return "\n".join(lines)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _placeholder_png(path: Path, width: int = 960, height: int = 320) -> None:
    scanline = b"\x00" + b"\xff\xff\xff" * width
    data = scanline * height
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(data, 9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def write_plot(
    path: Path,
    leaderboard: list[dict[str, Any]],
    *,
    metric: str = "primary_ate_m_rmse",
    xlabel: str = "ATE RMSE (m, lower is better)",
    title: str = "Absolute Trajectory Error",
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        _placeholder_png(path)
        return f"matplotlib unavailable: {error}"

    grades = ["A", "B1", "B2"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, grade in zip(axes, grades):
        rows = [
            row
            for row in leaderboard
            if row["reference_grade"] == grade
            and row["canonical"]
            and row["status"] == "complete"
            and _number(row.get(metric)) is not None
        ]
        rows.sort(key=lambda row: float(row[metric]), reverse=True)
        if not rows:
            axis.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=16)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"Grade {grade}")
            continue
        values = np.asarray([row[metric] for row in rows], dtype=float)
        low = np.asarray([row[f"{metric}_ci_low"] for row in rows], dtype=float)
        high = np.asarray([row[f"{metric}_ci_high"] for row in rows], dtype=float)
        errors = np.maximum(0.0, np.vstack((values - low, high - values)))
        positions = np.arange(len(rows))
        colors = ["#2f6f6d" if index % 2 == 0 else "#b85c38" for index in positions]
        axis.barh(positions, values, xerr=errors, color=colors, alpha=0.9, capsize=2)
        axis.set_yticks(positions, [str(row["method_display_name"]) for row in rows], fontsize=8)
        axis.set_xlabel(xlabel)
        axis.set_title(f"Grade {grade}")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(f"Ego RGB Camera Pose Benchmark: {title}")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return None


def generate_report(
    config: dict[str, Any], runs: Iterable[RunSpec], *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    run_list = list(runs)
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(config["benchmark"]["output_root"]) / "report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    runs_rows = run_metric_rows(config, run_list)
    sequences_rows = sequence_metric_rows(runs_rows)
    benchmark = config["benchmark"]
    leaderboard = leaderboard_rows(
        sequences_rows,
        runs_rows,
        bootstrap_samples=int(benchmark.get("bootstrap_samples", 10000)),
        bootstrap_seed=int(benchmark.get("bootstrap_seed", 20260724)),
    )
    status_counts = Counter(str(row["run_status"]) for row in runs_rows)
    evaluated_count = sum(row["evaluation_status"] == "success" for row in runs_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_name": benchmark.get("name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": len(run_list),
        "evaluated_run_count": evaluated_count,
        "status_counts": dict(sorted(status_counts.items())),
        "bootstrap": {
            "unit": "sequence_after_seed_aggregation",
            "seed": int(benchmark.get("bootstrap_seed", 20260724)),
            "samples": int(benchmark.get("bootstrap_samples", 10000)),
            "confidence": 0.95,
        },
        "leaderboard": leaderboard,
        "sequence_metrics": sequences_rows,
        "run_metrics": runs_rows,
    }

    paths = {
        "markdown": report_dir / "metrics_report_zh.md",
        "json": report_dir / "metrics_report.json",
        "leaderboard_csv": report_dir / "leaderboard.csv",
        "sequence_csv": report_dir / "sequence_metrics.csv",
        "run_csv": report_dir / "run_metrics.csv",
    }
    for spec in PLOT_SPECS:
        paths[str(spec["artifact"])] = report_dir / str(spec["filename"])
    write_csv(paths["run_csv"], runs_rows)
    write_csv(paths["sequence_csv"], sequences_rows)
    write_csv(paths["leaderboard_csv"], leaderboard)
    plot_warnings = []
    for spec in PLOT_SPECS:
        plot_warning = write_plot(
            paths[str(spec["artifact"])],
            leaderboard,
            metric=str(spec["metric"]),
            xlabel=str(spec["xlabel"]),
            title=str(spec["title"]),
        )
        if plot_warning:
            plot_warnings.append(f"{spec['metric']}: {plot_warning}")
    if plot_warnings:
        payload["plot_warning"] = "; ".join(plot_warnings)
    paths["markdown"].write_text(
        _markdown(payload, leaderboard, status_counts), encoding="utf-8"
    )
    payload["artifacts"] = {key: str(value) for key, value in paths.items()}
    write_json(paths["json"], payload)
    return payload
