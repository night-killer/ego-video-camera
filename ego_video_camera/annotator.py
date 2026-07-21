"""Local FastAPI application for browser-based camera keyframe authoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from .camera import PLY_TO_SPZ, parse_supersplat_camera
from .io_utils import PipelineInputError, read_json_object, require_input_file
from .schema import (
    PLY_WORLD_FRAME,
    CameraTrajectory,
    SceneSpec,
    load_trajectory,
    save_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATOR_HTML = PROJECT_ROOT / "web" / "annotator.html"
SPARK_ROOT = PROJECT_ROOT / "third_party" / "spark"
THREE_ROOT = PROJECT_ROOT / "third_party" / "three.js"


@dataclass(frozen=True)
class AnnotationContext:
    ply_path: Path
    camera_json_path: Path
    display_asset_path: Path
    display_asset_kind: str
    display_from_ply: list[list[float]]
    initial_camera: dict[str, Any]
    camera_radius: float | None
    output_path: Path
    scene: SceneSpec
    saved_trajectory: CameraTrajectory | None

    def browser_config(self) -> dict[str, Any]:
        return {
            "scene": self.scene.model_dump(mode="json"),
            "asset_url": "/asset/display",
            "asset_kind": self.display_asset_kind,
            "initial_camera": self.initial_camera,
            "camera_radius": self.camera_radius,
            "defaults": {"width": 896, "height": 504, "fps": 15.0},
            "saved_trajectory": (
                self.saved_trajectory.model_dump(mode="json")
                if self.saved_trajectory is not None
                else None
            ),
            "output_path": str(self.output_path),
        }


def discover_display_asset(ply_path: Path) -> tuple[Path, str]:
    source = Path(ply_path).expanduser().resolve()
    candidates: list[Path] = []
    if source.parent.name == "ply":
        candidates.append(source.parent.parent / "spz" / f"{source.stem}.spz")
    candidates.append(source.with_suffix(".spz"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve(), "spz"
    return source, "ply"


def build_annotation_context(
    *,
    ply_path: Path,
    camera_json_path: Path,
    output_path: Path,
) -> AnnotationContext:
    ply = require_input_file(ply_path, "SuperSplat PLY")
    camera_path = require_input_file(camera_json_path, "SuperSplat camera JSON")
    payload = read_json_object(camera_path)
    if payload.get("status") != "available":
        raise PipelineInputError(
            f"camera JSON status is not available: {payload.get('error') or payload.get('status')}"
        )
    display_asset, display_kind = discover_display_asset(ply)
    display_transform = PLY_TO_SPZ if display_kind == "spz" else np.eye(4, dtype=np.float64)
    if display_kind == "spz":
        declared = payload.get("coordinate_transform")
        expected = "SuperSplat [x, y, z] -> local SPZ [-x, y, -z]"
        if declared != expected:
            raise PipelineInputError(
                f"unsupported or missing SuperSplat/SPZ coordinate transform: {declared!r}"
            )
    initial_camera = parse_supersplat_camera(payload, asset_kind=display_kind)
    if display_kind == "spz":
        ply_camera = parse_supersplat_camera(payload, asset_kind="ply")
        for field in ("position", "target"):
            homogeneous = np.asarray([*ply_camera[field], 1.0], dtype=np.float64)
            expected_display = (display_transform @ homogeneous)[:3]
            if not np.allclose(initial_camera[field], expected_display, atol=1e-5):
                raise PipelineInputError(
                    f"spz_camera.{field} is inconsistent with supersplat_camera.{field} "
                    "and the declared PLY-to-SPZ transform"
                )
        if not np.isclose(
            initial_camera["fov_y_degrees"], ply_camera["fov_y_degrees"], atol=1e-6
        ):
            raise PipelineInputError("SPZ and SuperSplat cameras declare different FOV values")
    radius_value = None
    spz_camera = payload.get("spz_camera")
    if isinstance(spz_camera, dict) and isinstance(spz_camera.get("radius"), (int, float)):
        radius_value = float(spz_camera["radius"])
    scene = SceneSpec(
        scene_id=str(payload.get("resource_id") or ply.stem),
        ply_path=str(ply),
        camera_json_path=str(camera_path),
        display_asset_path=str(display_asset),
        display_asset_kind=display_kind,
        display_from_ply=display_transform.tolist(),
    )
    target = Path(output_path).expanduser().resolve()
    saved = load_trajectory(target) if target.is_file() else None
    if saved is not None and saved.trajectory_type != "keyframes":
        raise PipelineInputError(f"existing annotation is not a keyframe trajectory: {target}")
    if saved is not None and saved.coordinate_system != PLY_WORLD_FRAME:
        raise PipelineInputError(
            f"existing annotation does not use the canonical PLY world frame: {target}"
        )
    if saved is not None and saved.scene.scene_id != scene.scene_id:
        raise PipelineInputError(
            f"existing annotation belongs to scene {saved.scene.scene_id!r}, "
            f"not {scene.scene_id!r}: {target}"
        )
    return AnnotationContext(
        ply_path=ply,
        camera_json_path=camera_path,
        display_asset_path=display_asset,
        display_asset_kind=display_kind,
        display_from_ply=display_transform.tolist(),
        initial_camera=initial_camera,
        camera_radius=radius_value,
        output_path=target,
        scene=scene,
        saved_trajectory=saved,
    )


def _safe_vendor_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="vendor asset not found") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="vendor asset not found")
    return candidate


def create_annotation_app(
    context: AnnotationContext,
    *,
    html_path: Path = ANNOTATOR_HTML,
    spark_root: Path = SPARK_ROOT,
    three_root: Path = THREE_ROOT,
) -> FastAPI:
    app = FastAPI(title="Egocentric Camera Annotator", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not html_path.is_file():
            raise HTTPException(status_code=500, detail=f"annotator HTML is missing: {html_path}")
        return FileResponse(html_path, media_type="text/html")

    @app.get("/api/config", include_in_schema=False)
    def config() -> JSONResponse:
        return JSONResponse(context.browser_config())

    @app.post("/api/save", include_in_schema=False)
    def save(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        candidate = dict(payload)
        candidate["trajectory_type"] = "keyframes"
        candidate["coordinate_system"] = PLY_WORLD_FRAME
        candidate["scene"] = context.scene.model_dump(mode="json")
        try:
            trajectory = CameraTrajectory.model_validate(candidate)
        except Exception as exc:  # Pydantic includes precise field locations in str(exc).
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        save_trajectory(context.output_path, trajectory)
        return JSONResponse(
            {
                "status": "saved",
                "output_path": str(context.output_path),
                "keyframe_count": len(trajectory.frames),
            }
        )

    @app.get("/asset/display", include_in_schema=False)
    def display_asset() -> FileResponse:
        media_type = "application/octet-stream"
        return FileResponse(context.display_asset_path, media_type=media_type)

    @app.get("/vendor/spark/{relative_path:path}", include_in_schema=False)
    def spark_asset(relative_path: str) -> FileResponse:
        return FileResponse(_safe_vendor_file(spark_root, relative_path))

    @app.get("/vendor/three/{relative_path:path}", include_in_schema=False)
    def three_asset(relative_path: str) -> FileResponse:
        return FileResponse(_safe_vendor_file(three_root, relative_path))

    return app
