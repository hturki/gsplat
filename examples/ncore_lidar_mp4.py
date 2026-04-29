# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate non-interactive SimULi-style NCore LiDAR MP4s.

This script intentionally avoids intensity coloring. It exports observed/checkpoint
RGB with range-colored native/rendered LiDAR projections, plus ego-centered
bird's-eye-view point videos.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

RGB_DARK_BLUE = np.array([25, 54, 145], dtype=np.uint8)
RGB_SIMULI_GREEN = np.array([0, 190, 0], dtype=np.uint8)
LIDAR_HIT_THRESHOLD = 0.5
LIDAR_RETURN_THRESHOLD = 0.5
LIDAR_RAYDROP_THRESHOLD = 0.5
LIDAR_NEAR_PLANE_M = 0.2
LIDAR_FAR_PLANE_M = 1e10
LIDAR_RAY_DIRECTION_SCALE = 0.002
LIDAR_RENDER_MODE = "d"
_RUNTIME_IMPORTS_READY = False


def _ensure_runtime_imports() -> None:
    global _RUNTIME_IMPORTS_READY, ncore, NCoreParser, _get_nearest_lidar_frame_index
    global _load_native_range_image, build_gsplat_lidar_coeffs_from_sensor, rasterization
    global RollingShutterType, LidarModel
    if _RUNTIME_IMPORTS_READY:
        return
    import ncore as _ncore
    import ncore.data  # noqa: F401
    import ncore.data.v4  # noqa: F401
    from ncore.sensors import LidarModel as _LidarModel
    from datasets.ncore import (
        NCoreParser as _NCoreParser,
        _get_nearest_lidar_frame_index as _nearest_lidar_frame_index,
        _load_native_range_image as _load_native_range_image_impl,
        build_gsplat_lidar_coeffs_from_sensor as _build_gsplat_lidar_coeffs_from_sensor,
    )
    from gsplat import RollingShutterType as _RollingShutterType
    from gsplat.rendering import rasterization as _rasterization

    ncore = _ncore
    NCoreParser = _NCoreParser
    _get_nearest_lidar_frame_index = _nearest_lidar_frame_index
    _load_native_range_image = _load_native_range_image_impl
    build_gsplat_lidar_coeffs_from_sensor = _build_gsplat_lidar_coeffs_from_sensor
    RollingShutterType = _RollingShutterType
    LidarModel = _LidarModel
    rasterization = _rasterization
    _RUNTIME_IMPORTS_READY = True


@dataclass
class NativeLidarFrame:
    lidar_frame_idx: int
    ranges_image_m: np.ndarray
    valid_image: np.ndarray
    xyz_sensor_m: np.ndarray
    scene_points: np.ndarray
    ranges_m: np.ndarray
    ray_directions_sensor: np.ndarray
    ray_origins_scene: np.ndarray
    ray_directions_scene: np.ndarray
    lidar_coeffs: object
    T_sensor_scene: np.ndarray
    T_sensor_scene_start: np.ndarray
    T_sensor_scene_end: np.ndarray
    ray_grid_pixels: int
    ray_return_pixels: int
    ray_missing_pixels: int
    direction_error_mean: float
    direction_error_max: float


@dataclass
class RenderedLidarFrame:
    points_sensor_scene: np.ndarray
    points_sensor_m: np.ndarray
    scene_points: np.ndarray
    ranges_m: np.ndarray
    render_range_image_scene: np.ndarray
    render_range_image_m: np.ndarray
    render_alpha_image: np.ndarray
    render_raydrop_image: np.ndarray
    valid_mask: np.ndarray
    valid_fraction: float


class Mp4Writer:
    def __init__(self, path: Path, fps: float, frame_shape: tuple[int, int, int]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open MP4 writer for {path}")

    def write(self, frame_rgb: np.ndarray) -> None:
        frame = np.asarray(frame_rgb)
        if frame.dtype != np.uint8:
            raise TypeError(f"Expected uint8 frame, got {frame.dtype}")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


def _open_sequence_loader(
    meta_json_path: Path,
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    open_consolidated: bool,
) -> ncore.data.SequenceLoaderProtocol:
    return ncore.data.v4.SequenceLoaderV4(
        ncore.data.v4.SequenceComponentGroupsReader(
            [meta_json_path], open_consolidated=open_consolidated
        ),
        poses_component_group_name=poses_component_group,
        intrinsics_component_group_name=intrinsics_component_group,
        masks_component_group_name=masks_component_group,
    )


def _load_ckpt_splats(ckpt_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ckpt["splats"].items()}


def _checkpoint_step(ckpt_path: Path) -> Optional[int]:
    match = re.search(r"ckpt_(\d+)", ckpt_path.name)
    return int(match.group(1)) if match else None


def _camera_distortion_tensors(render_data, device: torch.device):
    radial_coeffs = None
    tangential_coeffs = None
    thin_prism_coeffs = None
    if render_data.radial_coeffs is not None:
        radial_coeffs = (
            torch.from_numpy(render_data.radial_coeffs).to(device).unsqueeze(0)
        )
    if render_data.tangential_coeffs is not None:
        tangential_coeffs = (
            torch.from_numpy(render_data.tangential_coeffs).to(device).unsqueeze(0)
        )
    if render_data.thin_prism_coeffs is not None:
        thin_prism_coeffs = (
            torch.from_numpy(render_data.thin_prism_coeffs).to(device).unsqueeze(0)
        )
    return (
        render_data.ftheta_coeffs,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
    )


def _extract_projection_pixels(projection) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pixels = getattr(projection, "pixels", None)
    if pixels is None:
        pixels = getattr(projection, "image_points", None)
    if pixels is None:
        raise AttributeError(f"Unsupported projection return fields: {dir(projection)}")
    valid_indices = getattr(projection, "valid_indices", None)
    if valid_indices is None:
        valid_flag = getattr(projection, "valid_flag", None)
        if valid_flag is not None:
            valid_indices = np.nonzero(np.asarray(valid_flag))[0]
    pixels_np = np.asarray(pixels).reshape(-1, np.asarray(pixels).shape[-1])
    valid_np = None if valid_indices is None else np.asarray(valid_indices).reshape(-1)
    return pixels_np, valid_np


def _project_scene_points_to_camera(
    camera_model, camtoworld: np.ndarray, scene_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(scene_points) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    T_world_sensor = np.linalg.inv(camtoworld).astype(np.float32)
    projection = camera_model.world_points_to_pixels_static_pose(
        scene_points.astype(np.float32), T_world_sensor, return_valid_indices=True
    )
    pixels, valid_indices = _extract_projection_pixels(projection)
    if valid_indices is None:
        valid_indices = np.arange(len(pixels), dtype=np.int64)
    if len(pixels) == len(scene_points):
        return pixels[valid_indices, :2], valid_indices
    if len(valid_indices) == len(pixels):
        return pixels[:, :2], valid_indices
    n = min(len(valid_indices), len(pixels))
    return pixels[:n, :2], valid_indices[:n]


def _finite_points_mask(points: np.ndarray, max_abs: float = 1e6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.isfinite(points).all(axis=1) & (np.abs(points).max(axis=1) < max_abs)


def _sanitize_points(points: np.ndarray, max_abs: float = 1e6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    keep = _finite_points_mask(points, max_abs=max_abs)
    return np.ascontiguousarray(points[keep], dtype=np.float32)


def _scene_points_to_sensor_frame(
    scene_points: np.ndarray, T_sensor_scene: np.ndarray
) -> np.ndarray:
    points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    R = np.asarray(T_sensor_scene[:3, :3], dtype=np.float32)
    t = np.asarray(T_sensor_scene[:3, 3], dtype=np.float32)
    return ((points - t[None, :]) @ R).astype(np.float32)


def _mid_sensor_to_scene_pose(
    T_sensor_scene_start: np.ndarray, T_sensor_scene_end: np.ndarray
) -> np.ndarray:
    """Use a mid-frame projection pose and END pose for rolling interpolation."""
    mid = np.eye(4, dtype=np.float32)
    mid[:3, 3] = (
        0.5 * (T_sensor_scene_start[:3, 3] + T_sensor_scene_end[:3, 3])
    ).astype(np.float32)
    if np.allclose(
        T_sensor_scene_start[:3, :3], T_sensor_scene_end[:3, :3], rtol=1e-6, atol=1e-7
    ):
        mid[:3, :3] = T_sensor_scene_start[:3, :3].astype(np.float32)
        return mid
    key_rots = Rotation.from_matrix(
        np.stack([T_sensor_scene_start[:3, :3], T_sensor_scene_end[:3, :3]], axis=0)
    )
    mid[:3, :3] = Slerp([0.0, 1.0], key_rots)([0.5]).as_matrix()[0].astype(np.float32)
    return mid


def _subsample_indices(n: int, limit: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, limit, dtype=np.int64)


def _range_color_vmax(
    ranges: np.ndarray,
    upper_quantile: float,
    fallback: float = 1.0,
) -> float:
    ranges = np.asarray(ranges, dtype=np.float32).reshape(-1)
    finite = np.isfinite(ranges) & (ranges > 0)
    if not finite.any():
        return max(float(fallback), 1e-3)
    return max(float(np.quantile(ranges[finite], upper_quantile)), 1e-3)


def _range_to_rgb(
    ranges: np.ndarray,
    upper_quantile: float,
    cmap_name: str,
    vmax: Optional[float] = None,
) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if ranges.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if vmax is None:
        vmax = _range_color_vmax(ranges, upper_quantile)
    else:
        vmax = max(float(vmax), 1e-3)
    values = (255.0 * np.clip(ranges / vmax, 0.0, 1.0)).astype(np.uint8)
    colors = cv2.applyColorMap(
        values.reshape(-1, 1),
        getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_TURBO),
    )
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)[:, 0, :]


def _point_colors(
    ranges: np.ndarray,
    mode: str,
    uniform_rgb: np.ndarray,
    upper_quantile: float,
    cmap_name: str,
    range_vmax: Optional[float] = None,
) -> np.ndarray:
    if mode == "range":
        return _range_to_rgb(ranges, upper_quantile, cmap_name, vmax=range_vmax)
    return np.repeat(uniform_rgb.reshape(1, 3), len(ranges), axis=0)


def _range_image_to_rgb(
    values: np.ndarray,
    valid_mask: np.ndarray,
    upper_quantile: float,
    cmap_name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values) & (values > 0)
    canvas = np.full(values.shape + (3,), 255, dtype=np.uint8)
    if not valid.any():
        return canvas
    vmax = float(np.quantile(values[valid], upper_quantile))
    vmax = max(vmax, 1e-3)
    scaled = (255.0 * np.clip(values[valid] / vmax, 0.0, 1.0)).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(
        scaled.reshape(-1, 1),
        getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_TURBO),
    )
    canvas[valid] = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)[:, 0, :]
    return canvas


def _error_image_to_rgb(
    errors: np.ndarray,
    valid_mask: np.ndarray,
    max_error_m: float,
) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(errors)
    canvas = np.full(errors.shape + (3,), 255, dtype=np.uint8)
    if not valid.any():
        return canvas
    vmax = max(float(max_error_m), 1e-3)
    scaled = (255.0 * np.clip(errors[valid] / vmax, 0.0, 1.0)).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(scaled.reshape(-1, 1), cv2.COLORMAP_MAGMA)
    canvas[valid] = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)[:, 0, :]
    return canvas


def _draw_projected_points(
    image_rgb: np.ndarray,
    pixels_xy: np.ndarray,
    colors_rgb: np.ndarray,
    alpha: float,
    point_radius: int,
) -> np.ndarray:
    canvas = np.asarray(image_rgb, dtype=np.uint8)
    h, w = canvas.shape[:2]
    coords = np.round(pixels_xy).astype(np.int32).reshape(-1, 2)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    n = min(len(coords), len(colors_rgb))
    coords = coords[:n]
    colors_rgb = colors_rgb[:n]
    keep = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < h)
    )
    coords = coords[keep]
    colors_rgb = colors_rgb[keep]
    if len(coords) == 0:
        return canvas.copy()

    layer = canvas.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    if point_radius <= 1:
        xs = coords[:, 0]
        ys = coords[:, 1]
        layer[ys, xs] = colors_rgb
        mask[ys, xs] = 1
    else:
        for (x, y), color in zip(coords, colors_rgb):
            cv2.circle(
                layer,
                (int(x), int(y)),
                point_radius,
                tuple(int(v) for v in color),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                mask,
                (int(x), int(y)),
                point_radius,
                1,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    mask_bool = mask.astype(bool)
    out = canvas.copy()
    out[mask_bool] = (
        alpha * layer[mask_bool].astype(np.float32)
        + (1.0 - alpha) * canvas[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return out


def _draw_bev(
    points_sensor: np.ndarray,
    canvas_size: int,
    range_m: float,
    point_radius: int,
    point_rgb: np.ndarray,
    max_points: int,
) -> np.ndarray:
    points = _sanitize_points(points_sensor)
    if max_points > 0 and len(points) > max_points:
        points = points[_subsample_indices(len(points), max_points)]
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    if len(points) == 0:
        return canvas
    scale = (canvas_size - 1) / (2.0 * range_m)
    center = (canvas_size - 1) / 2.0
    cols = np.round(center - points[:, 1] * scale).astype(np.int32)
    rows = np.round(center - points[:, 0] * scale).astype(np.int32)
    keep = (rows >= 0) & (rows < canvas_size) & (cols >= 0) & (cols < canvas_size)
    rows = rows[keep]
    cols = cols[keep]
    if point_radius <= 1:
        canvas[rows, cols] = point_rgb
    else:
        for x, y in zip(cols, rows):
            cv2.circle(
                canvas,
                (int(x), int(y)),
                point_radius,
                tuple(int(v) for v in point_rgb),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return canvas


class NCoreLidarMp4Exporter:
    def __init__(self, args: argparse.Namespace):
        _ensure_runtime_imports()
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = NCoreParser(
            meta_json_path=args.data_dir,
            factor=1.0 / args.data_factor if args.data_factor > 1 else 1.0,
            test_every=args.test_every,
            camera_ids=args.ncore_camera_ids or None,
            lidar_ids=args.ncore_lidar_ids or None,
            seek_offset_sec=args.ncore_seek_offset_sec,
            duration_sec=args.ncore_duration_sec,
            max_lidar_points=args.ncore_max_lidar_points,
            poses_component_group=args.ncore_poses_component_group,
            intrinsics_component_group=args.ncore_intrinsics_component_group,
            masks_component_group=args.ncore_masks_component_group,
            open_consolidated=args.open_consolidated,
            normalize_world_space=args.normalize_world_space,
            load_lidar_points=args.normalize_world_space,
        )
        self.sequence_loader = _open_sequence_loader(
            Path(args.data_dir),
            args.ncore_poses_component_group,
            args.ncore_intrinsics_component_group,
            args.ncore_masks_component_group,
            args.open_consolidated,
        )
        self.camera_id = args.camera_id or self.parser.camera_ids[0]
        self.lidar_id = args.lidar_id or self.parser.lidar_ids[0]
        if self.camera_id not in self.parser.camera_ids:
            raise ValueError(
                f"Camera {self.camera_id!r} is not in parser camera ids {self.parser.camera_ids}"
            )
        if self.lidar_id not in self.parser.lidar_ids:
            raise ValueError(
                f"LiDAR {self.lidar_id!r} is not in parser lidar ids {self.parser.lidar_ids}"
            )
        self.camera_sensor = self.sequence_loader.get_camera_sensor(self.camera_id)
        self.lidar_sensor = self.sequence_loader.get_lidar_sensor(self.lidar_id)
        self.lidar_coeffs = build_gsplat_lidar_coeffs_from_sensor(
            self.lidar_sensor, device=self.device
        )
        self.scene_range_scale = float(self.parser.normalization_scale)
        if not np.isfinite(self.scene_range_scale) or self.scene_range_scale <= 0:
            raise ValueError(
                f"Invalid NCore normalization scale: {self.scene_range_scale}"
            )
        self.world_target_scale = float(
            getattr(self.parser.world_global_to_scene, "target_scale", 1.0)
        )
        self._native_cache: dict[int, NativeLidarFrame] = {}
        self._rendered_cache: dict[int, RenderedLidarFrame] = {}
        self._camera_render_cache: dict[int, np.ndarray] = {}
        self._lidar_model = None
        self._ray_elements: Optional[np.ndarray] = None
        self._ray_directions_sensor: Optional[np.ndarray] = None
        self.splats = _load_ckpt_splats(Path(args.ckpt), self.device)
        self.means = self.splats["means"]
        self.quats = self.splats["quats"]
        self.scales = torch.exp(self.splats["scales"])
        self.opacities = torch.sigmoid(self.splats["opacities"])
        self.lidar_extra_signal = self.splats.get("lidar_extra_signal")
        if self.lidar_extra_signal is None:
            raise ValueError(
                "Checkpoint has no lidar_extra_signal. LiDAR export "
                "requires intensity plus raydrop extra signals."
            )
        if self.lidar_extra_signal.shape[-1] < 3:
            raise ValueError(
                "Checkpoint lidar_extra_signal must contain intensity plus two "
                f"raydrop logits; got shape {tuple(self.lidar_extra_signal.shape)}."
            )
        self.colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        self.sh_degree = int(math.sqrt(self.colors.shape[1]) - 1)
        self.ckpt_step = _checkpoint_step(Path(args.ckpt))
        self.render_dir = Path(args.ckpt).resolve().parents[1] / "renders"

    def selected_frames(self) -> list[tuple[int, int]]:
        entries = [
            (i, fidx)
            for i, (cid, fidx) in enumerate(self.parser.frame_list)
            if cid == self.camera_id
        ]
        if self.args.frame_source == "val":
            entries = [
                (i, fidx) for i, fidx in entries if i % self.parser.test_every == 0
            ]
        if self.args.frame_stride > 1:
            entries = entries[:: self.args.frame_stride]
        if self.args.max_frames > 0:
            entries = entries[: self.args.max_frames]
        if not entries:
            raise ValueError("No frames selected for export")
        return entries

    def load_camera_image(self, frame_idx: int) -> np.ndarray:
        width, height = self.parser.imsize_dict[self.camera_id]
        image = self.camera_sensor.get_frame_image_array(frame_idx)
        if self.parser.factor != 1.0:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(image[:, :, :3], dtype=np.uint8)

    @torch.no_grad()
    def render_camera_image(self, flat_index: int) -> np.ndarray:
        if flat_index in self._camera_render_cache:
            return self._camera_render_cache[flat_index]
        if self.ckpt_step is not None and flat_index % self.parser.test_every == 0:
            val_idx = flat_index // self.parser.test_every
            trainer_png = (
                self.render_dir / f"val_step{self.ckpt_step}_{val_idx:04d}.png"
            )
            if trainer_png.exists():
                canvas = np.asarray(imageio.imread(trainer_png), dtype=np.uint8)
                pred = canvas[:, canvas.shape[1] // 2 :, :3]
                self._camera_render_cache[flat_index] = pred
                return pred
        width, height = self.parser.imsize_dict[self.camera_id]
        camtoworld = (
            torch.from_numpy(self.parser.camtoworlds[flat_index])
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        Ks = (
            torch.from_numpy(self.parser.Ks_dict[self.camera_id])
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        render_data = self.parser.camera_render_data[self.camera_id]
        ftheta_coeffs, radial_coeffs, tangential_coeffs, thin_prism_coeffs = (
            _camera_distortion_tensors(render_data, self.device)
        )
        render_colors, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            packed=False,
            camera_model=render_data.camera_model,
            with_ut=True,
            with_eval3d=True,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            render_mode="RGB",
        )
        image = (
            render_colors[0, ..., :3].clamp(0, 1).detach().cpu().numpy() * 255.0
        ).astype(np.uint8)
        self._camera_render_cache[flat_index] = image
        return image

    def _lidar_model_params(self):
        model_params = getattr(self.lidar_sensor, "model_parameters", None)
        if model_params is None and hasattr(
            self.lidar_sensor, "get_lidar_model_parameters"
        ):
            model_params = self.lidar_sensor.get_lidar_model_parameters()
        if model_params is None:
            raise ValueError(
                f"LiDAR sensor {self.lidar_id!r} does not expose model parameters"
            )
        return model_params

    def _ncore_lidar_model(self):
        if self._lidar_model is None:
            model = LidarModel.maybe_from_parameters(self._lidar_model_params())
            if model is None:
                raise ValueError(
                    f"LiDAR sensor {self.lidar_id!r} has unsupported model parameters"
                )
            self._lidar_model = model
        return self._lidar_model

    def dense_lidar_elements(self) -> np.ndarray:
        if self._ray_elements is not None:
            return self._ray_elements
        n_rows = int(self.lidar_coeffs.n_rows)
        n_cols = int(self.lidar_coeffs.n_columns)
        rows, cols = np.meshgrid(
            np.arange(n_rows, dtype=np.int64),
            np.arange(n_cols, dtype=np.int64),
            indexing="ij",
        )
        self._ray_elements = np.ascontiguousarray(
            np.stack([rows, cols], axis=-1).reshape(-1, 2)
        )
        return self._ray_elements

    def dense_lidar_directions_sensor(self) -> np.ndarray:
        if self._ray_directions_sensor is not None:
            return self._ray_directions_sensor
        model_params = self._lidar_model_params()
        row_elevations = np.asarray(model_params.row_elevations_rad, dtype=np.float64)
        column_azimuths = np.asarray(model_params.column_azimuths_rad, dtype=np.float64)
        row_offsets_raw = model_params.row_azimuth_offsets_rad
        if row_offsets_raw is None:
            row_offsets = np.zeros_like(row_elevations)
        else:
            row_offsets = np.asarray(row_offsets_raw, dtype=np.float64)
        n_rows = int(self.lidar_coeffs.n_rows)
        n_cols = int(self.lidar_coeffs.n_columns)
        if row_elevations.shape != (n_rows,):
            raise ValueError(
                f"LiDAR model row count {row_elevations.shape} does not match gsplat rows {n_rows}"
            )
        if row_offsets.shape != (n_rows,):
            raise ValueError(
                f"LiDAR row-offset count {row_offsets.shape} does not match gsplat rows {n_rows}"
            )
        if column_azimuths.shape != (n_cols,):
            raise ValueError(
                f"LiDAR model column count {column_azimuths.shape} does not match gsplat columns {n_cols}"
            )
        rays = self._ncore_lidar_model().elements_to_sensor_rays(
            self.dense_lidar_elements()
        )
        directions = (
            rays.detach().cpu().numpy().reshape(n_rows, n_cols, 3).astype(np.float32)
        )
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        if not np.all(np.isfinite(norms)) or float(norms.min()) <= 0.0:
            raise ValueError("Constructed invalid dense LiDAR ray directions")
        directions = directions / norms
        self._ray_directions_sensor = np.ascontiguousarray(directions, dtype=np.float32)
        return self._ray_directions_sensor

    def dense_lidar_world_rays_scene(
        self,
        T_sensor_scene_start: np.ndarray,
        T_sensor_scene_end: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rays = (
            self._ncore_lidar_model()
            .elements_to_world_rays_shutter_pose(
                self.dense_lidar_elements(),
                np.asarray(T_sensor_scene_start, dtype=np.float32),
                np.asarray(T_sensor_scene_end, dtype=np.float32),
                sensor_rays=self.dense_lidar_directions_sensor().reshape(-1, 3),
            )
            .world_rays
        )
        n_rows = int(self.lidar_coeffs.n_rows)
        n_cols = int(self.lidar_coeffs.n_columns)
        rays_np = (
            rays.detach().cpu().numpy().astype(np.float32).reshape(n_rows, n_cols, 6)
        )
        origins = np.ascontiguousarray(rays_np[..., :3], dtype=np.float32)
        directions = np.ascontiguousarray(rays_np[..., 3:], dtype=np.float32)
        return origins, directions

    def _native_range_image(
        self, lidar_frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ray_directions_sensor = self.dense_lidar_directions_sensor()
        ranges_image_m, return_image = _load_native_range_image(
            self.lidar_sensor,
            lidar_frame_idx,
        )
        ranges_image_m = np.asarray(ranges_image_m, dtype=np.float32)
        return_image = np.asarray(return_image, dtype=bool)
        if ranges_image_m.shape != ray_directions_sensor.shape[:2]:
            raise ValueError(
                "Native LiDAR range-image shape does not match dense ray grid: "
                f"{ranges_image_m.shape} vs {ray_directions_sensor.shape[:2]}"
            )
        if return_image.shape != ranges_image_m.shape:
            raise ValueError(
                "Native LiDAR return-mask shape does not match range image: "
                f"{return_image.shape} vs {ranges_image_m.shape}"
            )
        valid_image = return_image & np.isfinite(ranges_image_m) & (ranges_image_m > 0)
        return ranges_image_m, valid_image, ray_directions_sensor

    def _native_return_mask_and_ranges(
        self,
        ranges_image_m: np.ndarray,
        valid_image: np.ndarray,
        ray_directions_sensor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz_sensor = (
            ray_directions_sensor[valid_image] * ranges_image_m[valid_image, None]
        ).astype(np.float32)
        ranges_m = ranges_image_m[valid_image].astype(np.float32)
        keep = _finite_points_mask(xyz_sensor) & np.isfinite(ranges_m) & (ranges_m > 0)
        valid_rows, valid_cols = np.nonzero(valid_image)
        valid_image_kept = np.zeros_like(valid_image, dtype=bool)
        valid_image_kept[valid_rows[keep], valid_cols[keep]] = True
        return np.ascontiguousarray(ranges_m[keep], dtype=np.float32), valid_image_kept

    def _lidar_sensor_to_scene_poses(
        self, lidar_frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        start = self.parser.get_sensor_to_scene_pose(
            self.lidar_sensor,
            lidar_frame_idx,
            frame_timepoint=ncore.data.FrameTimepoint.START,
        )
        end = self.parser.get_sensor_to_scene_pose(
            self.lidar_sensor,
            lidar_frame_idx,
            frame_timepoint=ncore.data.FrameTimepoint.END,
        )
        return start.astype(np.float32), end.astype(np.float32)

    def _native_scene_points(
        self,
        ranges_image_m: np.ndarray,
        valid_image_kept: np.ndarray,
        ray_origins_scene: np.ndarray,
        ray_directions_scene: np.ndarray,
        T_sensor_scene_end: np.ndarray,
        ranges_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ranges_image_scene = ranges_image_m * np.float32(self.scene_range_scale)
        scene_points = (
            ray_origins_scene[valid_image_kept]
            + ray_directions_scene[valid_image_kept]
            * ranges_image_scene[valid_image_kept, None]
        ).astype(np.float32)
        keep = _finite_points_mask(scene_points)
        if not bool(keep.all()):
            ranges_m = np.ascontiguousarray(ranges_m[keep], dtype=np.float32)
            scene_points = np.ascontiguousarray(scene_points[keep], dtype=np.float32)
        xyz_sensor = _scene_points_to_sensor_frame(
            scene_points, T_sensor_scene_end
        ) / np.float32(self.scene_range_scale)
        return (
            np.ascontiguousarray(xyz_sensor, dtype=np.float32),
            np.ascontiguousarray(scene_points, dtype=np.float32),
            np.ascontiguousarray(ranges_m, dtype=np.float32),
        )

    def _validate_dense_return_directions(
        self,
        lidar_frame_idx: int,
        ray_directions_sensor: np.ndarray,
    ) -> tuple[int, float, float]:
        directions = np.asarray(
            self.lidar_sensor.get_frame_ray_bundle_direction(lidar_frame_idx),
            dtype=np.float32,
        ).reshape(-1, 3)
        model_elements = np.asarray(
            self.lidar_sensor.get_frame_ray_bundle_model_element(lidar_frame_idx)
        )
        if model_elements.ndim != 2 or model_elements.shape[1] < 2:
            raise ValueError(
                f"Unexpected LiDAR model-element shape {model_elements.shape}"
            )
        if directions.shape[0] != model_elements.shape[0]:
            raise ValueError(
                "LiDAR direction/model-element count mismatch: "
                f"{directions.shape[0]} vs {model_elements.shape[0]}"
            )
        rows = model_elements[:, 0].astype(np.int64)
        cols = model_elements[:, 1].astype(np.int64)
        n_rows = int(self.lidar_coeffs.n_rows)
        n_cols = int(self.lidar_coeffs.n_columns)
        in_bounds = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        if not bool(in_bounds.all()):
            raise ValueError(
                "Packed LiDAR model-element indices exceed model dimensions: "
                f"rows [{int(rows.min())}, {int(rows.max())}] / {n_rows}, "
                f"cols [{int(cols.min())}, {int(cols.max())}] / {n_cols}"
            )
        direction_errors = np.linalg.norm(
            ray_directions_sensor[rows, cols] - directions, axis=-1
        )
        direction_error_mean = (
            float(direction_errors.mean()) if len(direction_errors) else 0.0
        )
        direction_error_max = (
            float(direction_errors.max()) if len(direction_errors) else 0.0
        )
        if direction_error_mean > 1e-2 or direction_error_max > 5e-2:
            raise ValueError(
                "Dense LiDAR direction formula does not match NCore returned directions: "
                f"mean={direction_error_mean:.6g}, max={direction_error_max:.6g}"
            )
        return len(model_elements), direction_error_mean, direction_error_max

    def load_native_lidar(self, lidar_frame_idx: int) -> NativeLidarFrame:
        if lidar_frame_idx in self._native_cache:
            return self._native_cache[lidar_frame_idx]
        ranges_image_m, valid_image, ray_directions_sensor = self._native_range_image(
            lidar_frame_idx
        )
        ranges_m, valid_image_kept = self._native_return_mask_and_ranges(
            ranges_image_m, valid_image, ray_directions_sensor
        )
        T_sensor_scene_start, T_sensor_scene_end = self._lidar_sensor_to_scene_poses(
            lidar_frame_idx
        )
        ray_origins_scene, ray_directions_scene = self.dense_lidar_world_rays_scene(
            T_sensor_scene_start,
            T_sensor_scene_end,
        )
        ray_direction_norms = np.linalg.norm(ray_directions_scene, axis=-1)
        bad_scene_rays = ~np.isfinite(ray_direction_norms) | (ray_direction_norms <= 0)
        if bool(bad_scene_rays.any()):
            raise ValueError(
                "Dense LiDAR scene rays contain invalid directions for "
                f"{int(bad_scene_rays.sum())} rays."
            )
        xyz_sensor, scene_points, ranges_m = self._native_scene_points(
            ranges_image_m,
            valid_image_kept,
            ray_origins_scene,
            ray_directions_scene,
            T_sensor_scene_end,
            ranges_m,
        )
        _, direction_error_mean, direction_error_max = (
            self._validate_dense_return_directions(
                lidar_frame_idx, ray_directions_sensor
            )
        )
        ray_grid_pixels = int(valid_image.size)
        ray_return_pixels = int(valid_image.sum())
        native = NativeLidarFrame(
            lidar_frame_idx=lidar_frame_idx,
            ranges_image_m=np.ascontiguousarray(ranges_image_m, dtype=np.float32),
            valid_image=np.ascontiguousarray(valid_image, dtype=bool),
            xyz_sensor_m=xyz_sensor,
            scene_points=np.ascontiguousarray(scene_points, dtype=np.float32),
            ranges_m=np.ascontiguousarray(ranges_m, dtype=np.float32),
            ray_directions_sensor=ray_directions_sensor,
            ray_origins_scene=ray_origins_scene,
            ray_directions_scene=ray_directions_scene,
            lidar_coeffs=self.lidar_coeffs,
            T_sensor_scene=T_sensor_scene_end,
            T_sensor_scene_start=T_sensor_scene_start,
            T_sensor_scene_end=T_sensor_scene_end,
            ray_grid_pixels=ray_grid_pixels,
            ray_return_pixels=ray_return_pixels,
            ray_missing_pixels=ray_grid_pixels - ray_return_pixels,
            direction_error_mean=direction_error_mean,
            direction_error_max=direction_error_max,
        )
        self._native_cache[lidar_frame_idx] = native
        return native

    def _lidar_render_matrices(
        self, native: NativeLidarFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T_sensor_scene_mid = _mid_sensor_to_scene_pose(
            native.T_sensor_scene_start, native.T_sensor_scene_end
        )
        viewmats = (
            torch.from_numpy(np.linalg.inv(T_sensor_scene_mid))
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        viewmats_rs = (
            torch.from_numpy(np.linalg.inv(native.T_sensor_scene_end))
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        Ks = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        return viewmats, viewmats_rs, Ks

    def _rasterize_checkpoint_lidar(
        self, native: NativeLidarFrame
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        viewmats, viewmats_rs, Ks = self._lidar_render_matrices(native)
        return rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=None,
            viewmats=viewmats,
            viewmats_rs=viewmats_rs,
            Ks=Ks,
            width=native.lidar_coeffs.n_columns,
            height=native.lidar_coeffs.n_rows,
            near_plane=LIDAR_NEAR_PLANE_M * self.scene_range_scale,
            far_plane=LIDAR_FAR_PLANE_M * self.scene_range_scale,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            camera_model="lidar",
            lidar_coeffs=native.lidar_coeffs,
            with_ut=True,
            with_eval3d=True,
            global_z_order=False,
            extra_signals=self.lidar_extra_signal,
            render_mode=LIDAR_RENDER_MODE,
            ray_direction_scale=LIDAR_RAY_DIRECTION_SCALE,
            rolling_shutter=RollingShutterType.ROLLING_LEFT_TO_RIGHT,
        )

    def _rendered_range_and_raydrop(
        self,
        render_range: torch.Tensor,
        info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        render_extra = info.get("render_extra_signals")
        if render_extra is None:
            raise RuntimeError("Expected render_extra_signals for lidar extra signals.")
        if render_extra.shape[-1] < 3:
            raise RuntimeError(
                "Expected rendered lidar extra signals to contain intensity plus two raydrop logits."
            )
        render_raydrop_logits = render_extra[0, ..., 1:3]
        render_raydrop = torch.softmax(render_raydrop_logits, dim=-1)[..., 0]
        render_raydrop_np = render_raydrop.detach().cpu().numpy().astype(np.float32)
        return render_range[0, ..., 0], render_raydrop, render_raydrop_np

    def _rendered_valid_mask(
        self,
        render_range: torch.Tensor,
        render_alpha_image: torch.Tensor,
        render_raydrop: torch.Tensor,
    ) -> np.ndarray:
        valid_mask = (
            (render_alpha_image >= LIDAR_HIT_THRESHOLD)
            & torch.isfinite(render_range)
            & (render_range > 0)
            & (render_raydrop < LIDAR_RAYDROP_THRESHOLD)
        )
        return valid_mask.detach().cpu().numpy()

    def _rendered_points_from_mask(
        self,
        native: NativeLidarFrame,
        render_range_np: np.ndarray,
        valid_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scene_points = (
            native.ray_origins_scene[valid_np]
            + native.ray_directions_scene[valid_np] * render_range_np[valid_np, None]
        ).astype(np.float32)
        points_sensor_scene = _scene_points_to_sensor_frame(
            scene_points, native.T_sensor_scene_end
        )
        points_sensor_m = points_sensor_scene / np.float32(self.scene_range_scale)
        ranges_m = np.linalg.norm(points_sensor_m, axis=-1).astype(np.float32)
        n = min(
            len(points_sensor_scene),
            len(points_sensor_m),
            len(scene_points),
            len(ranges_m),
        )
        points_sensor_scene = points_sensor_scene[:n]
        points_sensor_m = points_sensor_m[:n]
        scene_points = scene_points[:n]
        ranges_m = ranges_m[:n]
        keep = (
            _finite_points_mask(points_sensor_scene)
            & _finite_points_mask(points_sensor_m)
            & _finite_points_mask(scene_points)
            & np.isfinite(ranges_m)
            & (ranges_m > 0)
        )
        return (
            np.ascontiguousarray(points_sensor_scene[keep], dtype=np.float32),
            np.ascontiguousarray(points_sensor_m[keep], dtype=np.float32),
            np.ascontiguousarray(scene_points[keep], dtype=np.float32),
            np.ascontiguousarray(ranges_m[keep], dtype=np.float32),
        )

    @torch.no_grad()
    def render_checkpoint_lidar(self, lidar_frame_idx: int) -> RenderedLidarFrame:
        if lidar_frame_idx in self._rendered_cache:
            return self._rendered_cache[lidar_frame_idx]
        native = self.load_native_lidar(lidar_frame_idx)
        render_range, render_alpha, info = self._rasterize_checkpoint_lidar(native)
        render_range, render_raydrop, render_raydrop_np = (
            self._rendered_range_and_raydrop(render_range, info)
        )
        render_alpha_image = render_alpha[0, ..., 0]
        render_range_np = render_range.detach().cpu().numpy().astype(np.float32)
        render_alpha_np = render_alpha_image.detach().cpu().numpy().astype(np.float32)
        valid_np = self._rendered_valid_mask(
            render_range, render_alpha_image, render_raydrop
        )
        points_sensor_scene, points_sensor_m, scene_points, ranges_m = (
            self._rendered_points_from_mask(native, render_range_np, valid_np)
        )
        rendered = RenderedLidarFrame(
            points_sensor_scene=points_sensor_scene,
            points_sensor_m=points_sensor_m,
            scene_points=scene_points,
            ranges_m=ranges_m,
            render_range_image_scene=np.ascontiguousarray(
                render_range_np, dtype=np.float32
            ),
            render_range_image_m=np.ascontiguousarray(
                render_range_np / np.float32(self.scene_range_scale), dtype=np.float32
            ),
            render_alpha_image=np.ascontiguousarray(render_alpha_np, dtype=np.float32),
            render_raydrop_image=np.ascontiguousarray(
                render_raydrop_np, dtype=np.float32
            ),
            valid_mask=np.ascontiguousarray(valid_np, dtype=bool),
            valid_fraction=float(valid_np.mean()),
        )
        self._rendered_cache[lidar_frame_idx] = rendered
        return rendered

    def overlay_frame(
        self,
        base_rgb: np.ndarray,
        flat_index: int,
        points: np.ndarray,
        ranges: np.ndarray,
        range_vmax: Optional[float] = None,
    ) -> np.ndarray:
        pixels_xy, valid_indices = _project_scene_points_to_camera(
            self.parser.camera_models[self.camera_id],
            self.parser.camtoworlds[flat_index],
            points,
        )
        if len(valid_indices) == 0:
            return base_rgb.copy()
        valid_indices = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
        pixels_xy = np.asarray(pixels_xy, dtype=np.float32).reshape(-1, 2)
        n = min(len(valid_indices), len(pixels_xy))
        valid_indices = valid_indices[:n]
        pixels_xy = pixels_xy[:n]
        ranges = np.asarray(ranges, dtype=np.float32).reshape(-1)
        in_range = (valid_indices >= 0) & (valid_indices < len(ranges))
        valid_indices = valid_indices[in_range]
        pixels_xy = pixels_xy[in_range]
        if len(valid_indices) == 0:
            return base_rgb.copy()
        idx = _subsample_indices(len(valid_indices), self.args.max_overlay_points)
        valid_indices = valid_indices[idx]
        pixels_xy = pixels_xy[idx]
        colors = _point_colors(
            ranges[valid_indices],
            self.args.overlay_color_mode,
            RGB_DARK_BLUE,
            self.args.range_upper_quantile,
            self.args.range_colormap,
            range_vmax=range_vmax,
        )
        return _draw_projected_points(
            base_rgb,
            pixels_xy,
            colors,
            self.args.overlay_alpha,
            self.args.overlay_point_radius,
        )

    @staticmethod
    def _safe_ratio(numer: int, denom: int) -> Optional[float]:
        return None if denom <= 0 else float(numer) / float(denom)

    def compute_validation_metrics(
        self, native: NativeLidarFrame, rendered: RenderedLidarFrame
    ) -> dict[str, Optional[float]]:
        native_valid = (
            native.valid_image
            & np.isfinite(native.ranges_image_m)
            & (native.ranges_image_m > 0)
        )
        render_range_m = rendered.render_range_image_m
        render_alpha = rendered.render_alpha_image
        render_raydrop = rendered.render_raydrop_image
        if render_range_m.shape != native.ranges_image_m.shape:
            raise ValueError(
                f"Rendered range shape {render_range_m.shape} != native range shape {native.ranges_image_m.shape}"
            )
        if render_alpha.shape != native.ranges_image_m.shape:
            raise ValueError(
                f"Rendered alpha shape {render_alpha.shape} != native range shape {native.ranges_image_m.shape}"
            )
        alpha = np.clip(render_alpha.astype(np.float32), 1e-6, 1.0 - 1e-6)
        ray_count = native.ray_grid_pixels
        empty_valid = ~native_valid
        target = native_valid.astype(np.float32)
        raydrop_target = 1.0 - target
        raydrop_prob = np.clip(render_raydrop.astype(np.float32), 1e-6, 1.0 - 1e-6)
        return_prob = 1.0 - raydrop_prob
        return_prob = np.clip(return_prob, 1e-6, 1.0 - 1e-6)
        pred_return = return_prob >= LIDAR_RETURN_THRESHOLD
        valid_range = np.isfinite(render_range_m) & (render_range_m > 0)
        pred_point = rendered.valid_mask & valid_range
        native_count = int(native_valid.sum())
        pred_count = int(pred_return.sum())
        pred_point_count = int(pred_point.sum())
        true_positive = int((pred_return & native_valid).sum())
        false_positive = int((pred_return & empty_valid).sum())
        false_negative = int((~pred_return & native_valid).sum())
        true_negative = int((~pred_return & empty_valid).sum())
        union = int((pred_return | native_valid).sum())
        raydrop_union = true_negative + false_negative + false_positive
        if ray_count:
            bce_values = target * np.log(return_prob) + (1.0 - target) * np.log(
                1.0 - return_prob
            )
            bce = -float(bce_values.mean())
            raydrop_mse = float(((raydrop_prob - raydrop_target) ** 2).mean())
        else:
            bce = None
            raydrop_mse = None
        metrics: dict[str, Optional[float]] = {
            "native_valid_pixels": float(native_count),
            "native_supervised_pixels": float(ray_count),
            "render_return_pixels": float(pred_count),
            "render_point_pixels": float(pred_point_count),
            "render_valid_fraction": float(rendered.valid_fraction),
            "return_accuracy": self._safe_ratio(
                true_positive + true_negative, ray_count
            ),
            "return_precision": self._safe_ratio(
                true_positive, true_positive + false_positive
            ),
            "return_recall": self._safe_ratio(
                true_positive, true_positive + false_negative
            ),
            "return_iou": self._safe_ratio(true_positive, union),
            "raydrop_accuracy": self._safe_ratio(
                true_positive + true_negative, ray_count
            ),
            "raydrop_precision": self._safe_ratio(
                true_negative, true_negative + false_negative
            ),
            "raydrop_recall": self._safe_ratio(
                true_negative, true_negative + false_positive
            ),
            "raydrop_iou": self._safe_ratio(true_negative, raydrop_union),
            "false_return_fraction": self._safe_ratio(
                false_positive, int(empty_valid.sum())
            ),
            "missed_return_fraction": self._safe_ratio(false_negative, native_count),
            "raydrop_bce": bce,
            "raydrop_mse": raydrop_mse,
            "has_lidar_raydrop_signal": True,
            "alpha_native_mean": (
                float(alpha[native_valid].mean()) if native_count else None
            ),
            "alpha_empty_mean": (
                float(alpha[empty_valid].mean()) if empty_valid.any() else None
            ),
            "raydrop_native_mean": (
                float(raydrop_prob[native_valid].mean()) if native_count else None
            ),
            "raydrop_empty_mean": (
                float(raydrop_prob[empty_valid].mean()) if empty_valid.any() else None
            ),
        }
        if native_count:
            delta = render_range_m[native_valid] - native.ranges_image_m[native_valid]
            abs_delta = np.abs(delta)
            metrics.update(
                {
                    "range_mae_m_on_native": float(abs_delta.mean()),
                    "range_rmse_m_on_native": float(np.sqrt(np.mean(delta * delta))),
                    "range_median_abs_m_on_native": float(np.quantile(abs_delta, 0.5)),
                    "range_p90_abs_m_on_native": float(np.quantile(abs_delta, 0.9)),
                    "native_range_mean_m": float(
                        native.ranges_image_m[native_valid].mean()
                    ),
                    "render_range_mean_m_on_native": float(
                        render_range_m[native_valid].mean()
                    ),
                }
            )
        return metrics

    def write_validation_stills(
        self,
        out_idx: int,
        flat_index: int,
        camera_frame_idx: int,
        native: NativeLidarFrame,
        rendered: RenderedLidarFrame,
    ) -> None:
        still_dir = self.output_dir / "stills"
        still_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{out_idx:04d}_flat_{flat_index:04d}_cam_{camera_frame_idx:04d}_lidar_{native.lidar_frame_idx:04d}"
        native_bev = _draw_bev(
            native.xyz_sensor_m,
            self.args.bev_size,
            self.args.bev_range_m,
            self.args.bev_point_radius,
            RGB_SIMULI_GREEN,
            self.args.max_bev_points,
        )
        rendered_bev = _draw_bev(
            rendered.points_sensor_m,
            self.args.bev_size,
            self.args.bev_range_m,
            self.args.bev_point_radius,
            RGB_SIMULI_GREEN,
            self.args.max_bev_points,
        )
        native_range = _range_image_to_rgb(
            native.ranges_image_m,
            native.valid_image,
            self.args.range_upper_quantile,
            self.args.range_colormap,
        )
        rendered_range = _range_image_to_rgb(
            rendered.render_range_image_m,
            rendered.valid_mask,
            self.args.range_upper_quantile,
            self.args.range_colormap,
        )
        range_error = _error_image_to_rgb(
            np.abs(rendered.render_range_image_m - native.ranges_image_m),
            native.valid_image,
            self.args.error_max_m,
        )
        imageio.imwrite(still_dir / f"{stem}_native_bev.png", native_bev)
        imageio.imwrite(still_dir / f"{stem}_rendered_bev.png", rendered_bev)
        imageio.imwrite(still_dir / f"{stem}_native_range.png", native_range)
        imageio.imwrite(still_dir / f"{stem}_rendered_range.png", rendered_range)
        imageio.imwrite(still_dir / f"{stem}_range_error_on_native.png", range_error)

    def _needs_rendered_lidar(self, outputs: set[str]) -> bool:
        if self.args.write_stills:
            return True
        return any(
            name.startswith("rendered") or name == "comparison" for name in outputs
        )

    def _write_video_frame(
        self,
        writers: dict[str, Mp4Writer],
        name: str,
        frame: np.ndarray,
    ) -> None:
        if self.args.skip_mp4:
            return
        if name not in writers:
            writers[name] = Mp4Writer(
                self.output_dir / f"{name}.mp4", self.args.fps, frame.shape
            )
        writers[name].write(frame)

    def _native_bev_frame(self, native: NativeLidarFrame) -> np.ndarray:
        return _draw_bev(
            native.xyz_sensor_m,
            self.args.bev_size,
            self.args.bev_range_m,
            self.args.bev_point_radius,
            RGB_SIMULI_GREEN,
            self.args.max_bev_points,
        )

    def _rendered_bev_frame(self, rendered: RenderedLidarFrame) -> np.ndarray:
        return _draw_bev(
            rendered.points_sensor_m,
            self.args.bev_size,
            self.args.bev_range_m,
            self.args.bev_point_radius,
            RGB_SIMULI_GREEN,
            self.args.max_bev_points,
        )

    def _comparison_frame(
        self,
        flat_index: int,
        observed: np.ndarray,
        pred: np.ndarray,
        native: NativeLidarFrame,
        rendered: RenderedLidarFrame,
    ) -> np.ndarray:
        range_vmax = _range_color_vmax(
            np.concatenate([native.ranges_m, rendered.ranges_m]),
            self.args.range_upper_quantile,
        )
        native_overlay = self.overlay_frame(
            observed, flat_index, native.scene_points, native.ranges_m, range_vmax
        )
        rendered_overlay = self.overlay_frame(
            pred, flat_index, rendered.scene_points, rendered.ranges_m, range_vmax
        )
        target_h = observed.shape[0]
        native_bev = cv2.resize(
            self._native_bev_frame(native),
            (target_h, target_h),
            interpolation=cv2.INTER_AREA,
        )
        rendered_bev = cv2.resize(
            self._rendered_bev_frame(rendered),
            (target_h, target_h),
            interpolation=cv2.INTER_AREA,
        )
        return np.concatenate(
            [
                np.concatenate([native_overlay, native_bev], axis=1),
                np.concatenate([rendered_overlay, rendered_bev], axis=1),
            ],
            axis=0,
        )

    def _write_requested_outputs(
        self,
        writers: dict[str, Mp4Writer],
        outputs: set[str],
        flat_index: int,
        camera_frame_idx: int,
        native: NativeLidarFrame,
        rendered: Optional[RenderedLidarFrame],
    ) -> None:
        observed: Optional[np.ndarray] = None
        pred: Optional[np.ndarray] = None

        def observed_frame() -> np.ndarray:
            nonlocal observed
            if observed is None:
                observed = self.load_camera_image(camera_frame_idx)
            return observed

        def pred_frame() -> np.ndarray:
            nonlocal pred
            if pred is None:
                pred = self.render_camera_image(flat_index)
            return pred

        overlay_range_vmax = None
        if (
            rendered is not None
            and self.args.overlay_color_mode == "range"
            and {"native_overlay", "rendered_overlay"} & outputs
        ):
            overlay_range_vmax = _range_color_vmax(
                np.concatenate([native.ranges_m, rendered.ranges_m]),
                self.args.range_upper_quantile,
            )

        if "native_overlay" in outputs:
            frame = self.overlay_frame(
                observed_frame(),
                flat_index,
                native.scene_points,
                native.ranges_m,
                overlay_range_vmax,
            )
            self._write_video_frame(writers, "native_overlay", frame)
        if "rendered_overlay" in outputs:
            assert rendered is not None
            frame = self.overlay_frame(
                pred_frame(),
                flat_index,
                rendered.scene_points,
                rendered.ranges_m,
                overlay_range_vmax,
            )
            self._write_video_frame(writers, "rendered_overlay", frame)
        if "native_bev" in outputs:
            self._write_video_frame(
                writers, "native_bev", self._native_bev_frame(native)
            )
        if "rendered_bev" in outputs:
            assert rendered is not None
            self._write_video_frame(
                writers, "rendered_bev", self._rendered_bev_frame(rendered)
            )
        if "comparison" in outputs:
            assert rendered is not None
            frame = self._comparison_frame(
                flat_index, observed_frame(), pred_frame(), native, rendered
            )
            self._write_video_frame(writers, "comparison", frame)

    def _frame_stats(
        self,
        out_idx: int,
        flat_index: int,
        camera_frame_idx: int,
        lidar_frame_idx: int,
        native: NativeLidarFrame,
        rendered: Optional[RenderedLidarFrame],
        validation_metrics: dict[str, Optional[float]],
    ) -> dict[str, object]:
        return {
            "output_frame": out_idx,
            "flat_index": int(flat_index),
            "camera_frame_index": int(camera_frame_idx),
            "lidar_frame_index": int(lidar_frame_idx),
            "native_points": int(len(native.xyz_sensor_m)),
            "native_ray_grid_pixels": native.ray_grid_pixels,
            "native_ray_return_pixels": native.ray_return_pixels,
            "native_ray_missing_pixels": native.ray_missing_pixels,
            "native_direction_error_mean": native.direction_error_mean,
            "native_direction_error_max": native.direction_error_max,
            "rendered_points": (
                int(len(rendered.points_sensor_m)) if rendered is not None else None
            ),
            "rendered_valid_fraction": (
                rendered.valid_fraction if rendered is not None else None
            ),
            **validation_metrics,
        }

    @staticmethod
    def _metric_summary(stats: list[dict[str, object]]) -> dict[str, float]:
        skip_keys = {
            "output_frame",
            "flat_index",
            "camera_frame_index",
            "lidar_frame_index",
        }
        metric_keys = sorted(
            {
                key
                for row in stats
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in skip_keys
            }
        )
        summary: dict[str, float] = {}
        for key in metric_keys:
            values = [
                float(row[key])
                for row in stats
                if isinstance(row.get(key), (int, float))
                and np.isfinite(float(row[key]))
            ]
            if values:
                summary[key] = float(np.mean(values))
        return summary

    def _manifest(
        self,
        frame_count: int,
        outputs: set[str],
        writers: dict[str, Mp4Writer],
        stats: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "data_dir": self.args.data_dir,
            "ckpt": self.args.ckpt,
            "output_dir": str(self.output_dir),
            "camera_id": self.camera_id,
            "lidar_id": self.lidar_id,
            "frame_source": self.args.frame_source,
            "frame_stride": self.args.frame_stride,
            "frame_count": frame_count,
            "outputs": sorted(writers),
            "requested_outputs": sorted(outputs),
            "skip_mp4": bool(self.args.skip_mp4),
            "write_stills": bool(self.args.write_stills),
            "still_count": int(self.args.still_count),
            "overlay_color_mode": self.args.overlay_color_mode,
            "render_mode": LIDAR_RENDER_MODE,
            "hit_threshold": LIDAR_HIT_THRESHOLD,
            "return_threshold": LIDAR_RETURN_THRESHOLD,
            "raydrop_threshold": LIDAR_RAYDROP_THRESHOLD,
            "lidar_near_plane_m": LIDAR_NEAR_PLANE_M,
            "lidar_far_plane_m": LIDAR_FAR_PLANE_M,
            "ray_direction_scale": LIDAR_RAY_DIRECTION_SCALE,
            "lidar_near_plane_scene": float(
                LIDAR_NEAR_PLANE_M * self.scene_range_scale
            ),
            "lidar_far_plane_scene": float(LIDAR_FAR_PLANE_M * self.scene_range_scale),
            "has_lidar_intensity_signal": True,
            "has_lidar_raydrop_signal": True,
            "normalize_world_space": bool(self.args.normalize_world_space),
            "normalization_scale": self.scene_range_scale,
            "world_global_to_scene_target_scale": self.world_target_scale,
            "metric_summary": self._metric_summary(stats),
            "stats": stats,
        }

    def export(self) -> None:
        frame_entries = self.selected_frames()
        outputs = set(self.args.outputs)
        writers: dict[str, Mp4Writer] = {}
        stats: list[dict[str, object]] = []
        needs_rendered = self._needs_rendered_lidar(outputs)

        try:
            for out_idx, (flat_index, camera_frame_idx) in enumerate(frame_entries):
                camera_timestamp_us = int(
                    self.camera_sensor.get_frame_timestamp_us(camera_frame_idx)
                )
                lidar_frame_idx = _get_nearest_lidar_frame_index(
                    self.lidar_sensor, camera_timestamp_us
                )
                native = self.load_native_lidar(lidar_frame_idx)
                rendered = (
                    self.render_checkpoint_lidar(lidar_frame_idx)
                    if needs_rendered
                    else None
                )
                validation_metrics = (
                    self.compute_validation_metrics(native, rendered)
                    if rendered is not None
                    else {}
                )
                self._write_requested_outputs(
                    writers,
                    outputs,
                    flat_index,
                    camera_frame_idx,
                    native,
                    rendered,
                )
                if (
                    rendered is not None
                    and self.args.write_stills
                    and (self.args.still_count <= 0 or out_idx < self.args.still_count)
                ):
                    self.write_validation_stills(
                        out_idx, flat_index, camera_frame_idx, native, rendered
                    )
                stats.append(
                    self._frame_stats(
                        out_idx,
                        flat_index,
                        camera_frame_idx,
                        lidar_frame_idx,
                        native,
                        rendered,
                        validation_metrics,
                    )
                )
                if (out_idx + 1) % max(1, self.args.log_every) == 0:
                    print(
                        f"wrote {out_idx + 1}/{len(frame_entries)} frames", flush=True
                    )
        finally:
            for writer in writers.values():
                writer.close()
        manifest = self._manifest(len(frame_entries), outputs, writers, stats)
        with (self.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            json.dumps(
                {k: manifest[k] for k in manifest if k != "stats"},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-factor", type=int, default=2)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--ncore-camera-ids", nargs="*", default=[])
    parser.add_argument("--ncore-lidar-ids", nargs="*", default=[])
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--lidar-id", default=None)
    parser.add_argument("--ncore-max-lidar-points", type=int, default=100000)
    parser.add_argument("--ncore-seek-offset-sec", type=float, default=None)
    parser.add_argument("--ncore-duration-sec", type=float, default=None)
    parser.add_argument("--ncore-poses-component-group", default="default")
    parser.add_argument("--ncore-intrinsics-component-group", default="default")
    parser.add_argument("--ncore-masks-component-group", default="default")
    parser.add_argument(
        "--open-consolidated", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalize-world-space", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--frame-source", choices=("val", "all"), default="val")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=(
            "native_overlay",
            "rendered_overlay",
            "native_bev",
            "rendered_bev",
            "comparison",
        ),
        default=[
            "native_overlay",
            "rendered_overlay",
            "native_bev",
            "rendered_bev",
            "comparison",
        ],
    )
    parser.add_argument(
        "--overlay-color-mode", choices=("uniform", "range"), default="range"
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.75)
    parser.add_argument("--overlay-point-radius", type=int, default=1)
    parser.add_argument("--max-overlay-points", type=int, default=80000)
    parser.add_argument("--range-colormap", default="turbo")
    parser.add_argument("--range-upper-quantile", type=float, default=0.98)
    parser.add_argument("--bev-size", type=int, default=1024)
    parser.add_argument("--bev-range-m", type=float, default=60.0)
    parser.add_argument("--bev-point-radius", type=int, default=1)
    parser.add_argument("--max-bev-points", type=int, default=250000)
    parser.add_argument("--skip-mp4", action="store_true")
    parser.add_argument("--write-stills", action="store_true")
    parser.add_argument("--still-count", type=int, default=5)
    parser.add_argument("--error-max-m", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = NCoreLidarMp4Exporter(args)
    exporter.export()


if __name__ == "__main__":
    main()
