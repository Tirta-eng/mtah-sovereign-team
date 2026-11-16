"""Visual feature extractor for Multimodal Bundle V3."""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    for parent in Path(__file__).resolve().parents:
        geom = parent / "runtime/v3-geometry-venv/Lib/site-packages"
        if geom.exists() and str(geom) not in sys.path:
            sys.path.insert(0, str(geom))
            break
    import cv2

from .contracts import VISUAL_FEATURES

def project_pitch_to_pixel(
    x_m: Optional[float],
    y_m: Optional[float],
    H_mat: Optional[np.ndarray],
    is_normalised_coords: bool = False,
    img_width: int = 1920,
    img_height: int = 1080,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Tuple[int, int]:
    """Map centred pitch metres (x_m, y_m) to video pixels (u, v)."""
    if H_mat is None or x_m is None or y_m is None or not (np.isfinite(x_m) and np.isfinite(y_m)):
        return img_width // 2, img_height // 2

    if is_normalised_coords:
        # Official homography.csv maps normalized coordinates (0..1) to pixels
        x_norm = x_m / pitch_length + 0.5
        y_norm = 0.5 - y_m / pitch_width
        p_homo = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    else:
        # SoccerTrack homography.npy maps pitch metres [0, pitch_length] x [0, pitch_width]
        x_pitch = x_m + pitch_length / 2.0
        y_pitch = y_m + pitch_width / 2.0
        p_homo = np.array([x_pitch, y_pitch, 1.0], dtype=np.float64)

    proj = H_mat @ p_homo
    w = proj[2]
    if abs(w) < 1e-6 or not np.isfinite(w):
        return img_width // 2, img_height // 2

    u = int(np.clip(round(proj[0] / w), 0, img_width - 1))
    v = int(np.clip(round(proj[1] / w), 0, img_height - 1))
    return u, v


class VisualFeatureExtractor:
    """Extracts 8 visual features from streaming panorama video frames."""

    def __init__(
        self,
        homography_source: Union[np.ndarray, Dict[int, Optional[np.ndarray]]],
        is_normalised_coords: bool = False,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.is_normalised_coords = is_normalised_coords
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        if isinstance(homography_source, dict):
            self.homography_dict: Optional[Dict[int, Optional[np.ndarray]]] = homography_source
            self.static_H: Optional[np.ndarray] = None
        else:
            self.homography_dict = None
            self.static_H = np.asarray(homography_source, dtype=np.float64)
            if self.static_H.shape != (3, 3):
                raise ValueError(f"Expected 3x3 homography matrix, got {self.static_H.shape}")

        self.prev_small_gray: Optional[np.ndarray] = None
        self.prev_crop128: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state on video discontinuity."""
        self.prev_small_gray = None
        self.prev_crop128 = None

    def get_homography_for_frame(self, frame_idx: Optional[int]) -> Optional[np.ndarray]:
        if self.static_H is not None:
            return self.static_H
        if self.homography_dict is not None and frame_idx is not None:
            return self.homography_dict.get(frame_idx)
        return None

    def extract_frame(
        self,
        frame_bgr: np.ndarray,
        ball_x_m: Optional[float],
        ball_y_m: Optional[float],
        frame_idx: Optional[int] = None,
    ) -> List[float]:
        """Compute the 8 visual features for a single video frame."""
        img_height, img_width = frame_bgr.shape[:2]
        H_curr = self.get_homography_for_frame(frame_idx)

        u, v = project_pitch_to_pixel(
            ball_x_m,
            ball_y_m,
            H_curr,
            is_normalised_coords=self.is_normalised_coords,
            img_width=img_width,
            img_height=img_height,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
        )

        # 1. Global and pitch motion on downsampled frame (512x288)
        small = cv2.resize(frame_bgr, (512, 288), interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev_small_gray is not None:
            diff_small = cv2.absdiff(gray_small, self.prev_small_gray)
            global_motion = float(diff_small.mean())
            # Pitch active region in 512x288: central band
            pitch_motion = float(diff_small[40:240, 50:460].mean())
        else:
            global_motion = 0.0
            pitch_motion = 0.0

        # 2. Crops around projected ball
        u0_128, u1_128 = max(0, u - 64), min(img_width, u + 64)
        v0_128, v1_128 = max(0, v - 64), min(img_height, v + 64)
        crop128 = frame_bgr[v0_128:v1_128, u0_128:u1_128]

        u0_64, u1_64 = max(0, u - 32), min(img_width, u + 32)
        v0_64, v1_64 = max(0, v - 32), min(img_height, v + 32)
        crop64 = frame_bgr[v0_64:v1_64, u0_64:u1_64]

        # Ball vicinity motion
        if self.prev_crop128 is not None and crop128.shape == self.prev_crop128.shape:
            ball_motion = float(cv2.absdiff(crop128, self.prev_crop128).mean())
        else:
            ball_motion = 0.0

        # Ball crop statistics
        if crop64.size > 0:
            ball_crop_std = float(crop64.std())
            b64 = crop64[:, :, 0].astype(float)
            g64 = crop64[:, :, 1].astype(float)
            r64 = crop64[:, :, 2].astype(float)
            white_ratio = float(np.mean((b64 > 180) & (g64 > 180) & (r64 > 180)))
        else:
            ball_crop_std = 0.0
            white_ratio = 0.0

        # Jersey color and edge energy
        if crop128.size > 0:
            b128 = crop128[:, :, 0].astype(float)
            g128 = crop128[:, :, 1].astype(float)
            r128 = crop128[:, :, 2].astype(float)

            home_color = float(np.mean(np.maximum(0.0, b128 - (r128 + g128) / 2.0)))
            away_mask = (b128 > 140) & (g128 > 140) & (r128 > 140) & (np.std([r128, g128, b128], axis=0) < 15.0)
            away_color = float(np.mean(away_mask))

            gray128 = cv2.cvtColor(crop128, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray128, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray128, cv2.CV_32F, 0, 1, ksize=3)
            edge_energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        else:
            home_color = 0.0
            away_color = 0.0
            edge_energy = 0.0

        self.prev_small_gray = gray_small
        self.prev_crop128 = crop128

        return [
            global_motion,
            pitch_motion,
            ball_motion,
            ball_crop_std,
            white_ratio,
            home_color,
            away_color,
            edge_energy,
        ]
