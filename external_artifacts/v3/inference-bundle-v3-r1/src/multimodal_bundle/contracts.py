"""Data structures and contracts for Multimodal Bundle V3."""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class FrameKey:
    """Unique identifier for a match frame."""
    match_id: str
    period: int
    source_frame: int

@dataclass(frozen=True)
class EntityPosition:
    """Position of a tracked entity (player or ball) in centred pitch metres."""
    entity_id: str
    team: str  # 'H', 'A', or 'BALL'
    x_m: Optional[float]
    y_m: Optional[float]

TRACKING_FEATURES = (
    "ball_record_present",
    "ball_x_observed",
    "ball_y_observed",
    "ball_xy_observed",
    "ball_speed_mps",
    "home_records",
    "home_xy_observed",
    "away_records",
    "away_xy_observed",
    "nearest_home_m",
    "nearest_away_m",
    "ball_longitudinal_margin_m",
    "ball_lateral_margin_m",
    "ball_inside_pitch",
    "ball_xy_fraction_1s",
    "ball_xy_age_s",
    "nearest_home_median_1s",
    "nearest_away_median_1s",
)

VISUAL_FEATURES = (
    "vis_global_motion",
    "vis_pitch_motion",
    "vis_ball_vicinity_motion",
    "vis_ball_crop_std",
    "vis_ball_white_ratio",
    "vis_home_color_score",
    "vis_away_color_score",
    "vis_player_edge_energy",
)

ALL_FEATURES_26 = TRACKING_FEATURES + VISUAL_FEATURES
LABELS = ("H", "A", "D")
