"""Kinematic tracking feature extractor for Multimodal Bundle V3."""

from collections import deque
import math
from typing import Dict, Iterable, Optional, Tuple

from .contracts import EntityPosition, FrameKey, TRACKING_FEATURES

STEP_US = 40_000  # 25 fps step

def _finite(value: Optional[float]) -> Optional[float]:
    return float(value) if value is not None and math.isfinite(value) else None

class TrackingFeatureExtractor:
    """Extracts 18 tracking features from entity positions in centred metres."""

    def __init__(self, length_m: float = 105.0, width_m: float = 68.0):
        self.length_m = float(length_m)
        self.width_m = float(width_m)
        self.previous_ball: Optional[Tuple[FrameKey, int, float, float]] = None
        self.previous_key_time: Optional[Tuple[str, int, int, int]] = None
        self.window = deque(maxlen=25)
        self.last_complete_ts: Optional[int] = None

    def reset(self):
        """Reset temporal state on half/match boundary."""
        self.previous_ball = None
        self.previous_key_time = None
        self.window.clear()
        self.last_complete_ts = None

    def extract(self, key: FrameKey, timestamp_us: int, entities: Iterable[EntityPosition]) -> Dict[str, Optional[float]]:
        # Detect discontinuity
        expected_prev = (key.match_id, key.period, key.source_frame - 1, timestamp_us - STEP_US)
        if self.previous_key_time != expected_prev:
            self.reset()
        self.previous_key_time = (key.match_id, key.period, key.source_frame, timestamp_us)

        # Parse entities
        seen = set()
        ball: Optional[Tuple[Optional[float], Optional[float]]] = None
        teams = {"H": [], "A": []}

        for e in entities:
            if e.entity_id in seen:
                raise ValueError(f"Duplicate entity ID {e.entity_id}")
            seen.add(e.entity_id)
            xy = (_finite(e.x_m), _finite(e.y_m))
            if e.team == "BALL":
                if ball is not None:
                    raise ValueError("Multiple ball records in frame")
                ball = xy
            elif e.team in teams:
                teams[e.team].append(xy)
            else:
                raise ValueError(f"Unknown entity team {e.team}")

        x, y = ball if ball is not None else (None, None)
        complete_ball = x is not None and y is not None

        features: Dict[str, Optional[float]] = {name: None for name in TRACKING_FEATURES}
        features.update(
            ball_record_present=1.0 if ball is not None else 0.0,
            ball_x_observed=1.0 if x is not None else 0.0,
            ball_y_observed=1.0 if y is not None else 0.0,
            ball_xy_observed=1.0 if complete_ball else 0.0,
        )

        # Speed
        if complete_ball and self.previous_ball is not None:
            old_key, old_ts, old_x, old_y = self.previous_ball
            if (key.match_id, key.period, key.source_frame - 1, timestamp_us - STEP_US) == (old_key.match_id, old_key.period, old_key.source_frame, old_ts):
                dt_s = (timestamp_us - old_ts) / 1e6
                features["ball_speed_mps"] = math.hypot(x - old_x, y - old_y) / dt_s if dt_s > 0 else 0.0
            else:
                features["ball_speed_mps"] = 0.0
        elif complete_ball:
            features["ball_speed_mps"] = 0.0

        if complete_ball:
            self.previous_ball = (key, timestamp_us, x, y)
        else:
            self.previous_ball = None

        # Player counts and distances
        for team_label, prefix in (("H", "home"), ("A", "away")):
            players = teams[team_label]
            valid_players = [(px, py) for px, py in players if px is not None and py is not None]
            features[f"{prefix}_records"] = float(len(players))
            features[f"{prefix}_xy_observed"] = float(len(valid_players))
            if complete_ball and valid_players:
                features[f"nearest_{prefix}_m"] = min(math.hypot(x - px, y - py) for px, py in valid_players)

        # Margins & inside pitch
        if x is not None:
            features["ball_longitudinal_margin_m"] = self.length_m / 2.0 - abs(x)
        if y is not None:
            features["ball_lateral_margin_m"] = self.width_m / 2.0 - abs(y)
        if complete_ball:
            features["ball_inside_pitch"] = 1.0 if (abs(x) <= self.length_m / 2.0 and abs(y) <= self.width_m / 2.0) else 0.0

        # Trailing context
        self.window.append((features["ball_xy_observed"], features["nearest_home_m"], features["nearest_away_m"]))
        if complete_ball:
            self.last_complete_ts = timestamp_us

        features["ball_xy_fraction_1s"] = sum(r[0] for r in self.window) / len(self.window)
        features["ball_xy_age_s"] = ((timestamp_us - self.last_complete_ts) / 1e6) if self.last_complete_ts is not None else None

        for col_idx, feat_name in ((1, "nearest_home_median_1s"), (2, "nearest_away_median_1s")):
            observed_dists = [r[col_idx] for r in self.window if r[col_idx] is not None]
            if observed_dists:
                s = sorted(observed_dists)
                n = len(s)
                features[feat_name] = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            else:
                features[feat_name] = None

        return features
