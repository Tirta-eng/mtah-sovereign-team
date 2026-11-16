"""FIFA/Metrica XML metadata and ATD tracking frame streaming parser."""

import importlib.util
import math
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from .contracts import EntityPosition

def validate_periods(periods: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Validate two ordered inclusive 1-based ATD intervals."""
    validated = [tuple(row) for row in periods]
    if len(validated) != 2 or any(len(row) != 3 for row in validated):
        raise ValueError("Two explicit half intervals required")
    if any(type(val) is not int for row in validated for val in row):
        raise ValueError("Half boundaries must be integer frame IDs")
    if [row[0] for row in validated] != [1, 2]:
        raise ValueError("Ordered half IDs 1 and 2 required")
    if any(start < 1 or end < start for _, start, end in validated):
        raise ValueError("Invalid half bounds")
    if validated[0][2] >= validated[1][1]:
        raise ValueError("Overlapping or reversed halves")
    return validated


def provider_periods(parameters: Dict[str, str]) -> List[Tuple[int, int, int]]:
    """Parse provider-specific parameters as inclusive raw ATD frame bounds."""
    periods = []
    for number, name in ((1, "first_half"), (2, "second_half")):
        try:
            start, end = int(parameters[f"{name}_start"]), int(parameters[f"{name}_end"])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"Provider half bounds {name} missing, empty or non-integer") from err
        if start < 1 or end < start:
            raise ValueError(f"Invalid half bounds for {name}: {start}..{end}")
        periods.append((number, start, end))
    return validate_periods(periods)


def organizer_period_scope(source_path: Path, game: str, expected_frames: int, fps: int = 25) -> Dict[str, object]:
    """Read documented organizer scope from verified public metadata file."""
    if not game:
        raise ValueError("--game is required for organizer-public periods")
    path = Path(source_path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_mtah_public_period_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if game not in module.GAMES:
        raise ValueError(f"Game '{game}' absent from public organizer metadata")
    if module.GAMES[game]["fps"] != fps:
        raise ValueError("Public scope FPS differs from adapter")
    if module.GAMES[game]["n_video_frames"] != expected_frames:
        raise ValueError("Public scope recording length differs from expected tracking length")
    video = [tuple(row) for row in module.half_video_blocks(game)]
    if any(len(row) != 3 or any(type(v) is not int for v in row) for row in video):
        raise ValueError("Invalid public video scope contract")
    if any(first < 0 or last >= expected_frames for _, first, last in video):
        raise ValueError("Public video scope outside recording")
    periods = validate_periods([(period, first + 1, last + 1) for period, first, last in video])
    return dict(
        source="organizer-public",
        game=game,
        periods=periods,
        video_blocks=video,
        source_file=str(path),
    )


def metadata(path: Path, period_scope: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Parse Metrica FIFA metadata XML without positional roster assumptions."""
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    meta = root.find("Metadata") if root.tag != "Metadata" else root
    if meta is None:
        raise ValueError("FIFA Metadata element required")
    if float(meta.findtext("./GlobalConfig/FrameRate", "nan")) != 25:
        raise ValueError("25-Hz metadata required")
    length = float(meta.findtext(".//FieldSize/Width", "nan"))
    width = float(meta.findtext(".//FieldSize/Height", "nan"))
    if not all(math.isfinite(v) and v > 0 for v in (length, width)):
        raise ValueError("Finite positive pitch dimensions required")
    teams = [e.get("id") for e in meta.findall("./Teams/Team")]
    if len(teams) != 2 or len(set(teams)) != 2 or not all(teams):
        raise ValueError("Two unique teams required")
    score = meta.find(".//MatchParameters/Score")
    home = score.get("idLocalTeam") if score is not None else teams[0]
    away = score.get("idVisitingTeam") if score is not None else teams[1]
    if set((home, away)) != set(teams):
        raise ValueError("Home/away metadata conflict")

    players = {}
    for e in meta.findall("./Players/Player"):
        identity = e.get("id")
        team = e.get("teamId")
        if not identity or identity in players or team not in teams:
            raise ValueError("Invalid player/team metadata")
        players[identity] = "H" if team == home else "A"

    channels = {}
    for e in meta.findall(".//PlayerChannels/PlayerChannel"):
        identity = e.get("id")
        player = e.get("playerId")
        axis = e.get("channelId")
        if not identity or identity in channels or player not in players or not axis:
            raise ValueError("Invalid player channel")
        channels[identity] = (player, axis)

    parameters = {}
    for e in meta.findall("./GlobalConfig/ProviderGlobalParameters/ProviderParameter"):
        name = e.findtext("Name")
        value = (e.findtext("Value") or "").strip()
        if name in parameters:
            raise ValueError(f"Duplicate provider parameter {name}")
        parameters[name] = value

    periods = provider_periods(parameters) if period_scope is None else validate_periods(period_scope["periods"])

    formats = []
    for spec in root.findall("./DataFormatSpecifications/DataFormatSpecification"):
        start, end = int(spec.attrib["startFrame"]), int(spec.attrib["endFrame"])
        children = list(spec)
        layout = []
        seen = set()
        for group in children[1]:
            refs = [channels.get(e.get("playerChannelId")) if e.tag == "PlayerChannelRef" else None for e in group]
            if not refs or any(r is None for r in refs) or len({r[0] for r in refs}) != 1:
                raise ValueError("Unresolved player channel reference")
            player = refs[0][0]
            axes = [r[1] for r in refs]
            seen.add(player)
            layout.append(dict(player=player, team=players[player], axes=axes))
        ball = [e.get("channelId") if e.tag == "BallChannelRef" else None for e in children[2]]
        formats.append(dict(start=start, end=end, players=layout, ball=ball))

    formats.sort(key=lambda f: f["start"])
    return dict(
        length_m=length,
        width_m=width,
        fps=25,
        home_id=home,
        away_id=away,
        periods=periods,
        formats=formats,
        period_scope=period_scope or dict(source="provider", output_convention="atd_1_based_inclusive"),
    )


def normalized_pair(token: str, axes: Sequence[str], length: float, width: float) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Return centred metre x/y from normalized coordinates."""
    if token.strip().lower() in ("", "nan", "n"):
        return None
    values = token.split(",")
    if len(values) != len(axes):
        raise ValueError("ATD channel count mismatch")
    result = []
    for axis, scale in (("x", length), ("y", width)):
        value = values[axes.index(axis)].strip()
        number = float(value) if value.lower() not in ("", "nan", "n") else float("nan")
        coord = ((number - 0.5) if axis == "x" else (0.5 - number)) * scale if math.isfinite(number) else None
        result.append(coord)
    return tuple(result)


def frames(path: Path, meta: Dict[str, object], expected: int) -> Generator[Tuple[int, Optional[int], int, List[EntityPosition]], None, None]:
    """Stream exact ATD IDs 1..expected, emitting zero-based video-frame keys."""
    if expected < 1:
        raise ValueError("Expected frames must be >= 1")
    number = 0
    format_index = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split(":")
            number += 1
            if len(fields) != 3 or int(fields[0]) != number or number > expected:
                raise ValueError("ATD IDs must be exactly 1..expected")
            while format_index < len(meta["formats"]) and meta["formats"][format_index]["end"] < number:
                format_index += 1
            if format_index == len(meta["formats"]) or meta["formats"][format_index]["start"] > number:
                raise ValueError("Frame has no declared ATD layout")
            spec = meta["formats"][format_index]
            groups = fields[1].split(";") if fields[1] else [""] * len(spec["players"])
            if len(groups) == len(spec["players"]) + 1 and groups[-1] == "":
                groups.pop()
            entities = []
            for token, slot in zip(groups, spec["players"]):
                xy = normalized_pair(token, slot["axes"], meta["length_m"], meta["width_m"])
                if xy is not None and any(v is not None for v in xy):
                    entities.append(EntityPosition(slot["player"], slot["team"], *xy))
            ball = normalized_pair(fields[2], spec["ball"], meta["length_m"], meta["width_m"])
            if ball is not None:
                entities.append(EntityPosition("BALL", "BALL", *ball))
            period = next((p for p, a, b in meta["periods"] if a <= number <= b), None)
            yield number - 1, period, (number - 1) * 40000, entities
    if number != expected:
        raise ValueError(f"ATD row count {number} != expected {expected}")
