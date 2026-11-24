"""TAR/JSON manual measurement-line reading (spec section 7)."""
from __future__ import annotations

import json
import math
import tarfile

import numpy as np

from .angles import acute_angle_between_lines, line_angle_from_points

ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "cp936")
MIN_LINE_LENGTH = 1e-3


class ManualError(ValueError):
    pass


def _find_measure_models(node, acc):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "MeasureModel" and isinstance(value, list):
                acc.extend(value)
            _find_measure_models(value, acc)
    elif isinstance(node, list):
        for item in node:
            _find_measure_models(item, acc)


def _parse_json_members(tar: tarfile.TarFile):
    parsed = []
    for member in tar.getmembers():
        if not member.isfile():
            continue
        raw = tar.extractfile(member).read()
        obj = None
        for enc in ENCODINGS:
            try:
                obj = json.loads(raw.decode(enc))
                break
            except Exception:
                continue
        if obj is not None:
            parsed.append((member.name, obj))
    return parsed


def _valid_lines_from_object(obj):
    models = []
    _find_measure_models(obj, models)
    lines = []
    for item in models:
        pos = item.get("Pos") if isinstance(item, dict) else None
        if not isinstance(pos, dict):
            continue
        p1, p2 = pos.get("p1"), pos.get("p2")
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple))):
            continue
        if len(p1) < 2 or len(p2) < 2:
            continue
        try:
            p1f = [float(p1[0]), float(p1[1])]
            p2f = [float(p2[0]), float(p2[1])]
        except Exception:
            continue
        if not all(math.isfinite(v) for v in p1f + p2f):
            continue
        if math.hypot(p2f[0] - p1f[0], p2f[1] - p1f[1]) <= MIN_LINE_LENGTH:
            continue
        lines.append((p1f, p2f))
    return lines


def read_manual_lines(tar_path: str):
    """Return exactly two manual lines [(p1,p2), (p1,p2)] or raise ManualError."""
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            parsed = _parse_json_members(tar)
    except Exception as exc:
        raise ManualError(f"tar open/parse failed: {tar_path}: {exc}") from exc
    if not parsed:
        raise ManualError(f"no parseable JSON in tar: {tar_path}")
    lines, sources = [], []
    for name, obj in parsed:
        obj_lines = _valid_lines_from_object(obj)
        for ln in obj_lines:
            lines.append(ln)
            sources.append(name)
    if len(lines) != 2:
        raise ManualError(
            f"valid_manual_line_count={len(lines)} != 2 in {tar_path}")
    return lines, sources


def manual_record(tar_path: str) -> dict:
    lines, sources = read_manual_lines(tar_path)
    angles = [line_angle_from_points(*ln) for ln in lines]
    lengths = [float(np.hypot(ln[1][0] - ln[0][0], ln[1][1] - ln[0][1])) for ln in lines]
    rec = {
        "manual_line0_angle": angles[0],
        "manual_line1_angle": angles[1],
        "manual_line0_length": lengths[0],
        "manual_line1_length": lengths[1],
        "manual_full_angle": acute_angle_between_lines(angles[0], angles[1]),
        "manual_line_count": len(lines),
        "manual_tar_members": json.dumps(sources, ensure_ascii=False),
    }
    for i, ln in enumerate(lines):
        rec[f"manual_line{i}_p1_x"] = ln[0][0]
        rec[f"manual_line{i}_p1_y"] = ln[0][1]
        rec[f"manual_line{i}_p2_x"] = ln[1][0]
        rec[f"manual_line{i}_p2_y"] = ln[1][1]
    return rec
