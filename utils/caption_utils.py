"""
Caption loading from annotation.hdf5 for Xperience-10M.
"""

import json
import re
from pathlib import Path

import h5py


def _find_nearest_frame_index(timestamp_int, name_to_index):
    """Find frame index whose name (as timestamp) is nearest to timestamp_int."""
    best_idx = -1
    best_diff = float("inf")
    for name, idx in name_to_index.items():
        stem = name.rsplit(".", 1)[0] if "." in name else name
        try:
            ts = int(stem)
        except ValueError:
            continue
        diff = abs(ts - timestamp_int)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


# Values at/above this are treated as device timestamps (~1e13), not sequential frame indices.
_TIMESTAMP_THRESHOLD = 100_000_000


def _resolve_frame_ref(ref, name_to_index, N):
    """Resolve a caption frame reference to a 0-based frame index.

    Caption data references frames in several conventions:
      * exact image name / stem match (timestamp-style img_names, e.g. "75294505147569.jpg")
      * sequential names "frame_000123" produced by the captioning pipeline (1-based)
      * bare integer / digit-string sequential indices (1-based, e.g. 51, "11")
      * large integer device timestamps (nearest-frame fallback)
    Returns -1 if it cannot be resolved.
    """
    if ref is None:
        return -1
    s = str(ref).strip()
    if not s:
        return -1
    # 1) Exact image name / stem match (timestamp-style img_names).
    if s in name_to_index:
        return name_to_index[s]
    stem = s.rsplit(".", 1)[0] if "." in s else s
    if stem in name_to_index:
        return name_to_index[stem]
    # 2) "frame_000123" -> 1-based sequential index.
    m = re.search(r"frame_(\d+)", s)
    if m:
        return min(max(int(m.group(1)) - 1, 0), N - 1)
    # 3) Bare integer: small -> 1-based sequential index; large -> device timestamp.
    digits = stem[1:] if stem.startswith("-") else stem
    if digits.isdigit():
        val = int(stem)
        if val >= _TIMESTAMP_THRESHOLD:
            return _find_nearest_frame_index(val, name_to_index)
        return min(max(val - 1, 0), N - 1)
    return -1


def _build_frame_info_map_from_caption(data, name_to_index, N):
    """Build frame_info_map and segment_boundaries from caption segments."""
    frame_info_map = {}
    segments = data.get("segments", [])
    segment_boundaries = []
    for seg in segments:
        theme = seg.get("Sub Task", "")
        start_frame = seg.get("start_frame", 0)
        end_frame = seg.get("end_frame", 0)
        seg_id = seg.get("segment_id", 0)
        actions = seg.get("Current Action", [])
        action_ranges = []
        for action in actions:
            start_ref = action.get("start_frame_name")
            end_ref = action.get("end_frame_name")
            if not start_ref and action.get("start_frame") is not None:
                start_ref = action["start_frame"]
            if not end_ref and action.get("end_frame") is not None:
                end_ref = action["end_frame"]
            label = action.get("label", "")
            desc = action.get("description", "")
            if start_ref is not None and end_ref is not None:
                si = _resolve_frame_ref(start_ref, name_to_index, N)
                ei = _resolve_frame_ref(end_ref, name_to_index, N)
                if si >= 0 and ei >= 0:
                    if ei < si:
                        si, ei = ei, si
                    action_ranges.append((si, ei, label, desc))
        action_ranges.sort(key=lambda x: x[0])
        indices_in_segment = set()
        for start_idx, end_idx, label, desc in action_ranges:
            for idx in range(start_idx, min(end_idx + 1, N)):
                indices_in_segment.add(idx)
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                frame_info_map[idx]["theme"] = theme
                frame_info_map[idx]["action_label"] = label
                frame_info_map[idx]["action_desc"] = desc
        objects_map = seg.get("objects", {})
        interaction_map = seg.get("interaction", {})
        for fname in set(objects_map.keys()) | set(interaction_map.keys()):
            idx = _resolve_frame_ref(fname, name_to_index, N)
            if idx < 0:
                continue
            indices_in_segment.add(idx)
            if idx not in frame_info_map:
                frame_info_map[idx] = {}
            if "theme" not in frame_info_map[idx]:
                frame_info_map[idx]["theme"] = theme
            if fname in objects_map:
                frame_info_map[idx]["objects"] = objects_map[fname]
            if fname in interaction_map:
                frame_info_map[idx]["interaction"] = interaction_map[fname]
        if indices_in_segment:
            seg_start = _resolve_frame_ref(start_frame, name_to_index, N)
            seg_end = _resolve_frame_ref(end_frame, name_to_index, N)
            if seg_start >= 0 and seg_end >= 0 and seg_end < seg_start:
                seg_start, seg_end = seg_end, seg_start
            if seg_start < 0:
                seg_start = min(indices_in_segment)
            if seg_end < 0:
                seg_end = max(indices_in_segment)
            segment_boundaries.append((seg_start, seg_end, theme, seg_id))
            for idx in range(seg_start, seg_end + 1):
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                if "theme" not in frame_info_map[idx]:
                    frame_info_map[idx]["theme"] = theme
                if not frame_info_map[idx].get("action_label"):
                    prev = None
                    for si, ei, label, desc in action_ranges:
                        if ei < idx:
                            prev = (ei, label, desc)
                        elif si > idx:
                            break
                    if prev:
                        _, label, desc = prev
                        frame_info_map[idx]["action_label"] = label
                        frame_info_map[idx]["action_desc"] = desc
                    elif action_ranges:
                        _, _, label, desc = action_ranges[0]
                        frame_info_map[idx]["action_label"] = label
                        frame_info_map[idx]["action_desc"] = desc
            frames_with_objects = sorted({i for i in (_resolve_frame_ref(f, name_to_index, N) for f in objects_map.keys()) if i >= 0})
            frames_with_interaction = sorted({i for i in (_resolve_frame_ref(f, name_to_index, N) for f in interaction_map.keys()) if i >= 0})
            for idx in range(seg_start, seg_end + 1):
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                if frames_with_objects:
                    prev_o = max([f for f in frames_with_objects if f <= idx], default=None)
                    if prev_o is not None and "objects" not in frame_info_map[idx]:
                        frame_info_map[idx]["objects"] = frame_info_map[prev_o]["objects"]
                if frames_with_interaction:
                    prev_i = max([f for f in frames_with_interaction if f <= idx], default=None)
                    if prev_i is not None and "interaction" not in frame_info_map[idx]:
                        frame_info_map[idx]["interaction"] = frame_info_map[prev_i]["interaction"]
    for idx in range(N):
        if idx not in frame_info_map:
            frame_info_map[idx] = {}
    for idx in range(1, N):
        prev = frame_info_map.get(idx - 1, {})
        for key in ("theme", "action_label", "action_desc"):
            if key not in frame_info_map[idx] and key in prev:
                frame_info_map[idx][key] = prev[key]
    return frame_info_map, segment_boundaries


def load_caption_data_from_annotation_hdf5(annotation_path, data_root, img_names):
    """
    Load caption data from annotation.hdf5 only (dataset 'caption' or 'captions').
    Returns (main_task, frame_info_map, segment_boundaries, task_to_id).
    If no data found, returns ("", None, [], {}).
    """
    data = None
    data_root_path = Path(data_root)
    try:
        with h5py.File(annotation_path, "r") as f:
            for key in ("caption", "captions"):
                if key not in f:
                    continue
                raw = f[key][...]
                if hasattr(raw, "ndim") and getattr(raw, "ndim", -1) == 0 and hasattr(raw, "item"):
                    raw = raw.item()
                elif hasattr(raw, "size") and getattr(raw, "size", 0) == 1 and hasattr(raw, "item"):
                    raw = raw.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                elif not isinstance(raw, str) and hasattr(raw, "tobytes"):
                    raw = raw.tobytes().decode("utf-8", errors="replace")
                elif not isinstance(raw, str):
                    raw = str(raw)
                raw = raw.strip() if isinstance(raw, str) else ""
                if not raw:
                    continue
                if raw.startswith("{") or raw.startswith("["):
                    data = json.loads(raw)
                    break
                path_candidate = Path(raw)
                if not path_candidate.is_absolute():
                    path_candidate = data_root_path / path_candidate
                if path_candidate.exists():
                    with open(path_candidate, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    break
                try:
                    data = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    if data is None:
        return "", None, [], {}
    main_task = data.get("config", {}).get("Main Task", "N/A")
    N = len(img_names)
    name_to_index = {name: i for i, name in enumerate(img_names)}
    for i, name in enumerate(img_names):
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem not in name_to_index:
            name_to_index[stem] = i
    frame_info_map, segment_boundaries = _build_frame_info_map_from_caption(data, name_to_index, N)
    unique_tasks = []
    for _, _, task_name, _ in segment_boundaries:
        if task_name not in unique_tasks:
            unique_tasks.append(task_name)
    task_to_id = {name: idx + 1 for idx, name in enumerate(unique_tasks)}
    return main_task, frame_info_map, segment_boundaries, task_to_id


__all__ = ["load_caption_data_from_annotation_hdf5"]
