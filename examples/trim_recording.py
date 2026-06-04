"""
Trim a recording to the largest contiguous span of valid annotations.

A frame is invalid if body keypoints contain NaN/all-zeros, or SLAM poses contain NaN.

Usage:
  python examples/trim_recording.py --data_root /path/to/episode [--output_root /path/to/out]
  python examples/trim_recording.py --data_root /path/to/episode --dry_run
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------

def _decode_ts(v):
    if isinstance(v, (bytes, np.bytes_)):
        return int(v.decode())
    return int(v)


def compute_validity_mask(ann_path: str) -> np.ndarray:
    """Return bool array of length N; True = frame is valid."""
    with h5py.File(ann_path, "r") as f:
        body_kp = np.array(f["full_body_mocap/keypoints"])   # (N, 52, 3)
        quats   = np.array(f["slam/quat_wxyz"])               # (N, 4)
        trans   = np.array(f["slam/trans_xyz"])               # (N, 3)

    body_invalid = (
        np.any(np.isnan(body_kp), axis=(1, 2)) |
        np.all(body_kp == 0, axis=(1, 2))
    )
    slam_invalid = (
        np.any(np.isnan(quats), axis=1) |
        np.any(np.isnan(trans), axis=1) |
        (np.all(quats == 0, axis=1) & np.all(trans == 0, axis=1))
    )
    return ~(body_invalid | slam_invalid)


def find_longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
    """Return (start, end) inclusive indices of the longest contiguous valid run."""
    best_start, best_len = 0, 0
    cur_start, cur_len   = 0, 0
    in_run = False
    for i, v in enumerate(valid):
        if v:
            if not in_run:
                cur_start, cur_len, in_run = i, 1, True
            else:
                cur_len += 1
        else:
            if in_run:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                in_run = False
    if in_run and cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return best_start, best_start + best_len - 1


# ---------------------------------------------------------------------------
# HDF5 trimming
# ---------------------------------------------------------------------------

def _copy_dataset(src_f, dst_f, name, data):
    """Write dataset preserving string dtype when needed."""
    src_ds = src_f[name]
    kwargs = {}
    if src_ds.dtype.kind in ("S", "O"):
        kwargs["dtype"] = src_ds.dtype
    if src_ds.compression:
        kwargs["compression"] = src_ds.compression
        kwargs["compression_opts"] = src_ds.compression_opts
    dst_f.create_dataset(name, data=data, **kwargs)
    for k, v in src_ds.attrs.items():
        dst_f[name].attrs[k] = v


def trim_hdf5(src_path: str, dst_path: str, start: int, end_excl: int) -> None:
    """Write a trimmed copy of annotation.hdf5 with frames [start, end_excl)."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy root attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Determine N so we know which axis-0 dims are frame-indexed
        N = src["slam/quat_wxyz"].shape[0]
        n_out = end_excl - start

        # IMU: find the IMU sample range from keyframe_indices
        imu_ki  = np.array(src["imu/keyframe_indices"]).flatten()
        imu_start = int(imu_ki[start])
        imu_end   = int(imu_ki[end_excl - 1]) + 1   # inclusive last frame -> exclusive slice
        n_imu_out = imu_end - imu_start

        def _visit(name, obj):
            group_path = str(Path(name).parent)
            # Ensure parent group exists
            if group_path and group_path != ".":
                if group_path not in dst:
                    dst.require_group(group_path)

            if isinstance(obj, h5py.Group):
                grp = dst.require_group(name)
                for k, v in obj.attrs.items():
                    grp.attrs[k] = v
                return

            # Dataset
            data = obj[...]
            if data.ndim >= 1 and data.shape[0] == N:
                data = data[start:end_excl]
            elif name.startswith("imu/") and name not in ("imu/keyframe_indices",) and data.ndim >= 1 and data.shape[0] > 1:
                data = data[imu_start:imu_end]

            # Remap keyframe_indices so they index into the trimmed IMU array
            if name == "imu/keyframe_indices":
                data = imu_ki[start:end_excl] - imu_start

            # Rewrite video/frame_number to be 0-based
            if name == "video/frame_number":
                data = np.arange(n_out, dtype=data.dtype)

            _copy_dataset(src, dst, name, data)

        src.visititems(_visit)

    print(f"  HDF5 written: {dst_path}  ({n_out} frames, {n_imu_out} IMU samples)")


# ---------------------------------------------------------------------------
# Video trimming
# ---------------------------------------------------------------------------

def _video_start_end_sec(ann_path: str, start: int, end_incl: int) -> tuple[float, float]:
    with h5py.File(ann_path, "r") as f:
        ts_ds = f["video/device_timestamp"]
        ts0        = _decode_ts(ts_ds[0])
        ts_start   = _decode_ts(ts_ds[start])
        ts_end     = _decode_ts(ts_ds[end_incl])
    start_sec = (ts_start - ts0) / 1e9
    end_sec   = (ts_end   - ts0) / 1e9
    return start_sec, end_sec


def trim_video(src: str, dst: str, start_sec: float, end_sec: float) -> None:
    duration = end_sec - start_sec
    # Output-side -ss (after -i) gives frame-accurate seeking at the cost of decoding.
    # Re-encoding ensures the output starts exactly at the right frame so VideoFrameReference
    # at t=0 matches annotation frame 0 — stream copy would start at the prior keyframe.
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-ss", f"{start_sec:.6f}",
        "-t",  f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg failed for {Path(src).name}: {result.stderr[-300:]}")
    else:
        size_mb = Path(dst).stat().st_size / 1e6
        print(f"  Video written: {Path(dst).name}  ({duration:.1f}s, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trim recording to longest contiguous valid annotation span.")
    parser.add_argument("--data_root",   required=True, help="Source episode folder")
    parser.add_argument("--output_root", default=None,  help="Output folder (default: <data_root>_trimmed)")
    parser.add_argument("--dry_run",     action="store_true", help="Print trim range but do not write files")
    args = parser.parse_args()

    data_root  = Path(args.data_root).resolve()
    ann_path   = data_root / "annotation.hdf5"
    if not ann_path.exists():
        print(f"Not found: {ann_path}")
        sys.exit(1)

    output_root = Path(args.output_root).resolve() if args.output_root else data_root.parent / (data_root.name + "_trimmed")

    print(f"Source:      {data_root}")
    print(f"Destination: {output_root}")

    # --- Validity analysis ---
    print("\nAnalyzing validity...")
    valid = compute_validity_mask(str(ann_path))
    N     = len(valid)
    n_inv = int((~valid).sum())
    print(f"  Total frames : {N}")
    print(f"  Invalid frames: {n_inv}  ({100*n_inv/N:.1f}%)")

    start, end_incl = find_longest_valid_run(valid)
    span = end_incl - start + 1
    print(f"  Longest valid run: [{start} – {end_incl}]  ({span} frames, {span/10:.1f}s @ 10fps)")

    # Sanity: show how much is trimmed from start/end
    trim_head = start
    trim_tail = N - 1 - end_incl
    print(f"  Trimming head: {trim_head} frames  |  tail: {trim_tail} frames")

    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    # --- Create output dir ---
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Trim HDF5 ---
    print("\nTrimming HDF5...")
    trim_hdf5(str(ann_path), str(output_root / "annotation.hdf5"), start, end_incl + 1)

    # --- Trim videos ---
    start_sec, end_sec = _video_start_end_sec(str(ann_path), start, end_incl)
    print(f"\nTrimming videos  ({start_sec:.3f}s – {end_sec:.3f}s)...")

    video_names = [
        "stereo_left.mp4", "stereo_right.mp4",
        "fisheye_cam0.mp4", "fisheye_cam1.mp4", "fisheye_cam2.mp4", "fisheye_cam3.mp4",
    ]
    for vname in video_names:
        src_v = data_root / vname
        if not src_v.exists():
            print(f"  [skip] {vname} not found")
            continue
        trim_video(str(src_v), str(output_root / vname), start_sec, end_sec)

    print(f"\nDone. Output: {output_root}")


if __name__ == "__main__":
    main()
