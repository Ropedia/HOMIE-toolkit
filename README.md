<p align="center">
  <a href="https://ropedia.com/">
    <img src="assets/logo.png" alt="HOMIE-toolkit logo" width="400" />
  </a>
  <br />
  <em>Interactive Intelligence from Human Xperience</em>
</p>

# HOMIE-toolkit

Tools for **reading** and **visualizing** [Xperience-10M](https://huggingface.co/datasets/ropedia-ai/xperience-10m) data.

- Load annotation and use the data in your own scripts (export, training, custom viz)
- Reuse visualization helpers (depth colormap, skeleton, point cloud) with Rerun

## 📁 Layout

| Path | Description |
|------|-------------|
| `data_loader.py` | Load `annotation.hdf5` (calibration, SLAM, hand/body mocap, depth, IMU, point cloud); list contents and load video frames. |
| `visualization.py` | Helpers: `create_blueprint`, `depth_to_colormap`, `depth_to_pointcloud`, `build_line3d_skeleton`, `scale_image`, `transform_points_to_world`. |
| `utils/` | Calibration, caption, video, and constant helpers used by the loader. |
| `examples/example_load_annotation.py` | List HDF5 contents, load annotation, inspect calibration. |
| `examples/example_visualize_rrd.py` | Log skeleton + depth to a Rerun `.rrd` file; open with `rerun vis.rrd`. |
| `examples/trim_recording.py` | Trim an episode (HDF5 + videos) to its longest contiguous span of valid annotations. |

## 📦 Install

```bash
conda create -n homie python=3.12
conda activate homie
pip install -r requirements.txt
```

## 🚀 Getting Started

Download sample data [here](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample).

### 📋 List Annotations

```bash
python examples/example_load_annotation.py --data_root /path/to/episode
```

Example output (top-level structure + loaded summary):

```
--- annotation.hdf5 contents (top-level) ---
  calibration: group    (cam0, cam01, cam1, cam2, cam3: K, T_c_b, ...)
  depth: group           (depth, confidence, depth_min, depth_max, scale)
  full_body_mocap: group (keypoints, contacts, body_quats, ...)
  hand_mocap: group     (left_joints_3d, right_joints_3d, mano params)
  imu: group            (device_timestamp_ns, accel_xyz, gyro_xyz, keyframe_indices)
  slam: group           (quat_wxyz, trans_xyz, frame_names, point_cloud)
  caption: ...          metadata: ...

--- Loaded data summary ---
  Frames (img_names): N
  R_c2w_all: (N, 3, 3)   t_c2w_all: (N, 3)
  Hand left/right joints: (N, 21, 3)   Full-body keypoints: (N, 52, 3)
  Contacts: (N, 21)   Depth: lazy loader, N frames   IMU: M samples

--- Calibration ---
  cam01.K, cam0–cam3 T_c_b: available

Done. Use these arrays for your own processing or pass to example_visualize_rrd.py.
```

### 🎬 Visualize with Rerun

```bash
python examples/example_visualize_rrd.py --data_root /path/to/episode --output_rrd vis.rrd
```

Then open the Rerun viewer: `rerun vis.rrd`

![Rerun visualization](./assets/rerun.png)

<details>
<summary><strong>All <code>example_visualize_rrd.py</code> options</strong></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | *(required)* | Episode folder containing `annotation.hdf5`. |
| `--output_rrd` | `vis.rrd` | Output `.rrd` path (relative paths are resolved against the package root). |
| `--num_frames` | `-1` | Number of frames to log (`-1` = all). |
| `--annotation` | `annotation.hdf5` | Annotation HDF5 file name. |
| `--stereo_left` / `--stereo_right` | `stereo_left.mp4` / `stereo_right.mp4` | Stereo video file names. |
| `--fisheye_pattern` | `fisheye_{cam}.mp4` | Fisheye file name pattern (`{cam}` → `cam0`..`cam3`). |
| `--show_fisheye` / `--show_stereo` / `--show_depth_colormap` / `--show_depth_points` / `--show_skeleton` / `--show_frustum` / `--show_contacts` / `--show_imu` / `--show_caption` / `--show_slam_pc` | `True` | Toggle individual visualization layers. |

</details>

## 🧩 Usage (Python API)

Run from the package root (so `data_loader`, `visualization`, and `utils` are importable), or add the package root to `sys.path` as the examples do.

### Load an annotation

```python
from data_loader import load_from_annotation_hdf5, list_annotation_contents

# Inspect structure without loading arrays
for name, info in sorted(list_annotation_contents("episode/annotation.hdf5").items()):
    print(name, info)

# Load frames [start_idx, end_idx); end_idx=None or -1 loads all frames
ann = load_from_annotation_hdf5("episode/annotation.hdf5", start_idx=0, end_idx=None)

print(len(ann["img_names"]), "frames")
print(ann["R_c2w_all"].shape, ann["t_c2w_all"].shape)   # camera->world (N,3,3), (N,3)
print(ann["hand_left_joints"].shape)                      # (N, 21, 3), camera frame
print(ann["smplh_body_joints"].shape)                     # (N, 52, 3), world frame
print(ann["caption_main_task"])                           # main task string
```

`load_from_annotation_hdf5` returns a dict with the keys: `calib_data`, `R_c2w_all`, `t_c2w_all`, `img_names`, `depth_loader`, `depth_min`, `depth_max`, `depth_num_frames`, `hand_left_joints`, `hand_right_joints`, `smplh_body_joints`, `contacts`, `imu_ts`, `imu_accel_xyz`, `imu_gyro_xyz`, `imu_keyframe_indices`, `ground_height`, `slam_point_cloud`, and the caption fields (`caption_main_task`, `caption_frame_info_map`, `caption_segment_boundaries`, `caption_task_to_id`).

### Lazy depth + point cloud

`depth_loader` reads one frame at a time so depth is never fully loaded into memory:

```python
from visualization import depth_to_colormap, depth_to_pointcloud, transform_points_to_world

depth, confidence = ann["depth_loader"](0)          # frame 0
colormap = depth_to_colormap(depth, ann["depth_min"], ann["depth_max"])  # (H, W, 3) RGB

# Back-project to a colored point cloud, then move it into the world frame
K = ...  # 3x3 intrinsics from ann["calib_data"]["cam01"]["K"]
points, colors = depth_to_pointcloud(depth, K, confidence=confidence)
points_world = transform_points_to_world(points, ann["R_c2w_all"][0], ann["t_c2w_all"][0])
```

### Skeletons and video frames

```python
from data_loader import load_video_frame, MANO_PARENT_INDICES, SMPL_H_BODY_PARENT_INDICES
from visualization import build_line3d_skeleton

# Hands use MANO parents (parent_indices[0] == -1); body adds 1 for the root joint
hand_lines = build_line3d_skeleton(ann["hand_left_joints"][0], MANO_PARENT_INDICES, plus_one=False)
body_lines = build_line3d_skeleton(ann["smplh_body_joints"][0], SMPL_H_BODY_PARENT_INDICES, plus_one=True)

# Decode a single RGB frame from a video (frame_idx aligns with annotation frames)
rgb = load_video_frame("episode/stereo_left.mp4", frame_idx=0)
```

## ✂️ Trim a Recording

Trim an episode to its longest contiguous span of valid annotations (frames with NaN/all-zero body keypoints or SLAM poses are dropped). This rewrites both the HDF5 and the videos:

```bash
# Preview the trim range without writing anything
python examples/trim_recording.py --data_root /path/to/episode --dry_run

# Write trimmed episode (default output: <data_root>_trimmed)
python examples/trim_recording.py --data_root /path/to/episode --output_root /path/to/out
```

> Requires `ffmpeg` on your `PATH` for video trimming.

