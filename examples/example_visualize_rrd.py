"""
Example: Load Xperience-10M data and visualize to Rerun .rrd.

Run from package root:
  python examples/example_visualize_rrd.py --data_root /path/to/episode
  Then open: rerun vis.rrd
"""

import os
import sys
import argparse
from pathlib import Path

# Suppress Rerun internal
os.environ.setdefault("RUST_LOG", "warn,re_chunk=error")

import cv2

# Add package root to path so we can import data_loader, visualization, utils
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import rerun as rr
import tqdm
from scipy.spatial.transform import Rotation as R

from data_loader import (
    load_from_annotation_hdf5,
    get_fisheye_T_world_cam,
    load_video_frame,
    MANO_PARENT_INDICES,
    SMPL_H_BODY_PARENT_INDICES,
)
from visualization import (
    create_blueprint,
    depth_to_colormap,
    depth_to_pointcloud,
    build_line3d_skeleton,
    scale_image,
    transform_points_to_world,
)


def main():
    parser = argparse.ArgumentParser(description="Example: visualize Xperience-10M data to Rerun RRD.")
    parser.add_argument("--data_root", type=str, required=True, help="Episode folder (contains annotation.hdf5)")
    parser.add_argument("--output_rrd", type=str, default="vis.rrd", help="Output .rrd path")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of frames to log")

    # Input file names (relative to --data_root, or absolute paths)
    parser.add_argument("--annotation", type=str, default="annotation.hdf5", help="Annotation HDF5 file name")
    parser.add_argument("--stereo_left", type=str, default="stereo_left.mp4", help="Stereo left video file name")
    parser.add_argument("--stereo_right", type=str, default="stereo_right.mp4", help="Stereo right video file name")
    parser.add_argument("--fisheye_pattern", type=str, default="fisheye_{cam}.mp4", help="Fisheye video file name pattern ('{cam}' -> cam0..cam3)")

    # show_* (align with run_vis.py)
    parser.add_argument("--show_fisheye", action="store_true", default=True, help="Show fisheye camera images")
    parser.add_argument("--show_stereo", action="store_true", default=True, help="Show stereo camera images")
    parser.add_argument("--show_depth_colormap", action="store_true", default=True, help="Show depth colormap")
    parser.add_argument("--show_depth_points", action="store_true", default=True, help="Show depth point cloud in 3D")
    parser.add_argument("--show_skeleton", action="store_true", default=True, help="Show hand and body skeleton")
    parser.add_argument("--show_frustum", action="store_true", default=True, help="Show camera frustums")
    parser.add_argument("--show_contacts", action="store_true", default=True, help="Show foot contacts")
    parser.add_argument("--show_imu", action="store_true", default=True, help="Show IMU accel/gyro time series")
    parser.add_argument("--show_caption", action="store_true", default=True, help="Show caption panels (from annotation.hdf5 caption/captions dataset only)")
    parser.add_argument("--show_slam_pc", action="store_true", default=True, help="Show SLAM point cloud (static)")
    
    args = parser.parse_args()

    FPS = 10.0
    LOG_IMAGE_SCALE = 0.5
    MAX_DEPTH_POINTS = 20000
    NEAR_PLANE = 0.6
    FAR_PLANE = 5.0

    # Print arguments in order, with colors (ANSI)
    _c = lambda s, code: f"\033[{code}m{s}\033[0m"
    _key = lambda k: _c(k, "36")   # cyan
    _val = lambda v: _c(str(v), "33")  # yellow
    _sec = lambda s: _c(s, "32")   # green
    order = [
        "data_root", "output_rrd", "num_frames",
        "annotation", "stereo_left", "stereo_right", "fisheye_pattern",
        "show_stereo", "show_fisheye", "show_depth_colormap", "show_depth_points",
        "show_skeleton", "show_frustum", "show_contacts", "show_imu", "show_caption", "show_slam_pc",
    ]
    args_dict = vars(args)
    print(_sec("Arguments:"))
    for k in order:
        if k in args_dict:
            print(f"  {_key(k)}: {_val(args_dict[k])}")
    for k in sorted(args_dict):
        if k not in order:
            print(f"  {_key(k)}: {_val(args_dict[k])}")

    data_root = Path(args.data_root)
    annotation_path = data_root / args.annotation
    if not annotation_path.exists():
        print(f"Annotation not found: {annotation_path}")
        sys.exit(1)

    end_idx = min(args.num_frames, 999999)
    ann = load_from_annotation_hdf5(str(annotation_path), 0, end_idx)
    n_frames = len(ann["img_names"])
    if n_frames == 0:
        print("No frames in annotation slice.")
        sys.exit(1)

    calib_data = ann["calib_data"]
    R_c2w_all = ann["R_c2w_all"]
    t_c2w_all = ann["t_c2w_all"]

    cam01 = calib_data.get("cam01") if calib_data else None
    if cam01 is None:
        print("Calibration cam01 not found.")
        sys.exit(1)
    K_arr = np.asarray(cam01["K"]).flatten()
    fx, fy, cx, cy = float(K_arr[0]), float(K_arr[1]), float(K_arr[2]), float(K_arr[3])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    W_img, H_img = 512, 512  # for frustum

    depth_loader = ann.get("depth_loader")
    depth_min = ann.get("depth_min", 0.0)
    depth_max = ann.get("depth_max", 1.0)
    depth_num_frames = ann.get("depth_num_frames", 0)

    ground_height = ann.get("ground_height", -1.75)

    caption_main_task = ann.get("caption_main_task", "")
    caption_frame_info_map = ann.get("caption_frame_info_map")
    caption_segment_boundaries = ann.get("caption_segment_boundaries", [])
    caption_task_to_id = ann.get("caption_task_to_id", {})
    if args.show_caption and caption_frame_info_map is None:
        print("Caption requested but no caption data found in annotation.hdf5 (caption/captions dataset).")

    hand_left_joints = ann.get("hand_left_joints")
    hand_right_joints = ann.get("hand_right_joints")
    smplh_body_joints = ann.get("smplh_body_joints")
    contacts = ann.get("contacts") if args.show_contacts else None

    imu_accel = ann.get("imu_accel_xyz")
    imu_gyro = ann.get("imu_gyro_xyz")
    imu_keyframe_indices = ann.get("imu_keyframe_indices")
    n_imu = 0
    if args.show_imu and imu_accel is not None and imu_gyro is not None:
        n_imu = len(imu_keyframe_indices) if imu_keyframe_indices is not None else len(imu_accel)

    def _shp(x):
        if x is None:
            return "None"
        if hasattr(x, "shape"):
            return str(tuple(x.shape))
        if isinstance(x, (list, tuple)):
            return f"({len(x)},)"
        if isinstance(x, (int, float)):
            return "()"
        return "()"

    print(_sec("Loaded") + " data shapes:")
    print(f"  {_key('img_names')}: {_val(_shp(ann.get('img_names')))}")
    print(f"  {_key('R_c2w_all')}: {_val(_shp(R_c2w_all))}")
    print(f"  {_key('t_c2w_all')}: {_val(_shp(t_c2w_all))}")
    print(f"  {_key('ground_height')}: {_val(ground_height)}")
    print(f"  {_key('depth_frames')}: {_val(f'({depth_num_frames},)')}")
    print(f"  {_key('hand_left_joints')}: {_val(_shp(hand_left_joints))}")
    print(f"  {_key('hand_right_joints')}: {_val(_shp(hand_right_joints))}")
    print(f"  {_key('smplh_body_joints')}: {_val(_shp(smplh_body_joints))}")
    print(f"  {_key('contacts')}: {_val(_shp(contacts))}")
    print(f"  {_key('imu_accel_xyz')}: {_val(_shp(imu_accel))}")
    print(f"  {_key('imu_gyro_xyz')}: {_val(_shp(imu_gyro))}")
    print(f"  {_key('imu_keyframe_indices')}: {_val(_shp(imu_keyframe_indices))}")
    n_seg = len(caption_segment_boundaries) if caption_segment_boundaries else 0
    print(f"  {_key('caption_segments')}: {_val(f'({n_seg},)')}")

    rr.init("Xperience-10M")
    rr.send_blueprint(create_blueprint(
        show_fisheye=args.show_fisheye,
        show_stereo=args.show_stereo,
        show_depth_colormap=args.show_depth_colormap,
        ground_height=ground_height,
        show_imu=args.show_imu,
        show_caption=args.show_caption,
        show_3d_view=True,
    ))
    output_path = Path(args.output_rrd)
    if not output_path.is_absolute():
        output_path = PACKAGE_ROOT / output_path
    rr.save(str(output_path))
    rr.set_time("stable_time", duration=0)

    # SLAM point cloud (static, same as run_vis)
    if args.show_slam_pc:
        slam_points = ann.get("slam_point_cloud")
        if slam_points is not None and len(slam_points) > 0:
            rr.log("world/slam_point_cloud", rr.Points3D(slam_points, colors=[220, 220, 220], radii=0.002), static=True)

    # IMU static series (same as run_vis)
    if args.show_imu and n_imu > 0:
        for name, color in [("x", [255, 0, 0]), ("y", [0, 255, 0]), ("z", [0, 0, 255])]:
            rr.log(f"imu/accel/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)
            rr.log(f"imu/gyro/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)

    # Task timeline (Gantt-style bars from segment_boundaries, same as run_vis)
    if args.show_caption and caption_frame_info_map is not None and caption_segment_boundaries and caption_task_to_id:
        def _get_task_color(idx):
            colors = [
                [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [50, 255, 255],
                [255, 50, 255], [255, 128, 0], [128, 0, 128], [0, 128, 0], [0, 0, 128],
                [128, 128, 0], [128, 0, 0], [0, 128, 128],
            ]
            return colors[idx % len(colors)]

        runs = []
        for start_f, end_f, task_name, _ in caption_segment_boundaries:
            if runs and runs[-1]["task"] == task_name:
                runs[-1]["end"] = max(runs[-1]["end"], end_f)
            else:
                runs.append({"task": task_name, "start": start_f, "end": end_f})

        BAND_LINES = 1
        labeled_tasks = set()
        for run_idx, run in enumerate(runs):
            task_name = run["task"]
            task_id = caption_task_to_id.get(task_name, run_idx + 1)
            base_color = _get_task_color(task_id - 1)
            for i in range(BAND_LINES):
                entity_path = f"timeline/run_{run_idx:03d}/line_{i}"
                label = ""
                if i == BAND_LINES // 2 and task_name not in labeled_tasks:
                    label = task_name
                    labeled_tasks.add(task_name)
                try:
                    rr.log(entity_path, rr.SeriesLines(colors=base_color, widths=3, names=label), static=True)
                except TypeError:
                    rr.log(entity_path, rr.SeriesLines(colors=base_color, widths=3, names=label))

        for run_idx, run in enumerate(runs):
            for frame_idx in range(run["start"], min(run["end"] + 1, n_frames)):
                rr.set_time("stable_time", duration=float(frame_idx / FPS))
                for i in range(BAND_LINES):
                    entity_path = f"timeline/run_{run_idx:03d}/line_{i}"
                    rr.log(entity_path, rr.Scalars(0.0 + i * 0.1))

    rr.set_time("stable_time", duration=0)

    stereo_left_path = str(data_root / args.stereo_left) if args.show_stereo and (data_root / args.stereo_left).exists() else None
    stereo_right_path = str(data_root / args.stereo_right) if args.show_stereo and (data_root / args.stereo_right).exists() else None

    # AssetVideo + VideoFrameReference: avoid decoding every frame (same as run_vis.py), much faster.
    stereo_left_asset = None
    stereo_right_asset = None
    fisheye_assets = {}
    stereo_left_num_frames = n_frames
    stereo_right_num_frames = n_frames
    stereo_left_fps = FPS
    stereo_right_fps = FPS
    fisheye_info = {}  # cam_id -> {path, num_frames, fps}

    if args.show_stereo and stereo_left_path:
        cap = cv2.VideoCapture(stereo_left_path)
        if cap.isOpened():
            stereo_left_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
            stereo_left_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
            cap.release()
        try:
            stereo_left_asset = rr.AssetVideo(path=str(Path(stereo_left_path).resolve()))
            rr.log("world/stereo/vis_cam0/video", stereo_left_asset, static=True)
        except Exception:
            stereo_left_asset = None

    if args.show_stereo and stereo_right_path:
        cap = cv2.VideoCapture(stereo_right_path)
        if cap.isOpened():
            stereo_right_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
            stereo_right_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
            cap.release()
        try:
            stereo_right_asset = rr.AssetVideo(path=str(Path(stereo_right_path).resolve()))
            rr.log("world/stereo/vis_cam1/video", stereo_right_asset, static=True)
        except Exception:
            stereo_right_asset = None

    if args.show_fisheye:
        for cid in ["cam0", "cam1", "cam2", "cam3"]:
            vpath = data_root / args.fisheye_pattern.format(cam=cid)
            if vpath.exists():
                vstr = str(vpath.resolve())
                cap = cv2.VideoCapture(vstr)
                nf, fps = n_frames, FPS
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
                    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
                    cap.release()
                fisheye_info[cid] = {"path": vstr, "num_frames": nf, "fps": fps}
                try:
                    fisheye_assets[cid] = rr.AssetVideo(path=vstr)
                    rr.log(f"world/fisheye/{cid}/video", fisheye_assets[cid], static=True)
                except Exception:
                    fisheye_assets[cid] = None

    last_caption_interaction = ""
    last_caption_objects_str = ""

    for frame_idx in tqdm.tqdm(range(n_frames), desc="Frames", unit="frame"):
        t = float(frame_idx / FPS)
        rr.set_time("stable_time", duration=t)

        # Caption panels (Main Task, Sub Task, Current Action, Interaction, Objects) at this time
        if args.show_caption:
            if caption_frame_info_map is not None:
                info = caption_frame_info_map.get(frame_idx, {})
                rr.log(
                    "captions/Main_Task",
                    rr.TextDocument(f"**{caption_main_task}**", media_type=rr.MediaType.MARKDOWN),
                )
                theme = info.get("theme", "N/A")
                rr.log(
                    "captions/Sub_Task",
                    rr.TextDocument(f"**{theme}**", media_type=rr.MediaType.MARKDOWN),
                )
                action_label = info.get("action_label", "N/A")
                action_desc = info.get("action_desc", "")
                action_text = f"**{action_label}**\n\n" + (f"> {action_desc}" if action_desc else "")
                rr.log(
                    "captions/Current_Action",
                    rr.TextDocument(action_text, media_type=rr.MediaType.MARKDOWN),
                )
                if info.get("interaction") is not None:
                    last_caption_interaction = info["interaction"] or "N/A"
                rr.log(
                    "captions/details/interaction",
                    rr.TextDocument(f"**{last_caption_interaction}**", media_type=rr.MediaType.MARKDOWN),
                )
                if info.get("objects") is not None:
                    objs = info["objects"]
                    last_caption_objects_str = "\n".join([f"- {o}" for o in (objs if isinstance(objs, list) else [str(objs)])])
                rr.log(
                    "captions/details/objects",
                    rr.TextDocument(last_caption_objects_str or "N/A", media_type=rr.MediaType.MARKDOWN),
                )
            else:
                rr.log("captions/Main_Task", rr.TextDocument("**N/A**", media_type=rr.MediaType.MARKDOWN))
                rr.log("captions/Sub_Task", rr.TextDocument("**N/A**", media_type=rr.MediaType.MARKDOWN))
                rr.log("captions/Current_Action", rr.TextDocument("**N/A**", media_type=rr.MediaType.MARKDOWN))
                rr.log("captions/details/interaction", rr.TextDocument("**N/A**", media_type=rr.MediaType.MARKDOWN))
                rr.log("captions/details/objects", rr.TextDocument("N/A", media_type=rr.MediaType.MARKDOWN))

        R_c2w = R_c2w_all[frame_idx]
        t_c2w = t_c2w_all[frame_idx]

        # Depth colormap and point cloud (camera -> world)
        if (args.show_depth_colormap or args.show_depth_points) and depth_loader is not None and frame_idx < depth_num_frames:
            depth_frame, conf_frame = depth_loader(frame_idx)
            if depth_frame is not None:
                if args.show_depth_colormap:
                    depth_cmap = depth_to_colormap(depth_frame, depth_min, depth_max)
                    rr.log("world/depth/vis", rr.Image(scale_image(depth_cmap, LOG_IMAGE_SCALE)))
                if args.show_depth_points:
                    rgb = load_video_frame(stereo_left_path, frame_idx, LOG_IMAGE_SCALE) if stereo_left_path else None
                    points, colors = depth_to_pointcloud(
                        depth_frame, K,
                        rgb_image=rgb,
                        downsample_factor=4,
                        max_points=MAX_DEPTH_POINTS,
                        near_plane=NEAR_PLANE,
                        far_plane=FAR_PLANE,
                        confidence=conf_frame,
                        confidence_threshold=0.2,
                    )
                    if len(points) > 0:
                        points_world = transform_points_to_world(points, R_c2w, t_c2w)
                        rr.log("world/depth/points", rr.Points3D(points_world, colors=colors, radii=0.005))

        # Hand: camera frame -> transform to world (same as run_vis)
        if args.show_skeleton and hand_left_joints is not None and frame_idx < len(hand_left_joints):
            left_j = hand_left_joints[frame_idx]
            left_j_world = transform_points_to_world(left_j, R_c2w, t_c2w)
            if not np.allclose(left_j_world, 0.0, atol=1e-6):
                rr.log("world/hand_mocap/left_hand/joints", rr.Points3D(left_j_world, colors=[77, 188, 94], radii=0.005))
                lines = build_line3d_skeleton(left_j_world, MANO_PARENT_INDICES, plus_one=False)
                if len(lines) > 0:
                    rr.log("world/hand_mocap/left_hand/skeleton", rr.LineStrips3D(lines, colors=[77, 188, 94], radii=0.003))
            else:
                rr.log("world/hand_mocap/left_hand/joints", rr.Points3D(np.array([])))
                rr.log("world/hand_mocap/left_hand/skeleton", rr.LineStrips3D(np.array([])))

        if args.show_skeleton and hand_right_joints is not None and frame_idx < len(hand_right_joints):
            right_j = hand_right_joints[frame_idx]
            right_j_world = transform_points_to_world(right_j, R_c2w, t_c2w)
            if not np.allclose(right_j_world, 0.0, atol=1e-6):
                rr.log("world/hand_mocap/right_hand/joints", rr.Points3D(right_j_world, colors=[91, 77, 188], radii=0.005))
                lines = build_line3d_skeleton(right_j_world, MANO_PARENT_INDICES, plus_one=False)
                if len(lines) > 0:
                    rr.log("world/hand_mocap/right_hand/skeleton", rr.LineStrips3D(lines, colors=[91, 77, 188], radii=0.003))
            else:
                rr.log("world/hand_mocap/right_hand/joints", rr.Points3D(np.array([])))
                rr.log("world/hand_mocap/right_hand/skeleton", rr.LineStrips3D(np.array([])))

        # Body: already world frame in HDF5 -> use as-is (no transform, same as run_vis)
        if args.show_skeleton and smplh_body_joints is not None and frame_idx < len(smplh_body_joints):
            body_j = smplh_body_joints[frame_idx]
            if body_j.shape[0] >= 22 and not np.any(np.isnan(body_j)) and not np.allclose(body_j, 0.0, atol=1e-6):
                rr.log("world/smplh/joints", rr.Points3D(body_j, colors=[255, 165, 0], radii=0.01))
                lines = build_line3d_skeleton(body_j, SMPL_H_BODY_PARENT_INDICES, plus_one=True)
                if len(lines) > 0:
                    rr.log("world/smplh/skeleton", rr.LineStrips3D(lines, colors=[255, 165, 0], radii=0.005))
            else:
                rr.log("world/smplh/joints", rr.Points3D(np.array([])))
                rr.log("world/smplh/skeleton", rr.LineStrips3D(np.array([])))

        # Contacts (body joints already world; same indices as run_vis)
        if args.show_contacts and args.show_skeleton and contacts is not None and frame_idx < len(contacts):
            foot_contact_indices = [6, 7, 9, 10]
            foot_joint_indices = [7, 8, 10, 11]
            foot_joint_names = ["left_ankle", "right_ankle", "left_foot", "right_foot"]
            contacts_frame = contacts[frame_idx]
            if (not np.any(np.isnan(contacts_frame)) and len(contacts_frame) >= 11
                    and smplh_body_joints is not None and frame_idx < len(smplh_body_joints)):
                body_j = smplh_body_joints[frame_idx]
                for contact_idx, joint_idx, joint_name in zip(foot_contact_indices, foot_joint_indices, foot_joint_names):
                    contact_val = float(contacts_frame[contact_idx]) if contact_idx < len(contacts_frame) else 0.0
                    color = [0, int(255 * contact_val), int(255 * (1 - contact_val))]
                    rad = 0.01 + 0.02 * contact_val
                    rr.log(f"world/full_body_mocap/contacts/{joint_name}", rr.Points3D(
                        body_j[joint_idx:joint_idx + 1], colors=color, radii=rad
                    ))
                    if contact_val > 0.1 and ground_height is not None:
                        jp = body_j[joint_idx]
                        fp = np.array([jp[0], jp[1], ground_height])
                        rr.log(f"world/full_body_mocap/contacts/{joint_name}/line_to_floor",
                               rr.LineStrips3D(np.array([[[jp, fp]]]), colors=color, radii=0.003))
                    else:
                        rr.log(f"world/full_body_mocap/contacts/{joint_name}/line_to_floor", rr.LineStrips3D(np.array([])))
            else:
                for joint_name in ["left_ankle", "right_ankle", "left_foot", "right_foot"]:
                    rr.log(f"world/full_body_mocap/contacts/{joint_name}", rr.Points3D(np.array([])))
                    rr.log(f"world/full_body_mocap/contacts/{joint_name}/line_to_floor", rr.LineStrips3D(np.array([])))
        elif not args.show_contacts:
            for joint_name in ["left_ankle", "right_ankle", "left_foot", "right_foot"]:
                rr.log(f"world/full_body_mocap/contacts/{joint_name}", rr.Points3D(np.array([])))
                rr.log(f"world/full_body_mocap/contacts/{joint_name}/line_to_floor", rr.LineStrips3D(np.array([])))

        # IMU per frame
        if args.show_imu and imu_accel is not None and n_imu > 0:
            imu_i = int(imu_keyframe_indices[frame_idx]) if imu_keyframe_indices is not None else frame_idx
            imu_i = min(max(imu_i, 0), imu_accel.shape[0] - 1)
            rr.log("imu/accel/x", rr.Scalars(float(imu_accel[imu_i, 0])))
            rr.log("imu/accel/y", rr.Scalars(float(imu_accel[imu_i, 1])))
            rr.log("imu/accel/z", rr.Scalars(float(imu_accel[imu_i, 2])))
            rr.log("imu/gyro/x", rr.Scalars(float(imu_gyro[imu_i, 0])))
            rr.log("imu/gyro/y", rr.Scalars(float(imu_gyro[imu_i, 1])))
            rr.log("imu/gyro/z", rr.Scalars(float(imu_gyro[imu_i, 2])))

        # Camera frustums (same as run_vis: stereo left/right + fisheye cam0-3)
        if args.show_frustum:
            T_c2w_cam = np.eye(4)
            T_c2w_cam[:3, :3] = R_c2w
            T_c2w_cam[:3, 3] = t_c2w
            cam_transform = rr.Transform3D(
                translation=T_c2w_cam[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(R_c2w).as_quat()),
            )
            rr.log("world/stereo/left/camera", cam_transform)
            rr.log("world/stereo/left/camera", rr.Pinhole(image_from_camera=K, resolution=[W_img, H_img], image_plane_distance=0.08))
            # stereo_right: baseline offset from left
            cam01_data = calib_data.get("cam01") if calib_data else None
            baseline = float(cam01_data.get("baseline", 0.0)) if cam01_data else 0.0
            baseline_dir = (np.linalg.inv(R_c2w) @ np.array([1, 0, 0]).reshape(3, 1)).flatten()
            baseline_dir = baseline_dir / (np.linalg.norm(baseline_dir) + 1e-8)
            t_right = t_c2w + baseline_dir * baseline
            T_right = np.eye(4)
            T_right[:3, :3] = R_c2w
            T_right[:3, 3] = t_right
            rr.log("world/stereo/right/camera", rr.Transform3D(translation=t_right, rotation=rr.Quaternion(xyzw=R.from_matrix(R_c2w).as_quat())))
            rr.log("world/stereo/right/camera", rr.Pinhole(image_from_camera=K, resolution=[W_img, H_img], image_plane_distance=0.08))
            fisheye_poses = get_fisheye_T_world_cam(calib_data, R_c2w, t_c2w)
            for cid in ["cam0", "cam1", "cam2", "cam3"]:
                if cid in fisheye_poses:
                    T_f = fisheye_poses[cid]
                    rr.log(f"world/fisheye/{cid}/camera", rr.Transform3D(
                        translation=T_f[:3, 3],
                        rotation=rr.Quaternion(xyzw=R.from_matrix(T_f[:3, :3]).as_quat()),
                    ))
                    rr.log(f"world/fisheye/{cid}/camera", rr.Pinhole(image_from_camera=K, resolution=[W_img, H_img], image_plane_distance=0.08))
        else:
            for path in ["world/stereo/left/camera", "world/stereo/right/camera",
                         "world/fisheye/cam0/camera", "world/fisheye/cam1/camera", "world/fisheye/cam2/camera", "world/fisheye/cam3/camera"]:
                rr.log(path, rr.Clear(recursive=True))

        # Stereo/fisheye: use VideoFrameReference when AssetVideo is available (no decode), else fallback
        if args.show_stereo and frame_idx < stereo_left_num_frames:
            if stereo_left_asset is not None:
                t_sec = frame_idx / stereo_left_fps if stereo_left_fps > 0 else frame_idx / FPS
                try:
                    rr.log(
                        "world/stereo/vis_cam0",
                        rr.VideoFrameReference(seconds=t_sec, video_reference="world/stereo/vis_cam0/video"),
                    )
                except Exception:
                    stereo_left_asset = None
                    frame = load_video_frame(stereo_left_path, frame_idx, LOG_IMAGE_SCALE)
                    if frame is not None:
                        rr.log("world/stereo/vis_cam0", rr.Image(frame))
            elif stereo_left_path:
                frame = load_video_frame(stereo_left_path, frame_idx, LOG_IMAGE_SCALE)
                if frame is not None:
                    rr.log("world/stereo/vis_cam0", rr.Image(frame))
        if args.show_stereo and stereo_right_path and frame_idx < stereo_right_num_frames:
            if stereo_right_asset is not None:
                t_sec = frame_idx / stereo_right_fps if stereo_right_fps > 0 else frame_idx / FPS
                try:
                    rr.log(
                        "world/stereo/vis_cam1",
                        rr.VideoFrameReference(seconds=t_sec, video_reference="world/stereo/vis_cam1/video"),
                    )
                except Exception:
                    stereo_right_asset = None
                    frame = load_video_frame(stereo_right_path, frame_idx, LOG_IMAGE_SCALE)
                    if frame is not None:
                        rr.log("world/stereo/vis_cam1", rr.Image(frame))
            else:
                frame = load_video_frame(stereo_right_path, frame_idx, LOG_IMAGE_SCALE)
                if frame is not None:
                    rr.log("world/stereo/vis_cam1", rr.Image(frame))
        if args.show_fisheye:
            for cid in ["cam0", "cam1", "cam2", "cam3"]:
                if cid not in fisheye_info or frame_idx >= fisheye_info[cid]["num_frames"]:
                    continue
                info = fisheye_info[cid]
                if fisheye_assets.get(cid) is not None:
                    t_sec = frame_idx / info["fps"] if info["fps"] > 0 else frame_idx / FPS
                    try:
                        rr.log(
                            f"world/fisheye/{cid}",
                            rr.VideoFrameReference(seconds=t_sec, video_reference=f"world/fisheye/{cid}/video"),
                        )
                    except Exception:
                        fisheye_assets[cid] = None
                        frame = load_video_frame(info["path"], frame_idx, LOG_IMAGE_SCALE)
                        if frame is not None:
                            rr.log(f"world/fisheye/{cid}", rr.Image(frame))
                else:
                    frame = load_video_frame(info["path"], frame_idx, LOG_IMAGE_SCALE)
                    if frame is not None:
                        rr.log(f"world/fisheye/{cid}", rr.Image(frame))

    print(f"{_sec('Saved')} to {_val(str(output_path))}")
    print(f"{_sec('Open with')}: {_val('rerun ' + str(output_path))}")


if __name__ == "__main__":
    main()
