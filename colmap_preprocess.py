import os
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

def read_colmap_camera(cameras_txt_path):
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                fx, fy, cx, cy = map(float, parts[4:8])
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]
                ], dtype=np.float32)
                return K
    raise ValueError("Could not parse camera intrinsics.")

def read_colmap_poses(images_txt_path):
    poses = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or len(lines[i].strip()) == 0:
            i += 1
            continue
        parts = lines[i].strip().split()
        q = list(map(float, parts[1:5]))  # qw, qx, qy, qz
        t = list(map(float, parts[5:8]))
        image_name = parts[9]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # x,y,z,w
        poses[image_name] = (rot.as_matrix(), np.array(t))
        i += 2
    return poses

def project_points(K, R_dst, t_dst, R_src, t_src, points_2D, depth=1.0):
    projected_pts = []
    for pt in points_2D:
        pt_h = np.array([pt[0], pt[1], 1.0])
        x_norm = np.linalg.inv(K) @ pt_h
        X_cam_src = x_norm * depth
        X_world = R_src @ X_cam_src + t_src
        X_cam_dst = R_dst.T @ (X_world - t_dst)
        x_proj = K @ X_cam_dst
        x_proj = (x_proj[:2] / x_proj[2])
        projected_pts.append(x_proj)
    return np.array(projected_pts, dtype=np.float32)

def smooth_trajectory(points, window_length=21, polyorder=2):
    points = np.array(points)
    smoothed = np.zeros_like(points)
    for i in range(points.shape[1]):
        for j in range(2):  # x and y
            coord = points[:, i, j]
            if len(coord) < window_length:
                window_length = max(3, len(coord) // 2 * 2 + 1)
            smoothed[:, i, j] = savgol_filter(coord, window_length, polyorder)
    return smoothed

def compute_homographies(src_pts, traj_pts):
    src = np.array(src_pts).reshape(-1, 1, 2).astype(np.float32)
    homographies = []
    for i, dst in enumerate(traj_pts):
        dst = np.array(dst).reshape(-1, 1, 2).astype(np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        if H is None:
            H = np.eye(3)
        homographies.append({"frame_idx": i, "homography_matrix": H.tolist()})
    return homographies

# Main wrapper
def run_colmap_preprocess(colmap_sparse_dir, frames_dir, insert_pts_2d, output_json_path):
    cameras_txt = os.path.join(colmap_sparse_dir, "cameras.txt")
    images_txt = os.path.join(colmap_sparse_dir, "images.txt")

    K = read_colmap_camera(cameras_txt)
    poses = read_colmap_poses(images_txt)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
    R0, t0 = poses[frame_files[0]]

    all_traj = []
    for fname in frame_files:
        if fname not in poses:
            continue
        R_dst, t_dst = poses[fname]
        proj = project_points(K, R_dst, t_dst, R0, t0, insert_pts_2d, depth=1.0)
        all_traj.append(proj)

    smooth_traj = smooth_trajectory(all_traj)
    homographies = compute_homographies(insert_pts_2d, smooth_traj)

    with open(output_json_path, 'w') as f:
        json.dump(homographies, f, indent=2)

    return homographies, smooth_traj, K

if __name__ == "__main__":
    colmap_sparse_dir = "path/to/colmap/sparse"
    frames_dir = "path/to/frames"
    insert_pts_2d = np.array([[100, 200], [150, 250], [200, 300]])  
    output_json_path = "output/homographies.json"

    homographies, smooth_traj, K = run_colmap_preprocess(colmap_sparse_dir, frames_dir, insert_pts_2d, output_json_path)
    print("Homographies and smoothed trajectory computed successfully.")