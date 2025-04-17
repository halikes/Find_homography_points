import cv2
import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_002_7s.mp4', help='Path to target video')
parser.add_argument('--object_file', type=str, default='data/source.png', help='Path to object image')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=1080, help='Width to resize the first frame')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 交互式对象插入模块 -----------------------
def nothing(x):
    pass

def insertion_mouse_callback(event, x, y, flags, param):
    # 用于交互式调整插入位置和大小
    global mouse_coord
    mouse_coord = (x, y)
    (ox, oy, ow, oh) = param['obj_bbox']
    if event == cv2.EVENT_LBUTTONDOWN:
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            param['dragging'] = True
            param['drag_offset'] = [x - ox, y - oy]
    elif event == cv2.EVENT_MOUSEMOVE:
        if param.get('dragging', False):
            dx, dy = param['drag_offset']
            param['obj_pos'][0] = x - dx
            param['obj_pos'][1] = y - dy
    elif event == cv2.EVENT_LBUTTONUP:
        param['dragging'] = False

def interactive_insertion(video_file, object_file, resize_width, output_dir):
    # 读取视频第一帧
    cap = cv2.VideoCapture(video_file)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read video file: " + video_file)
    cap.release()

    # Resize 第一帧
    h0, w0 = first_frame.shape[:2]
    scale = resize_width / float(w0)
    first_frame = cv2.resize(first_frame, (resize_width, int(h0 * scale)))
    target_frame = first_frame.copy()
    orig_frame = first_frame.copy()

    # 加载对象图像（转换为 BGR 格式）
    object_img = np.array(Image.open(object_file).convert('RGB'))
    object_img = cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR)

    orig_obj_h, orig_obj_w = object_img.shape[:2]
    obj_pos = [50, 50]
    global scale_factor
    scale_factor = 1.0

    param = {
        'obj_pos': obj_pos,
        'obj_bbox': (obj_pos[0], obj_pos[1], orig_obj_w, orig_obj_h),
        'dragging': False,
        'drag_offset': [0, 0]
    }

    window_name = "Insertion Adjustment"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, insertion_mouse_callback, param)
    cv2.createTrackbar("Scale", window_name, 100, 200, nothing)

    while True:
        scale_val = cv2.getTrackbarPos("Scale", window_name) / 100.0
        scale_factor = scale_val
        new_w = int(orig_obj_w * scale_factor)
        new_h = int(orig_obj_h * scale_factor)
        cur_obj_img = cv2.resize(object_img, (new_w, new_h))
        param['obj_bbox'] = (param['obj_pos'][0], param['obj_pos'][1], new_w, new_h)

        display_img = target_frame.copy()
        x, y = param['obj_pos']
        H_disp, W_disp = display_img.shape[:2]
        x = max(0, min(x, W_disp - new_w))
        y = max(0, min(y, H_disp - new_h))
        param['obj_pos'][0] = x
        param['obj_pos'][1] = y

        # 融合对象图像到目标帧（直接覆盖）
        display_img[y:y + new_h, x:x + new_w] = cur_obj_img
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

    fused_img = display_img.copy()

    # source_img
    source_img = np.zeros_like(fused_img)
    source_img[y:y + new_h, x:x + new_w] = cur_obj_img
    #roi = source_img[y:y + new_h, x:x + new_w]
    #roi_smoothed = cv2.bilateralFilter(roi, d=5, sigmaColor=75, sigmaSpace=75)
    #source_img[y:y + new_h, x:x + new_w] = roi_smoothed  # 注意这里确保区域尺寸一致
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255

    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }

    cv2.imwrite(os.path.join(output_dir, "source_img_sample_video_02_7s_1_new.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "mask_img_sample_video_02_7s_1_new.png"), mask_img)

    return fused_img, source_img, mask_img, final_params, orig_frame

# ----------------------- SIFT+边缘校正模块 -----------------------
manual_points = []  # 用于存放用户点击的四个点

def sift_mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Selected point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to proceed.")

def find_nearest_edge_point(point, edge_image):
    indices = np.argwhere(edge_image > 0)
    if len(indices) == 0:
        return point
    distances = np.sqrt((indices[:, 1] - point[0]) ** 2 + (indices[:, 0] - point[1]) ** 2)
    idx = np.argmin(distances)
    best_y, best_x = indices[idx]
    return (int(best_x), int(best_y))

def select_and_correct_points(frame):
    global manual_points
    manual_points = []
    window_name = "Select 4 Points for Homography"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, sift_mouse_click)
    while len(manual_points) < 4:
        disp = frame.copy()
        for pt in manual_points:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        cv2.imshow(window_name, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    corrected_points = []
    for pt in manual_points:
        cp = find_nearest_edge_point(pt, edges)
        corrected_points.append(cp)
        print(f"Original: {pt}, Corrected: {cp}")
    return corrected_points

# ----------------------- 光流区域跟踪模块 -----------------------
def flow_to_color(flow, multiplier=50):
    """
    将光流（H, W, 2）转换为颜色编码的BGR图像用于可视化，
    使用HSV映射：角度对应色调，幅值对应亮度。
    multiplier 参数用于调节幅值到亮度的映射倍率
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # 计算幅值和角度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print("Flow magnitude: min =", mag.min(), ", max =", mag.max())  # 调试输出
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * multiplier, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# 使用 RAFT 光流直接对 4 个角点进行帧间变换跟踪（实验方法三）
# 该方法直接对 4 个角点进行光流跟踪，避免了多边形掩码的复杂性
def track_region_dense(video_file, region_points, resize_dim, visualize=False):
    """
    利用 RAFT 计算密集光流，在每帧中：
      1. 根据用户标记的四个点构成区域，生成多边形掩码；
      2. 从光流场中提取该区域内所有像素的流向，采用中位数计算整体位移；
      3. 如果整体位移幅值小于 motion_threshold，则认为区域静止，不更新区域位置；
      4. 否则，将该位移更新到区域的四个角点；
      5. 保存每帧的光流颜色图，便于调试。
    """
    motion_threshold = 0.1  # 如果中位数位移低于该阈值，则认为区域静止
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    RAFT = RAFT.eval()

    cap = cv2.VideoCapture(video_file)
    tracked_regions = []  # 每帧的区域四个角点列表
    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return tracked_regions
    frame1 = cv2.resize(frame1, resize_dim)
    prev_frame = frame1.copy()

    # 初始化区域点（4个点，格式：[(x,y), ...]）
    region_pts = np.array(region_points, dtype=np.float32)  # shape (4, 2)
    tracked_regions.append(region_pts.tolist())

    # 创建调试目录保存光流图
    debug_dir = os.path.join(args.output_dir, "flow_debug")
    os.makedirs(debug_dir, exist_ok=True)

    frame_idx = 0
    if visualize:
        window_name = "Region Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resize_dim[0], resize_dim[1])

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, resize_dim)
        
        # 转换为 tensor 并归一化
        prev_tensor = torch.tensor(prev_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        next_tensor = torch.tensor(frame2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        prev_tensor = prev_tensor.to(device)
        next_tensor = next_tensor.to(device)
        
        with torch.no_grad():
            flows = RAFT(prev_tensor, next_tensor)
            flow = flows[-1]
        flow = flow.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)

        # 保存当前帧光流颜色图
        flow_color = flow_to_color(flow, multiplier=50)
        # cv2.imwrite(os.path.join(debug_dir, f"flow_frame_{frame_idx:04d}.png"), flow_color)

        # 根据当前区域点生成多边形掩码
        mask = np.zeros((resize_dim[1], resize_dim[0]), dtype=np.uint8)
        pts = region_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        # 提取区域内的光流向量
        region_flow = flow[mask == 255]  # shape (N, 2)
        if region_flow.size == 0:
            displacement = np.array([0, 0], dtype=np.float32)
        else:
            displacement = np.median(region_flow, axis=0)
            
        # 如果整体位移小于阈值，则认为区域静止
        
        if np.linalg.norm(displacement) < motion_threshold:
            print(f"Frame {frame_idx}: displacement {displacement} below threshold, no update")
            displacement = np.array([0, 0], dtype=np.float32)
        else:
            print(f"Frame {frame_idx}: displacement = {displacement}")
        # 更新区域点
        # region_pts = region_pts + displacement

        # 每个角点局部平均光流更新
        new_region_pts = []
        for pt in region_pts:
            x, y = int(pt[0]), int(pt[1])
            x0, x1 = max(0, x - 2), min(resize_dim[0], x + 3)
            y0, y1 = max(0, y - 2), min(resize_dim[1], y + 3)
            local_flow = flow[y0:y1, x0:x1].reshape(-1, 2)
            avg_flow = np.mean(local_flow, axis=0)
            new_pt = pt + avg_flow
            new_region_pts.append(new_pt)
        region_pts = np.array(new_region_pts, dtype=np.float32)
        
        tracked_regions.append(region_pts.tolist())

        prev_frame = frame2.copy()
        frame_idx += 1
        
        if visualize:
            vis_frame = frame2.copy()
            pts_draw = region_pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_frame, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    return tracked_regions

def project_points_with_colmap(K, R0, t0, R_list, t_list, src_points, depth=1.0):
    """
    使用 COLMAP 相机位姿将源图像中的点投影到每一帧。
    src_points: (N, 2) — 初始帧上的角点（如插入区域四个角）
    R0, t0: 初始帧的相机位姿
    R_list, t_list: 所有帧的相机位姿
    """
    src_points = np.array(src_points, dtype=np.float32)
    projected_traj = []

    for Rt, tt in zip(R_list, t_list):
        projected = []
        for pt in src_points:
            pt_h = np.array([pt[0], pt[1], 1.0])  # 像素齐次坐标
            X = np.linalg.inv(K) @ pt_h           # 归一化坐标 → 相机坐标（深度为1）
            X = R0 @ (X * depth) + t0             # 初始帧 → 世界坐标
            X_cam = Rt.T @ (X - tt)               # 世界坐标 → 当前帧相机坐标
            x_proj = K @ X_cam
            x_proj = x_proj[:2] / x_proj[2]
            projected.append(x_proj)
        projected_traj.append(projected)

    return projected_traj

import numpy as np

def load_intrinsics_from_txt(path, camera_id=1):
    """
    从 COLMAP 的相机参数文件中加载指定 camera_id 的内参矩阵 K。
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    fx, cx, cy = None, None, None
    found = False
    for i, line in enumerate(lines):
        if line.strip() == f"# Camera ID: {camera_id}":
            found = True
            for j in range(i+1, len(lines)):
                if lines[j].startswith("Params:"):
                    param_line = lines[j].split("Params:")[1].strip().strip('[]')
                    params = [float(p) for p in param_line.split(',')]
                    fx, cx, cy = params[0], params[1], params[2]
                    break
            break

    if not found or fx is None:
        raise ValueError(f"Camera ID {camera_id} not found or Params line missing in K.txt")

    K = np.array([
        [fx, 0,  cx],
        [0,  fx, cy],
        [0,  0,  1]
    ])
    return K



def load_extrinsics_from_txt(r_path, t_path):
    """
    从 R.txt 和 t.txt 中解析每帧的旋转矩阵 R 和平移向量 t
    文件格式应如下：
    # Image: frame_00000.png
    0.99 0.01 ...
    0.00 1.00 ...
    0.00 0.00 ...

    返回：
    extrinsics = {
        'frame_00000.png': (R, t)
    }
    """
    def parse_matrix_file(path):
        matrices = {}
        current_key = None
        current_lines = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("# Image:"):
                    if current_key and current_lines:
                        matrices[current_key] = np.array(current_lines, dtype=np.float32)
                    current_key = line.split(":")[1].strip()
                    current_lines = []
                else:
                    current_lines.append([float(x) for x in line.strip().split()])
            # 最后一帧
            if current_key and current_lines:
                matrices[current_key] = np.array(current_lines, dtype=np.float32)
        return matrices

    R_dict = parse_matrix_file(r_path)
    t_dict = parse_matrix_file(t_path)

    # 验证帧对齐
    if set(R_dict.keys()) != set(t_dict.keys()):
        raise ValueError("Frame names in R.txt and t.txt do not match!")

    extrinsics = {}
    for fname in sorted(R_dict.keys()):
        R = R_dict[fname]
        t = t_dict[fname].reshape(3, 1)
        if R.shape != (3, 3) or t.shape != (3, 1):
            raise ValueError(f"Invalid shape for R or t in frame {fname}")
        extrinsics[fname] = (R, t)

    return extrinsics



def estimate_camera_pose(flow, K):
    """
    通过光流计算相机的相对位姿（R, t）
    flow: (H, W, 2) 光流场
    K: 相机内参矩阵 (3x3)

    返回:
    R: 旋转矩阵
    t: 平移向量
    """
    h, w = flow.shape[:2]
    
    # 生成像素坐标网格
    y, x = np.mgrid[0:h, 0:w]
    points1 = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    points2 = points1 + flow.reshape(-1, 2)  # 加上光流位移，得到对应点

    # 计算本质矩阵
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0)

    if E is None:
        print("Essential matrix estimation failed.")
        return np.eye(3), np.zeros((3, 1))

    # 从本质矩阵恢复 R 和 t
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)

    return R, t

# ----------------------- 轨迹平滑与可视化 -----------------------
def smooth_trajectories(tracked_points, window_length=15, polyorder=2):
    num_frames = len(tracked_points)
    num_points = len(tracked_points[0])
    traj = np.array(tracked_points)
    smoothed = np.zeros_like(traj)
    for i in range(num_points):
        x_coords = traj[:, i, 0]
        y_coords = traj[:, i, 1]
        win_len = window_length if window_length <= len(x_coords) and window_length % 2 == 1 else (len(x_coords) // 2) * 2 + 1
        smooth_x = savgol_filter(x_coords, window_length=win_len, polyorder=polyorder)
        smooth_y = savgol_filter(y_coords, window_length=win_len, polyorder=polyorder)
        smoothed[:, i, 0] = smooth_x
        smoothed[:, i, 1] = smooth_y
    return smoothed

def smooth_trajectories_with_kalman_and_savgol(tracked_points, method='savgol', window_length=21, polyorder=2):
    num_frames = len(tracked_points)
    num_points = len(tracked_points[0])
    traj = np.array(tracked_points)
    smoothed = np.zeros_like(traj)

    if method == 'savgol':
        for i in range(num_points):
            x_coords = traj[:, i, 0]
            y_coords = traj[:, i, 1]
            win_len = window_length if window_length <= len(x_coords) and window_length % 2 == 1 else (len(x_coords) // 2) * 2 + 1
            smooth_x = savgol_filter(x_coords, window_length=win_len, polyorder=polyorder)
            smooth_y = savgol_filter(y_coords, window_length=win_len, polyorder=polyorder)
            smoothed[:, i, 0] = smooth_x
            smoothed[:, i, 1] = smooth_y

    elif method == 'kalman':
        for i in range(num_points):
            # 初始化状态变量
            x, y = traj[0, i, 0], traj[0, i, 1]
            vx, vy = 0, 0
            state = np.array([x, y, vx, vy])

            # 状态转移矩阵 A
            A = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # 观测矩阵 H
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

            # 噪声协方差
            Q = np.eye(4) * 0.01
            R = np.eye(2) * 1.0
            P = np.eye(4)

            result = []
            for t in range(num_frames):
                # 预测
                state = A @ state
                P = A @ P @ A.T + Q

                # 更新
                z = traj[t, i, :]
                y_k = z - H @ state
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                state = state + K @ y_k
                P = (np.eye(4) - K @ H) @ P

                result.append(state[:2])

            smoothed[:, i, 0] = [r[0] for r in result]
            smoothed[:, i, 1] = [r[1] for r in result]

    else:
        raise ValueError("Unsupported smoothing method: choose 'savgol' or 'kalman'")

    return smoothed


def plot_trajectories(original_traj, smoothed_traj):
    num_frames = len(original_traj)
    num_points = len(original_traj[0])
    orig_array = np.array(original_traj)
    plt.figure(figsize=(15, 15))
    colors = ['r', 'g', 'b', 'c']
    for i in range(num_points):
        orig_points = orig_array[:, i, :]
        smooth_points = smoothed_traj[:, i, :]
        plt.plot(smooth_points[:, 0], smooth_points[:, 1], 's--', color=colors[i], alpha=0.5,
                 label=f"Smoothed Point {i + 1}")
        plt.plot(orig_points[:, 0], orig_points[:, 1], 'o-', color='gray',
                 label=f"Original Point {i + 1}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Original and Smoothed Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_homography_for_frames(src_points, smoothed_traj):
    homography_results = []
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    num_frames = smoothed_traj.shape[0]
    for idx in range(num_frames):
        dst = smoothed_traj[idx, :, :].reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": idx, "homography_matrix": H.tolist()})
    return homography_results

def compute_homography_with_residual_regularization(src_points, tracked_regions, smooth_method='savgol', window_length=21, polyorder=2):
    
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)  # (4,1,2)
    num_frames = len(tracked_regions)
    num_points = src.shape[0]

    # Step 1: 初始拟合 homographies 和 residuals
    raw_homographies = []
    raw_residuals = np.zeros((num_frames, num_points, 2), dtype=np.float32)

    for t in range(num_frames):
        dst = np.array(tracked_regions[t], dtype=np.float32).reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        raw_homographies.append(H)

        pred = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        residual = dst.reshape(-1, 2) - pred
        raw_residuals[t] = residual

    # Step 2: 平滑 residuals（默认 Savitzky-Golay）
    smoothed_residuals = np.zeros_like(raw_residuals)
    for i in range(num_points):
        x_coords = raw_residuals[:, i, 0]
        y_coords = raw_residuals[:, i, 1]
        win_len = window_length if window_length <= len(x_coords) and window_length % 2 == 1 else (len(x_coords) // 2) * 2 + 1

        if smooth_method == 'savgol':
            smooth_x = savgol_filter(x_coords, window_length=win_len, polyorder=polyorder)
            smooth_y = savgol_filter(y_coords, window_length=win_len, polyorder=polyorder)
        elif smooth_method == 'kalman':
            # 可扩展 kalman，这里暂时只实现 savgol
            smooth_x, smooth_y = x_coords, y_coords
        else:
            raise ValueError("Unsupported smooth method")

        smoothed_residuals[:, i, 0] = smooth_x
        smoothed_residuals[:, i, 1] = smooth_y

    # Step 3: 重构平滑后的目标角点序列
    final_traj = []
    for t in range(num_frames):
        pred = cv2.perspectiveTransform(src, raw_homographies[t]).reshape(-1, 2)
        refined_pts = pred + smoothed_residuals[t]
        final_traj.append(refined_pts.tolist())

    # Step 4: 用 refined 轨迹重新拟合 homography
    final_homographies = []
    for t in range(num_frames):
        dst = np.array(final_traj[t], dtype=np.float32).reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        final_homographies.append({"frame_idx": t, "homography_matrix": H.tolist()})

    return final_homographies, final_traj  # 可以用于后续可视化 or 插入

# ------------------ Evaluate Homography --------------

def evaluate_homographies(homographies, src_pts, gt_traj):

    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    gt_traj = np.array(gt_traj, dtype=np.float32)

    errors = []
    max_errors = []

    for i, H in enumerate(homographies):
        H = np.array(H, dtype=np.float32)
        pred_pts = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
        gt_pts = gt_traj[i]

        frame_errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
        errors.append(np.mean(frame_errors))
        max_errors.append(np.max(frame_errors))

    return errors, max_errors

def plot_homography_errors(errors, max_errors=None):
    plt.figure(figsize=(12, 5))
    plt.plot(errors, label='Average Corner Error (px)', linewidth=2)
    if max_errors is not None:
        plt.plot(max_errors, label='Max Corner Error (px)', linestyle='--', alpha=0.7)
    plt.title('Homography Projection Error per Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Pixel Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------- 主程序 -----------------------
def main():
    print("Starting interactive insertion...")
    fused_img, source_img, mask_img, ins_params, orig_frame = interactive_insertion(
        args.video_file, args.object_file, args.resize_width, args.output_dir)
    cv2.imshow("Final Insertion", fused_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("Now select 4 points for global Homography computation on the original frame...")
    corrected_src_points = select_and_correct_points(orig_frame)
    print("Corrected source points:", corrected_src_points)

    # 使用用户选取的区域作为初始区域
    resize_dim = (orig_frame.shape[1], orig_frame.shape[0])
    

    # 加载内参和位姿
    K = load_intrinsics_from_txt("colmap/K.txt", camera_id=1)
    extrinsics = load_extrinsics_from_txt("colmap/R.txt", "colmap/t.txt")
    frame_files = sorted(list(extrinsics.keys()))  # 如 frame_00000.jpg, frame_00001.jpg
    print("Loaded frames from extrinsics:", frame_files)
    R0, t0 = extrinsics[frame_files[0]]
    
    R_list = []
    t_list = []
    for frame_name in frame_files:
        R_dst, t_dst = extrinsics[frame_name]
        R_list.append(R_dst)
        t_list.append(t_dst)
        
    # 执行轨迹追踪
    colmap_tracked = project_points_with_colmap(K, R0, t0, R_list, t_list, corrected_src_points)

    USE_COLMAP_POSE = True  # 是否使用 COLMAP 进行位姿估计

    if USE_COLMAP_POSE:
        tracked_regions = colmap_tracked
    else:
        tracked_regions = track_region_dense(args.video_file, corrected_src_points, resize_dim, visualize=True)

    print(f"Tracked regions obtained on {len(tracked_regions)} frames.")
    
    cap = cv2.VideoCapture(args.video_file)
    idx = 0
    while cap.isOpened() and idx < len(tracked_regions):
        ret, frame = cap.read()
        if not ret:
            break
        pts = np.array(tracked_regions[idx], dtype=np.int32).reshape((-1, 1, 2))
        frame_vis = frame.copy()
        cv2.polylines(frame_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Tracked Polygon", frame_vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        idx += 1
    cap.release()
    cv2.destroyAllWindows()
    
    """
    smoothed_traj = smooth_trajectories_with_kalman_and_savgol(tracked_regions, method="kalman",window_length=21, polyorder=2)
    plot_trajectories(tracked_regions, smoothed_traj)

    #homography_results = compute_homography_for_frames(corrected_src_points, smoothed_traj)
    homography_results, refined_traj = compute_homography_with_residual_regularization(
        corrected_src_points, tracked_regions, smooth_method='savgol', window_length=21, polyorder=2)
    homography_json = os.path.join(args.output_dir, "global_homography_results_sample_video_02_7s_colmap.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)
    """

if __name__ == "__main__":
    main()
