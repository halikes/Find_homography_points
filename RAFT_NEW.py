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
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the first frame')
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

# 使用 RAFT 光流直接对 4 个角点进行帧间变换跟踪

def track_region_dense(video_file, region_points, resize_dim, visualize=False):
    
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
        # step 1: 获取光流区域 mask
        mask = np.zeros((resize_dim[1], resize_dim[0]), dtype=np.uint8)
        pts = region_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # step 2: 提取掩码内所有点的位置 和 它们的流动向量
        ys, xs = np.where(mask == 255)
        pts1 = np.stack([xs, ys], axis=-1).astype(np.float32)  # shape: (N, 2)
        flows = flow[ys, xs]  # shape: (N, 2)
        pts2 = pts1 + flows

        # step 3: 用这些点对估计仿射 or 单应矩阵
        # 建议优先用仿射，更稳定（如果变形不严重）
        H, inliers = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)

        if H is None:
            print(f"Frame {frame_idx}: affine estimation failed")
            H = np.eye(2, 3, dtype=np.float32)  # fallback

        # step 4: 用这个 H 统一变换四个角点
        region_pts = cv2.transform(region_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        
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
    """
    利用 residual 正则化方法构造稳定的时序 homography：
    1. 每帧根据 src_points 和目标点 tracked_pts 拟合初始 H_t；
    2. 计算每个角点的 residual；
    3. 对 residual 做平滑；
    4. 重构平滑后的目标点序列；
    5. 再次拟合最终 homography。
    """
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
    """
    对多个 homography 进行评估。
    
    homographies: List of H matrices (N, 3x3)
    src_pts: shape (4, 2) 初始四个点坐标
    gt_traj: shape (N, 4, 2) 每一帧的 ground-truth 四边形坐标（来自光流区域追踪）

    Returns:
        - errors: 每帧的平均 MSE
        - max_errors: 每帧的最大角点误差
    """
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


#------追加的函数：平滑轨迹和可视化------
def analyze_H_drift(homographies, src_pts):
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    drift_norms = []
    proj_pts_list = []
    last_proj = cv2.perspectiveTransform(src_pts, np.array(homographies[0]['homography_matrix']))
    proj_pts_list.append(last_proj.squeeze(1))
    for i in range(1, len(homographies)):
        curr_proj = cv2.perspectiveTransform(src_pts, np.array(homographies[i]['homography_matrix']))
        jump = np.linalg.norm(curr_proj - last_proj, axis=2).mean()
        drift_norms.append(jump)
        last_proj = curr_proj
        proj_pts_list.append(curr_proj.squeeze(1))
    return drift_norms, proj_pts_list

def plot_H_drift(drift_norms, threshold=1.0):
    plt.figure(figsize=(10, 4))
    plt.plot(drift_norms, label='Avg corner drift')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Geometric drift (px)')
    plt.title('Frame-to-frame Homography Drift')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def smooth_homography_sequence(h_list, alpha=0.9):
    smoothed = [h_list[0]]
    for i in range(1, len(h_list)):
        H_prev = smoothed[-1]
        H_curr = h_list[i]
        H_smooth = alpha * H_prev + (1 - alpha) * H_curr
        smoothed.append(H_smooth)
    return smoothed

def compute_edge_map(binary_mask):
    edges = cv2.Canny(binary_mask, 100, 200)
    return edges.astype(np.float32) / 255.0  # Normalize

def compute_edge_consistency_loss(ref_edge, warped_edges):
    losses = []
    for i, edge in enumerate(warped_edges):
        # L1 距离
        l1 = np.mean(np.abs(ref_edge - edge))
        losses.append(l1)
    return losses


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
    tracked_regions = track_region_dense(args.video_file, corrected_src_points, resize_dim, visualize=True)
    print(f"Tracked regions obtained on {len(tracked_regions)} frames.")

    smoothed_traj = smooth_trajectories_with_kalman_and_savgol(tracked_regions, method="kalman",window_length=21, polyorder=2)

    plot_trajectories(tracked_regions, smoothed_traj)

    #homography_results = compute_homography_for_frames(corrected_src_points, smoothed_traj)
    homography_results, refined_traj = compute_homography_with_residual_regularization(
        corrected_src_points, tracked_regions, smooth_method='savgol', window_length=21, polyorder=2)
    
    # === Step: 插入边缘一致性损失分析 ===
    print("🔍 Computing edge consistency loss across frames...")

    src_img = source_img
    mask = mask_img[..., 0]  # 提取灰度掩码
    ref_edge = compute_edge_map(mask)

    # 遍历每一帧，将原图 mask 用 homography warp 到目标帧
    warped_edges = []
    for h_entry in homography_results:
        H = np.array(h_entry["homography_matrix"])
        warped = cv2.warpPerspective(ref_edge, H, (mask.shape[1], mask.shape[0]))
        warped_edges.append(warped)

    losses = compute_edge_consistency_loss(ref_edge, warped_edges)
    plt.plot(losses)
    plt.title("Edge Consistency Loss over Frames")
    plt.xlabel("Frame")
    plt.ylabel("L1 Edge Loss")
    plt.grid(True)
    plt.show()

    print(f"📉 Mean Edge Consistency Loss: {np.mean(losses):.4f}")

    homography_json = os.path.join(args.output_dir, "sample_video_002_7s.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)

    
    h_list = [np.array(h["homography_matrix"]) for h in homography_results]
    drift_norms, proj_pts_list = analyze_H_drift(homography_results, corrected_src_points)
    plot_H_drift(drift_norms)
    print(f"Max drift: {np.max(drift_norms):.2f}px, Mean: {np.mean(drift_norms):.2f}px")

    drift_thresh = 1.0  # px
    if np.max(drift_norms) > drift_thresh:
        print("⚠️ Detected high drift. Applying H smoothing filter...")
        h_list_smooth = smooth_homography_sequence(h_list, alpha=0.9)
        homography_results = [
            {"frame_idx": i, "homography_matrix": H.tolist()} for i, H in enumerate(h_list_smooth)
        ]

        # 平滑后的投影点重新计算并可视化
        new_proj_pts_list = []
        src_pts_arr = np.array(corrected_src_points, dtype=np.float32).reshape(-1, 1, 2)
        for H in h_list_smooth:
            new_proj = cv2.perspectiveTransform(src_pts_arr, H)
            new_proj_pts_list.append(new_proj.squeeze(1))

        plt.figure(figsize=(8, 8))
        for pts in proj_pts_list:
            plt.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.3)
        for pts in new_proj_pts_list:
            plt.plot(pts[:, 0], pts[:, 1], color='green', linestyle='--', alpha=0.5)
        plt.title("Corner Trajectories Before (gray) and After (green dashed) Smoothing")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
