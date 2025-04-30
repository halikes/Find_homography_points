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


def flow_consistency_check(flow_fw, flow_bw, threshold=1.0):
    """
    计算 forward-backward consistency mask
    flow_fw: flow_{t→t+1}
    flow_bw: flow_{t+1→t}
    """
    h, w = flow_fw.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    coords = np.stack([x, y], axis=-1).astype(np.float32)
    coords_fw = coords + flow_fw
    coords_fw_clipped = np.clip(coords_fw, 0, [w-1, h-1])
    
    # sample flow_bw 在 coords_fw 上
    map_x = coords_fw_clipped[..., 0].astype(np.float32)
    map_y = coords_fw_clipped[..., 1].astype(np.float32)
    flow_bw_sampled = cv2.remap(flow_bw, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    cycle_error = np.linalg.norm(flow_bw_sampled + flow_fw, axis=2)
    mask = (cycle_error < threshold).astype(np.uint8)
    return mask

def track_region_dense_consistent(video_file, region_points, resize_dim, visualize=False):
    motion_threshold = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    cap = cv2.VideoCapture(video_file)
    tracked_regions = []

    ret, frame1 = cap.read()
    if not ret:
        return []
    frame1 = cv2.resize(frame1, resize_dim)
    prev_frame = frame1.copy()
    region_pts = np.array(region_points, dtype=np.float32)
    tracked_regions.append(region_pts.tolist())

    frame_idx = 0
    last_valid_pts = region_pts.copy()

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, resize_dim)

        prev_tensor = torch.tensor(prev_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        next_tensor = torch.tensor(frame2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            flow_list = RAFT(prev_tensor.to(device), next_tensor.to(device))
            flow_fw = flow_list[-1][0].detach().cpu().numpy().transpose(1, 2, 0)

            flow_list_bw = RAFT(next_tensor.to(device), prev_tensor.to(device))
            flow_bw = flow_list_bw[-1][0].detach().cpu().numpy().transpose(1, 2, 0)

        # 一致性检查（optional）
        consistency_mask = flow_consistency_check(flow_fw, flow_bw)
        if np.mean(consistency_mask) < 0.6:
            print(f"Frame {frame_idx}: low consistency, skipping update")
            tracked_regions.append(last_valid_pts.tolist())
            prev_frame = frame2.copy()
            frame_idx += 1
            continue

        # 区域掩码提取光流
        mask = np.zeros((resize_dim[1], resize_dim[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [region_pts.astype(np.int32).reshape((-1, 1, 2))], 255)
        ys, xs = np.where(mask == 255)
        pts1 = np.stack([xs, ys], axis=-1).astype(np.float32)
        flows = flow_fw[ys, xs]
        pts2 = pts1 + flows

        H, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC)
        if H is not None:
            region_pts = cv2.transform(region_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            last_valid_pts = region_pts.copy()
        else:
            print(f"Frame {frame_idx}: affine failed")
            region_pts = last_valid_pts.copy()

        tracked_regions.append(region_pts.tolist())

        if visualize:
            vis = frame2.copy()
            cv2.polylines(vis, [region_pts.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            cv2.imshow("Track", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prev_frame = frame2.copy()
        frame_idx += 1

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

def regularize_homographies(homographies, alpha=0.5):
    """
    对 homography 进行时间一致性正则化，避免 jitter。
    alpha 越大，越保留上一帧特征，越平稳。
    """
    smoothed = [np.array(homographies[0]["homography_matrix"], dtype=np.float32)]
    for i in range(1, len(homographies)):
        H_prev = smoothed[-1]
        H_curr = np.array(homographies[i]["homography_matrix"], dtype=np.float32)
        H_smooth = alpha * H_prev + (1 - alpha) * H_curr
        smoothed.append(H_smooth)

    result = []
    for i, H in enumerate(smoothed):
        result.append({
            "frame_idx": i,
            "homography_matrix": H.tolist()
        })
    return result

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

    tracked_regions = track_region_dense_consistent(
        args.video_file, corrected_src_points, resize_dim, visualize=True
    )

    smoothed_traj = smooth_trajectories_with_kalman_and_savgol(tracked_regions, method="kalman",window_length=21, polyorder=2)
    plot_trajectories(tracked_regions, smoothed_traj)

    #homography_results = compute_homography_for_frames(corrected_src_points, smoothed_traj)
    homography_results, refined_traj = compute_homography_with_residual_regularization(
        corrected_src_points, tracked_regions, smooth_method='savgol', window_length=21, polyorder=2)
    
    #print("Regularizing homography sequence for temporal smoothness...")
    #homography_results = regularize_homographies(homography_results, alpha=0.3)

    homography_json = os.path.join(args.output_dir, "flow_consist_regularize.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)


if __name__ == "__main__":
    main()
