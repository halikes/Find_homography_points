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
parser.add_argument('--video_file', type=str, default='data/sample_video_006.mp4', help='Path to target video')
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
    roi = source_img[y:y + new_h, x:x + new_w]
    roi_smoothed = cv2.bilateralFilter(roi, d=5, sigmaColor=75, sigmaSpace=75)
    source_img[y:y + new_h, x:x + new_w] = roi_smoothed  # 注意这里确保区域尺寸一致
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255

    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }

    cv2.imwrite(os.path.join(output_dir, "source_img_sample_video_06.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "mask_img_sample_video_06.png"), mask_img)

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
def track_region_dense(video_file, region_points, resize_dim, visualize=False):
    """
    改进功能：
    1. 跟踪区域内的所有特征点
    2. 使用双向光流验证筛选稳定点
    3. 根据稳定点计算全局单应性
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    
    cap = cv2.VideoCapture(video_file)
    w, h = resize_dim
    all_homographies = []
    
    # 初始化区域
    region_mask = np.zeros((h, w), dtype=np.uint8)
    region_pts = np.array(region_points, dtype=np.int32)
    cv2.fillPoly(region_mask, [region_pts], 255)
    
    # 提取初始特征点
    prev_frame = cv2.resize(cap.read()[1], (w, h))
    prev_points = cv2.goodFeaturesToTrack(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), 
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=5,
        mask=region_mask
    ).reshape(-1, 2)

    with torch.no_grad():
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        while True:
            ret, curr_frame = cap.read()
            if not ret: break
            curr_frame = cv2.resize(curr_frame, (w, h))
            
            # 前向光流计算
            prev_tensor = torch.from_numpy(prev_frame).permute(2,0,1).unsqueeze(0).to(device)/255.0
            curr_tensor = torch.from_numpy(curr_frame).permute(2,0,1).unsqueeze(0).to(device)/255.0
            flow = raft(prev_tensor, curr_tensor)[-1][0].cpu().numpy().transpose(1,2,0)
            
            # 跟踪点传播
            new_points = prev_points + flow[prev_points[:,1].astype(int), prev_points[:,0].astype(int)]
            
            # 反向验证
            back_flow = raft(curr_tensor, prev_tensor)[-1][0].cpu().numpy().transpose(1,2,0)
            back_points = new_points + back_flow[new_points[:,1].astype(int), new_points[:,0].astype(int)]
            
            # 计算双向误差
            error = np.linalg.norm(prev_points - back_points, axis=1)
            valid_mask = error < 2.0  # 可调参数
            
            # 筛选稳定点
            stable_points = new_points[valid_mask]
            
            # 计算全局单应性
            if len(stable_points) >= 4:
                H, _ = cv2.findHomography(prev_points[valid_mask], stable_points, cv2.RANSAC, 5.0)
            else:
                H = np.eye(3)
            
            all_homographies.append(H.tolist())
            
            # 更新跟踪点
            prev_points = stable_points
            prev_frame = curr_frame.copy()
            
            # 补充新特征点
            if len(prev_points) < 100:
                new_points = cv2.goodFeaturesToTrack(
                    cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
                    maxCorners=500,
                    qualityLevel=0.01,
                    minDistance=5,
                    mask=region_mask
                ).reshape(-1, 2)
                prev_points = np.concatenate([prev_points, new_points])
            
            # 边界约束
            prev_points[:,0] = np.clip(prev_points[:,0], 0, w-1)
            prev_points[:,1] = np.clip(prev_points[:,1], 0, h-1)
            
            pbar.update(1)
    
    cap.release()
    return all_homographies

# ----------------------- 改进的单应性计算模块 -----------------------
def compute_homography_for_frames(src_points, homographies):
    """使用改进后的单应性序列直接生成结果"""
    return [{
        "frame_idx": idx,
        "homography_matrix": H
    } for idx, H in enumerate(homographies)]

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

# ----------------------- 辅助函数 -----------------------
def convert_ndarray_to_list(data):
    """Recursively convert NumPy arrays to lists in a dictionary or list."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    else:
        return data
    
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
    
    print("Processing feature tracking...")
    resize_dim = (orig_frame.shape[1], orig_frame.shape[0])
    homographies = track_region_dense(args.video_file, corrected_src_points, resize_dim, visualize=True)
    
    # 保存结果
    homography_results = compute_homography_for_frames(corrected_src_points, homographies)
    homography_json = os.path.join(args.output_dir, "global_homography_results.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    
    print(f"Homography results saved to {homography_json}")

if __name__ == "__main__":
    main()
