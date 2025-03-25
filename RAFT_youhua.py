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
parser.add_argument('--video_file', type=str, default='data/sample_video_005.mp4', help='Path to target video')
parser.add_argument('--object_file', type=str, default='data/source_flower.png', help='Path to object image')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the frame')
parser.add_argument('--max_corners', type=int, default=100, help='Max number of candidate points')
parser.add_argument('--min_inliers', type=int, default=4, help='Minimum number of inliers to continue tracking')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 交互式对象插入模块 -----------------------
def nothing(x):
    pass

def insertion_mouse_callback(event, x, y, flags, param):
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
    cap = cv2.VideoCapture(video_file)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read video file: " + video_file)
    cap.release()

    h0, w0 = first_frame.shape[:2]
    scale = resize_width / float(w0)
    first_frame = cv2.resize(first_frame, (resize_width, int(h0 * scale)))
    target_frame = first_frame.copy()
    orig_frame = first_frame.copy()

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

        display_img[y:y + new_h, x:x + new_w] = cur_obj_img
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

    fused_img = display_img.copy()
    source_img = np.zeros_like(fused_img)
    source_img[y:y + new_h, x:x + new_w] = cur_obj_img
    roi = source_img[y:y + new_h, x:x + new_w]
    source_img[y:y + new_h, x:x + new_w] = roi  # 确保区域尺寸一致
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255

    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }

    cv2.imwrite(os.path.join(output_dir, "source_img_sample_video_05.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "mask_img_sample_video_05.png"), mask_img)

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


# ----------------------- 纹理增强模块 -----------------------
def enhance_texture(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ----------------------- 自动候选点提取 -----------------------
def extract_candidate_points(frame, max_corners=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.squeeze(corners)
    else:
        corners = np.empty((0,2), dtype=np.float32)
    return corners.astype(np.float32)

# ----------------------- 光流区域跟踪模块 -----------------------
def sample_flow_local_average(flow, x, y, window_size=5, sigma=1.0):
    half = window_size // 2
    h, w, _ = flow.shape
    flows = []
    weights = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            xi = int(np.clip(x + dx, 0, w - 1))
            yi = int(np.clip(y + dy, 0, h - 1))
            flows.append(flow[yi, xi])
            weight = np.exp(-((dx**2 + dy**2) / (2 * sigma**2)))
            weights.append(weight)
    flows = np.array(flows)
    weights = np.array(weights)
    return np.sum(flows * weights[:, None], axis=0) / np.sum(weights)

def track_candidates_with_raft(video_file, resize_dim, max_corners=100, visualize=False, beta=0.8):
    """
    自动提取候选角点，然后利用 RAFT 计算密集光流，对候选点逐帧跟踪，
    并利用RANSAC剔除跟踪失败的点，计算初始候选点与当前候选点之间的全局Homography。
    对每一帧的Homography矩阵都保存到列表中。
    """
    max_disp = 10.0  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    RAFT = RAFT.eval()
    
    cap = cv2.VideoCapture(video_file)
    tracked_points = []
    homography_results = []
    
    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return None, None
    frame1 = cv2.resize(frame1, resize_dim)
    frame1_enhanced = enhance_texture(frame1)
    prev_frame = frame1_enhanced.copy()
    
    # 自动候选点提取
    init_points = extract_candidate_points(frame1_enhanced, max_corners=max_corners)
    if init_points.shape[0] < 4:
        print("Not enough candidate points detected.")
        cap.release()
        return None, None
    points = init_points.copy()  # (N,2)
    tracked_points.append(points.tolist())
    
    # 保存初始候选点，用于Homography计算
    initial_points = points.copy()
    
    # 初始化累积位移
    accumulated_disp = np.zeros_like(points)
    
    # 帧0的Homography设为单位矩阵
    homography_results.append({"frame_idx": 0, "homography_matrix": np.eye(3, dtype=np.float32).tolist()})
    
    frame_idx = 1
    if visualize:
        window_name = "Candidate Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resize_dim[0], resize_dim[1])
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, resize_dim)
        frame2_enhanced = enhance_texture(frame2)
        
        prev_tensor = torch.tensor(prev_frame, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
        next_tensor = torch.tensor(frame2_enhanced, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
        prev_tensor = prev_tensor.to(device)
        next_tensor = next_tensor.to(device)
        
        with torch.no_grad():
            flows = RAFT(prev_tensor, next_tensor)
            flow = flows[-1]
        flow = flow.squeeze(0).detach().cpu().numpy().transpose(1,2,0)  # (H,W,2)
        
        # 对每个候选点进行局部加权采样获得位移
        raft_disp = []
        for pt in points:
            x, y = pt
            disp = sample_flow_local_average(flow, x, y, window_size=5, sigma=1.0)
            norm_disp = np.linalg.norm(disp)
            if norm_disp > max_disp:
                disp = disp / norm_disp * max_disp
            raft_disp.append(disp)
        raft_disp = np.array(raft_disp)
        
        # 多帧累积：指数平滑更新
        accumulated_disp = beta * accumulated_disp + (1 - beta) * raft_disp
        new_points = points + accumulated_disp
        
        # 计算初始候选点与当前候选点之间的Homography
        H, mask = cv2.findHomography(initial_points, new_points, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
            mask = np.ones((new_points.shape[0], 1), dtype=np.uint8)
        homography_results.append({"frame_idx": frame_idx, "homography_matrix": H.tolist()})
        
        # 剔除跟踪失败的点：仅保留RANSAC内点
        mask = mask.ravel().astype(bool)
        inlier_count = np.sum(mask)
        if inlier_count < args.min_inliers:
            print(f"Frame {frame_idx}: Not enough inliers ({inlier_count}), stopping tracking.")
            break
        new_points = new_points[mask]
        initial_points = initial_points[mask]
        accumulated_disp = accumulated_disp[mask]
        
        points = new_points.copy()
        tracked_points.append(points.tolist())
        
        prev_frame = frame2_enhanced.copy()
        frame_idx += 1
        
        if visualize:
            vis_frame = frame2.copy()
            for pt in points:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    # 保存Homography结果到JSON
    homography_json = os.path.join(args.output_dir, "global_homography_results.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)
    
    return tracked_points, homography_results

def flow_to_color(flow, multiplier=50):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    print("Compensated Flow magnitude: min =", mag.min(), ", max =", mag.max())
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = np.clip(mag * multiplier, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

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
    orig_array = np.array(original_traj, dtype=object)
    plt.figure(figsize=(15, 15))
    colors = ['r', 'g', 'b', 'c']
    for i in range(num_points):
        pts = []
        for frame in original_traj:
            if i < len(frame):
                pts.append(frame[i])
        pts = np.array(pts)
        plt.plot(pts[:,0], pts[:,1], 'o-', color=colors[i], label=f"Point {i+1}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Tracked Candidate Trajectories")
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

    # 使用用户选取的区域作为初始区域，进行全局运动补偿跟踪
    resize_dim = (orig_frame.shape[1], orig_frame.shape[0])
    tracked_regions, homography_results = track_candidates_with_raft(args.video_file, resize_dim, max_corners=args.max_corners, visualize=True)
    print(f"Tracked regions obtained on {len(tracked_regions)} frames.")

    smoothed_traj = smooth_trajectories(tracked_regions, window_length=15, polyorder=2)
    plot_trajectories(tracked_regions, smoothed_traj)

    homography_final = compute_homography_for_frames(corrected_src_points, smoothed_traj)
    homography_json = os.path.join(args.output_dir, "global_homography_results_sample_video_05_raft-homo.json")
    with open(homography_json, "w") as f:
        json.dump(homography_final, f, indent=4)
    print("Homography results saved to", homography_json)

if __name__ == "__main__":
    main()
