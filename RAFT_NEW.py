import cv2
import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_005.mp4', help='Path to target video')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the frame')
parser.add_argument('--max_corners', type=int, default=100, help='Max number of candidate points')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 纹理增强模块 -----------------------
def enhance_texture(frame):
    """
    对输入BGR图像进行纹理增强：
      1. 转换到Lab空间，对L通道采用CLAHE增强；
      2. 合并通道转换回BGR；
      3. 简单锐化处理。
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    # 简单锐化
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ----------------------- 自动候选点提取 -----------------------
def extract_candidate_points(frame, max_corners=100):
    """
    使用Shi-Tomasi角点检测自动提取候选角点
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.squeeze(corners)
    else:
        corners = np.empty((0,2), dtype=np.float32)
    return corners.astype(np.float32)

# ----------------------- 光流估计与候选点跟踪模块 -----------------------
def sample_flow_local_average(flow, x, y, window_size=5, sigma=1.0):
    """
    对光流场在 (x,y) 处，以 window_size 为窗口进行加权平均采样，
    使用高斯权重返回加权平均的光流向量。
    """
    half = window_size // 2
    h, w, _ = flow.shape
    flows = []
    weights = []
    for dy in range(-half, half+1):
        for dx in range(-half, half+1):
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
    自动提取候选角点，然后利用 RAFT 计算密集光流，
    对候选点逐帧跟踪（采用局部加权采样+指数平滑），
    并在每一帧计算初始候选点与当前候选点的全局Homography矩阵，
    返回候选点轨迹和每帧的Homography结果。
    """
    max_disp = 10.0  # 位移剪裁
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
    
    # 自动提取候选角点
    init_points = extract_candidate_points(frame1, max_corners)
    if init_points.shape[0] < 4:
        print("Not enough candidate points detected.")
        cap.release()
        return None, None
    points = init_points.copy()  # (N,2)
    tracked_points.append(points.tolist())
    
    # 保存初始候选点，用于计算全局Homography
    initial_points = points.copy()
    
    # 初始化累积位移为零
    accumulated_disp = np.zeros_like(points)
    
    frame_idx = 0
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
        
        # 对每个候选点采样光流，得到位移
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
        
        # 计算当前候选点与初始候选点之间的全局Homography
        H, status = cv2.findHomography(initial_points, new_points, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": frame_idx, "homography_matrix": H.tolist()})
        
        # 更新候选点
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
    homography_json = os.path.join(args.output_dir, "global_homography_results_sample_video_05_raft.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)
    
    return tracked_points, homography_results

def flow_to_color(flow, multiplier=50):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    print("Flow magnitude: min =", mag.min(), ", max =", mag.max())
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = np.clip(mag * multiplier, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def plot_trajectories(tracked_points):
    traj = np.array(tracked_points)
    plt.figure(figsize=(10,10))
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], 'o-', label=f"Point {i+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Tracked Candidate Trajectories")
    plt.legend()
    plt.show()

def main():
    resize_dim = (512, 512)
    tracked_points, homography_results = track_candidates_with_raft(args.video_file, resize_dim, max_corners=args.max_corners, visualize=True)
    if tracked_points is not None:
        plot_trajectories(tracked_points)

if __name__ == "__main__":
    main()
