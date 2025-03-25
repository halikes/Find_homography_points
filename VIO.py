import cv2
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_005.mp4', help='Path to target video')
parser.add_argument('--output_dir', type=str, default='results/vo', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize frames')
parser.add_argument('--max_corners', type=int, default=200, help='Maximum number of features to track')
parser.add_argument('--min_features', type=int, default=50, help='Minimum number of features; below this, reinitialize features')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 纹理增强模块 -----------------------
def enhance_texture(frame):
    """对输入BGR图像进行CLAHE和简单锐化，增强纹理信息"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ----------------------- 特征提取与跟踪 -----------------------
def extract_features(frame, max_corners):
    """使用Shi-Tomasi角点检测提取特征点"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    if features is not None:
        features = np.squeeze(features)
    else:
        features = np.empty((0, 2), dtype=np.float32)
    return features.astype(np.float32)

def track_features(prev_gray, curr_gray, prev_points, lk_params):
    """利用LK光流跟踪特征点"""
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)
    status = status.reshape(-1)
    # 对于跟踪失败的点，用上一帧位置替代
    curr_points = curr_points.reshape(-1, 2)
    for i in range(len(status)):
        if status[i] == 0:
            curr_points[i] = prev_points[i]
    return curr_points, status

# ----------------------- 视觉里程计主流程 -----------------------
def visual_odometry(video_file, resize_dim, max_corners):
    cap = cv2.VideoCapture(video_file)
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read video file")
    frame0 = cv2.resize(frame0, resize_dim)
    frame0_enh = enhance_texture(frame0)
    prev_gray = cv2.cvtColor(frame0_enh, cv2.COLOR_BGR2GRAY)
    # 提取初始特征点
    prev_pts = extract_features(frame0_enh, max_corners)
    if prev_pts.shape[0] < 4:
        cap.release()
        raise ValueError("Not enough features detected in first frame!")
    
    # 假设相机内参，根据分辨率简单设定
    h, w = resize_dim[1], resize_dim[0]
    # 这里假设焦距为resize_width
    focal = resize_dim[0]
    pp = (w/2, h/2)
    K = np.array([[focal, 0, pp[0]],
                  [0, focal, pp[1]],
                  [0, 0, 1]])
    
    # LK参数
    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    # 存储相机位姿（初始位姿为单位矩阵）
    trajectory = [np.eye(4)]
    homography_results = []
    
    frame_idx = 1
    traj_vis = []
    traj_vis.append((0, 0))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        frame_enh = enhance_texture(frame)
        curr_gray = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2GRAY)
        
        # 跟踪特征点
        curr_pts, status = track_features(prev_gray, curr_gray, prev_pts, lk_params)
        
        # 如果跟踪失败的点太多，则重新提取特征
        if curr_pts.shape[0] < args.min_features:
            curr_pts = extract_features(frame_enh, max_corners)
            print(f"Frame {frame_idx}: Reinitialized features, got {curr_pts.shape[0]}")
        
        # 计算Essential矩阵
        E, mask = cv2.findEssentialMat(curr_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is not None:
            _, R, t, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, K)
        else:
            R = np.eye(3)
            t = np.zeros((3,1))
        
        # 将相对位姿转换为4x4变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t
        
        # 累计全局位姿（后乘）
        T_prev = trajectory[-1]
        T_curr = T_prev @ T
        trajectory.append(T_curr)
        
        # 计算当前帧的Homography（用于平面运动建模），以初始特征点与当前内点为依据
        H, status_H = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": frame_idx, "homography_matrix": H.tolist()})
        
        # 更新可视化轨迹：这里简单取平移部分
        dx, dy = T_curr[0,3], T_curr[1,3]
        traj_vis.append((dx, dy))
        
        # 更新上一帧信息
        prev_gray = curr_gray.copy()
        prev_pts = curr_pts.copy()
        frame_idx += 1
        
        # 可视化跟踪结果（特征点和轨迹）
        vis = frame.copy()
        for pt in curr_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
        for i in range(1, len(traj_vis)):
            cv2.line(vis, (int(traj_vis[i-1][0]+w/2), int(traj_vis[i-1][1]+h/2)),
                         (int(traj_vis[i][0]+w/2), int(traj_vis[i][1]+h/2)), (0,0,255), 2)
        cv2.imshow("VO Tracking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存全局位姿和Homography矩阵
    traj_json = os.path.join(args.output_dir, "camera_trajectory.json")
    with open(traj_json, "w") as f:
        # 只保存平移部分及部分旋转信息作为示例
        traj_list = [T.tolist() for T in trajectory]
        json.dump(traj_list, f, indent=4)
    print("Camera trajectory saved to", traj_json)
    
    homo_json = os.path.join(args.output_dir, "global_homography_results.json")
    with open(homo_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homo_json)
    
    return trajectory, homography_results

def plot_trajectory(trajectory, resize_dim):
    # 简单绘制平移轨迹
    traj = np.array([T[:3,3] for T in trajectory])
    plt.figure(figsize=(8,8))
    plt.plot(traj[:,0], traj[:,1], 'o-')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Camera Trajectory")
    plt.grid(True)
    plt.show()

def main():
    resize_dim = (args.resize_width, args.resize_width)
    trajectory, homography_results = visual_odometry(args.video_file, resize_dim, args.max_corners)
    plot_trajectory(trajectory, resize_dim)

if __name__ == "__main__":
    main()
