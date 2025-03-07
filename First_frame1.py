import cv2
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_002.mp4', help='Path to target video')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the first frame')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- SIFT 关键点检测 -----------------------
def detect_sift_keypoints(frame):
    """使用 SIFT 进行关键点检测，并返回 SIFT 关键点和描述子"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# ----------------------- 选择桌面角点（SIFT + 手动） -----------------------
def select_table_corners_sift(frame):
    """
    使用 SIFT 自动检测桌面角点，并让用户手动调整
    """
    keypoints, _ = detect_sift_keypoints(frame)
    selected_points = sorted(keypoints, key=lambda kp: kp.pt[0])[:4] if len(keypoints) >= 4 else []

    manual_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
            manual_points.append((x, y))
            print(f"手动选择点 {len(manual_points)}: ({x}, {y})")

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    while len(manual_points) < 4:
        disp = frame.copy()
        for pt in manual_points:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Points", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return manual_points if len(manual_points) == 4 else [kp.pt for kp in selected_points]

# ----------------------- 结合稀疏光流（LK）和稠密光流（Farneback） -----------------------
def track_points_optical_flow(video_file, initial_points):
    """
    结合 LK（稀疏光流）和 Farneback（稠密光流）进行跟踪
    """
    cap = cv2.VideoCapture(video_file)
    tracked_points = []

    ret, frame = cap.read()
    if not ret:
        return []

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(initial_points, dtype=np.float32).reshape(-1, 1, 2)
    tracked_points.append([tuple(pt[0]) for pt in prev_points])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 稀疏光流（LK 追踪关键点）
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, prev_points, None)

        # 仅保留成功跟踪的点
        good_points = [next_points[i] if status[i] == 1 else prev_points[i] for i in range(len(status))]

        # 计算稠密光流（Farneback）
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 结合 LK 和 Farneback 结果
        alpha = 0.7  # LK 占 70% 权重
        final_points = []
        for i, pt in enumerate(good_points):
            x, y = int(pt[0][0]), int(pt[0][1])
            dx, dy = flow[y, x] if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0] else (0, 0)
            new_x = alpha * pt[0][0] + (1 - alpha) * (pt[0][0] + dx)
            new_y = alpha * pt[0][1] + (1 - alpha) * (pt[0][1] + dy)
            final_points.append((new_x, new_y))

        prev_points = np.array(final_points, dtype=np.float32).reshape(-1, 1, 2)
        tracked_points.append([tuple(pt[0]) for pt in prev_points])
        gray_prev = gray_curr.copy()

    cap.release()
    return tracked_points

# ----------------------- 计算 Homography 矩阵 -----------------------
def compute_homographies(src_points, tracked_points):
    homography_results = []
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)

    for frame_idx, dst_points in enumerate(tracked_points):
        dst = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": frame_idx, "homography_matrix": H.tolist()})

    return homography_results

# ----------------------- 处理视频并计算 Homography -----------------------
def process_video():
    print("Step 1: 选取桌面角点...")
    cap = cv2.VideoCapture(args.video_file)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("无法读取视频文件")

    first_frame = cv2.resize(first_frame, (args.resize_width, int(first_frame.shape[0] * (args.resize_width / first_frame.shape[1]))))

    # 自动 & 手动选点
    table_corners = select_table_corners_sift(first_frame)

    print("Step 2: 使用光流跟踪角点...")
    tracked_points = track_points_optical_flow(args.video_file, table_corners)

    print("Step 3: 计算 Homography 矩阵...")
    homography_results = compute_homographies(table_corners, tracked_points)

    # 保存 Homography 结果
    homography_json = os.path.join(args.output_dir, "homography_results.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography 结果已保存:", homography_json)

# ----------------------- 主程序 -----------------------
if __name__ == "__main__":
    process_video()
