# Canny边缘检测 来匹配最选点 + SIFT进行获取所有homography matix

import cv2
import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
# ------------- 参数设置 -------------
# 滑动窗口长度和多项式阶数，用于 Savitzky-Golay 滤波（根据视频帧率和噪声情况调整）
WINDOW_LENGTH = 7  # 必须为奇数
POLY_ORDER = 2

# 光流参数
lk_params = dict(winSize  = (21, 21),
                 maxLevel = 5,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# ------------- 手动标记初始点 -------------
manual_points = []  # 存放鼠标点击的原始坐标

def mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to start processing.")

# 用于在边缘图像中寻找距离给定点最近的边缘像素
def find_nearest_edge_point(point, edge_image):
    # point 为 (x, y)
    # np.argwhere 返回满足条件的 (y, x) 坐标
    indices = np.argwhere(edge_image > 0)
    if len(indices) == 0:
        return point  # 没有边缘则返回原点
    # 分别计算 x 和 y 的距离（注意 indices 的顺序为 [y, x]）
    distances = np.sqrt((indices[:, 1] - point[0])**2 + (indices[:, 0] - point[1])**2)
    idx = np.argmin(distances)
    best_y, best_x = indices[idx]
    return (int(best_x), int(best_y))

# ------------- 打开视频并读取第一帧 -------------
video_path = 'data/test02_4s.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

first_frame = cv2.resize(first_frame, (1024, 1024))

# 显示第一帧，等待手动点击选取点
cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_click)

while len(manual_points) < 4:
    temp = first_frame.copy()
    for pt in manual_points:
        cv2.circle(temp, pt, 3, (0, 0, 255), -1)
    cv2.imshow("Select 4 Points", temp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
cv2.destroyAllWindows()

# ------------- 边缘检测并修正手动选点 -------------
# 转换为灰度图后进行 Canny 边缘检测
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# 参数可根据实际情况调整
edges = cv2.Canny(gray_first, threshold1=50, threshold2=150)

# 对每个手动点击的点，寻找最近的边缘点
corrected_points = []
for pt in manual_points:
    corrected = find_nearest_edge_point(pt, edges)
    corrected_points.append(corrected)
    print(f"Original point: {pt}, corrected to edge: {corrected}")

# 可以显示边缘图和修正后的点进行对比验证
temp_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for pt in corrected_points:
    cv2.circle(temp_vis, pt, 5, (0, 255, 0), -1)
cv2.imshow("Edge Detection and Corrected Points", temp_vis)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# 转换为 np.float32 格式，形状为 (num_points, 1, 2)
initial_points = np.array(corrected_points, dtype=np.float32).reshape(-1, 1, 2)

# ------------- 初始化点轨迹存储 -------------
# 每个元素存储某个点在每一帧的坐标
point_trajectories = [[pt[0].tolist()] for pt in initial_points]

# 用于保存每一帧的追踪点数据，便于后续分析和保存
all_frame_points = [{"frame_idx": 0, "dst_points": [list(pt[0]) for pt in initial_points]}]

# ------------- 结合光流跟踪 -------------
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_points = initial_points.copy()

frame_idx = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 1024))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 利用光流跟踪上一次的点位置
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)

    # 反向光流验证，进一步过滤不稳定的跟踪点
    back_points, status_back, err_back = cv2.calcOpticalFlowPyrLK(gray_frame, prev_gray, next_points, None, **lk_params)
    d = abs(prev_points - back_points).reshape(-1, 2).max(-1)
    good = d < 1.0  # 允许误差，可以根据情况调整

    # 更新点位：对每个点，如果跟踪成功则更新，否则保持上一次位置
    valid_points = []
    for i, (new_pt, good_flag) in enumerate(zip(next_points, good)):
        if good_flag:
            pt = new_pt
        else:
            pt = prev_points[i]  # 或者可以使用其他插值手段
        valid_points.append(pt)
        # 保存轨迹（可结合 cv2.cornerSubPix 进一步精化）
        point_trajectories[i].append(pt[0].tolist())

    valid_points = np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)
    all_frame_points.append({"frame_idx": frame_idx, "dst_points": [list(pt[0]) for pt in valid_points]})

    # 绘制跟踪点
    for pt in valid_points:
        cv2.circle(frame, tuple(int(x) for x in pt[0]), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Tracked Points", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray_frame.copy()
    prev_points = valid_points.copy()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ------------- 后处理平滑 -------------
# 假设每一帧都有4个点，每个 point_trajectories[i] 是一个列表，长度等于帧数
num_frames = len(point_trajectories[0])
smoothed_points_all_frames = []

# 对于每个帧，平滑各个点的轨迹
for frame_idx in range(num_frames):
    smoothed_frame_points = []
    for i in range(len(point_trajectories)):
        traj = np.array(point_trajectories[i])  # shape: (num_frames, 2)
        # 窗口长度要小于轨迹长度且为奇数
        win_len = WINDOW_LENGTH if WINDOW_LENGTH <= len(traj) and WINDOW_LENGTH % 2 == 1 else (len(traj) // 2)*2+1
        smoothed_x = savgol_filter(traj[:, 0], window_length=win_len, polyorder=POLY_ORDER)
        smoothed_y = savgol_filter(traj[:, 1], window_length=win_len, polyorder=POLY_ORDER)
        smoothed_frame_points.append([round(smoothed_x[frame_idx], 2), round(smoothed_y[frame_idx], 2)])
    smoothed_points_all_frames.append({
        "frame_idx": frame_idx,
        "dst_points": smoothed_frame_points
    })

# ------------- 保存结果为 JSON -------------
output_json = "test01.json"
with open(output_json, 'w') as f:
    json.dump({"frames": smoothed_points_all_frames}, f, indent=4)
print(f"Smoothed tracking points saved to {output_json}")



# ------------- 计算每一帧的 Homography 矩阵 -------------
# 以第一帧的修正点作为源点 (src_points)，目标点为平滑后的 dst_points
src_points = np.array(corrected_points, dtype=np.float32)  # shape: (4,2)
homography_results = []

for frame_data in smoothed_points_all_frames:
    frame_idx = frame_data["frame_idx"]
    dst_points = np.array(frame_data["dst_points"], dtype=np.float32)  # shape: (4,2)
    # 计算 Homography 矩阵，使用 0 方法（精确计算）或 RANSAC 方法
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None:
        print(f"Frame {frame_idx}: Homography computation failed.")
        H = np.eye(3, dtype=np.float32)
    homography_results.append({
        "frame_idx": frame_idx,
        "homography_matrix": H.tolist()
    })

# 保存 Homography 结果到 JSON 文件
homography_json = "test01_homography_results.json"
with open(homography_json, 'w') as f:
    json.dump(homography_results, f, indent=4)
print(f"Homography results saved to {homography_json}")


# ------------- 绘制最终点轨迹分析图 -------------
# 加载平滑后的点数据（JSON文件）
with open(output_json, 'r') as f:
    data = json.load(f)

frames_data = data["frames"]

# 假设每帧有4个点，构建每个点的轨迹列表
points_traj = [[] for _ in range(4)]
for frame in frames_data:
    pts = frame["dst_points"]
    for i, pt in enumerate(pts):
        points_traj[i].append(pt)

# 绘制各个点随帧变化的轨迹
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c']
for i, traj in enumerate(points_traj):
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], marker='o', color=colors[i], label=f"Point {i+1}")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Trajectory of Points over Frames")
plt.legend()
plt.grid(True)
plt.show()