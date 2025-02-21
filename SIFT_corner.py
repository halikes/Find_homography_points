import cv2
import numpy as np
import json
from scipy.signal import savgol_filter

# ------------- 参数设置 -------------
# 滑动窗口长度和多项式阶数，用于 Savitzky-Golay 滤波
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

# 用于在 Harris 角点图中寻找距离给定点最近的角点
def find_nearest_corner_point(point, corner_response, threshold=0.01):
    """
    point: 手动点击的点，格式 (x, y)
    corner_response: Harris 角点响应图（float32类型）
    threshold: 阈值，相对于最大响应值的比例，用于确定角点
    """
    # 设定阈值，获取角点位置。注意：corner_response 的 shape 与图像一致，响应值越大代表角点可能性越高
    thresh_val = threshold * corner_response.max()
    # 获取满足响应值大于阈值的像素位置（返回的顺序为 (y,x)）
    indices = np.argwhere(corner_response > thresh_val)
    if len(indices) == 0:
        return point  # 如果没有找到角点，则返回原始点

    # 计算手动点与所有角点之间的欧式距离
    distances = np.sqrt((indices[:, 1] - point[0])**2 + (indices[:, 0] - point[1])**2)
    idx = np.argmin(distances)
    best_y, best_x = indices[idx]
    return (int(best_x), int(best_y))

# ------------- 打开视频并读取第一帧 -------------
video_path = 'data/video_zhuozi5.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

first_frame = cv2.resize(first_frame, (512, 512))

# 显示第一帧，等待手动点击选取点
cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_click)

while len(manual_points) < 4:
    temp = first_frame.copy()
    for pt in manual_points:
        cv2.circle(temp, pt, 5, (0, 0, 255), -1)
    cv2.imshow("Select 4 Points", temp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
cv2.destroyAllWindows()

# ------------- Harris角点检测并修正手动选点 -------------
# 将第一帧转换为灰度图，并转换为float32（Harris角点检测要求）
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
gray_first = np.float32(gray_first)
# 计算 Harris 角点响应图
# 参数：blockSize=2, ksize=3, k=0.04，可根据实际情况调整
harris_response = cv2.cornerHarris(gray_first, blockSize=2, ksize=3, k=0.04)
# 标记角点
harris_response = cv2.dilate(harris_response, None)


# 对每个手动点击的点，寻找最近的 Harris 角点
corrected_points = []
for pt in manual_points:
    corrected = find_nearest_corner_point(pt, harris_response, threshold=0.01)
    corrected_points.append(corrected)
    print(f"Original point: {pt}, corrected to corner: {corrected}")

# 显示 Harris 角点检测结果，并用蓝色标记出修正后的点
# 这里先将响应图归一化并转换为8位灰度，再转换为BGR方便绘制彩色标记
harris_norm = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)
harris_norm = np.uint8(harris_norm)
corner_vis = cv2.cvtColor(harris_norm, cv2.COLOR_GRAY2BGR)
for pt in corrected_points:
    cv2.circle(corner_vis, pt, 5, (255, 0, 0), 2)  # 蓝色标记，BGR: (255,0,0)
cv2.imshow("Harris Corners with Blue Markers", corner_vis)
cv2.waitKey(10000000)
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

    frame = cv2.resize(frame, (512, 512))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 利用光流跟踪上一次的点位置
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)

    # 反向光流验证，进一步过滤不稳定的跟踪点
    back_points, status_back, err_back = cv2.calcOpticalFlowPyrLK(gray_frame, prev_gray, next_points, None, **lk_params)
    d = abs(prev_points - back_points).reshape(-1, 2).max(-1)
    good = d < 1.0  # 允许误差，可根据实际情况调整

    # 更新点位：对每个点，如果跟踪成功则更新，否则保持上一次位置
    valid_points = []
    for i, (new_pt, good_flag) in enumerate(zip(next_points, good)):
        if good_flag:
            pt = new_pt
        else:
            pt = prev_points[i]
        valid_points.append(pt)
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
num_frames = len(point_trajectories[0])
smoothed_points_all_frames = []

for frame_idx in range(num_frames):
    smoothed_frame_points = []
    for i in range(len(point_trajectories)):
        traj = np.array(point_trajectories[i])  # shape: (num_frames, 2)
        win_len = WINDOW_LENGTH if WINDOW_LENGTH <= len(traj) and WINDOW_LENGTH % 2 == 1 else (len(traj) // 2)*2+1
        smoothed_x = savgol_filter(traj[:, 0], window_length=win_len, polyorder=POLY_ORDER)
        smoothed_y = savgol_filter(traj[:, 1], window_length=win_len, polyorder=POLY_ORDER)
        smoothed_frame_points.append([round(smoothed_x[frame_idx], 2), round(smoothed_y[frame_idx], 2)])
    smoothed_points_all_frames.append({
        "frame_idx": frame_idx,
        "dst_points": smoothed_frame_points
    })

# 保存平滑后的点数据到 JSON 文件
output_json = "auto_detected_points_smoothed_corner.json"
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
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None:
        print(f"Frame {frame_idx}: Homography computation failed.")
        H = np.eye(3, dtype=np.float32)
    homography_results.append({
        "frame_idx": frame_idx,
        "homography_matrix": H.tolist()
    })

homography_json = "homography_results_corner.json"
with open(homography_json, 'w') as f:
    json.dump(homography_results, f, indent=4)
print(f"Homography results saved to {homography_json}")
