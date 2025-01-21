"""
import cv2
import numpy as np
from scipy.signal import savgol_filter

# 全局变量
manual_points = []  # 第一帧手动选定的点
auto_points = []  # 每帧自动计算的点
point_trajectories = []  # 每个点的历史轨迹（Savitzky-Golay）
ema_points = []  # 每个点的 EMA 值

# 鼠标点击回调函数
def mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to start processing.")

# 图像增强函数
def enhance_image(frame, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_frame

# 加载视频
video_path = 'data/video_zhuozi5.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 手动标记第一帧的点
ret, base_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# 公司是1024x1024 我的是512x512
base_frame = cv2.resize(base_frame, (512, 512))
base_frame = enhance_image(base_frame)  # 图像增强
cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_click)

while len(manual_points) < 4:
    temp_frame = base_frame.copy()
    for point in manual_points:
        cv2.circle(temp_frame, point, 1, (0, 0, 255), -1)  # 绘制选点
    cv2.imshow("Select 4 Points", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# 转为 NumPy 数组
manual_points = np.array(manual_points, dtype=np.float32).reshape(-1, 1, 2)

# 初始化轨迹和 EMA
for point in manual_points:
    point_trajectories.append([point[0][0], point[0][1]])  # 初始化历史轨迹
    ema_points.append(point[0])  # 初始化 EMA

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()

# 在基准帧中检测特征点
keypoints1, descriptors1 = sift.detectAndCompute(base_frame, None)

# 初始化上一帧
prev_frame = base_frame
prev_points = manual_points.copy()

# EMA 参数
alpha = 0.6  # EMA 平滑因子

# 处理视频帧
frame_index = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧大小为 1024x1024并增强
    frame = cv2.resize(frame, (512, 512))
    frame = enhance_image(frame)  # 图像增强

    # 检测当前帧的特征点
    keypoints2, descriptors2 = sift.detectAndCompute(frame, None)

    # 匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # SIFT 使用欧几里得距离
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 使用比值测试过滤匹配（Lowe's ratio test）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用 RANSAC 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            mapped_points = cv2.perspectiveTransform(manual_points, H)

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame, frame, mapped_points, None,
                winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # 融合点
            fused_points = 0.6 * next_points + 0.4 * mapped_points
            valid_points = fused_points.reshape(-1, 2)

            # 更新轨迹
            for i, point in enumerate(valid_points):
                point_trajectories[i].append(point)  # 保存当前帧点

            # 选择滤波方法：Savitzky-Golay 或 EMA
            smoothed_points = []

            # --- 使用 Savitzky-Golay 滤波器 ---
            #for trajectory in point_trajectories:
            #    smoothed_x = savgol_filter([p[0] for p in trajectory], window_length=5, polyorder=2)
            #    smoothed_y = savgol_filter([p[1] for p in trajectory], window_length=5, polyorder=2)
            #    smoothed_points.append((smoothed_x[-1], smoothed_y[-1]))

            # --- 使用 EMA ---
            for i, point in enumerate(valid_points):
                ema_points[i] = alpha * point + (1 - alpha) * ema_points[i]  # EMA 平滑
                smoothed_points.append(tuple(ema_points[i]))

            valid_points = np.array(smoothed_points, dtype=np.float32)

            # 保存当前帧点
            auto_points.append((frame_index, valid_points.tolist()))

            # 绘制平滑后的点
            for point in valid_points:
                cv2.circle(frame, tuple(int(x) for x in point), 5, (0, 255, 0), -1)

            prev_frame = frame.copy()
            prev_points = valid_points.reshape(-1, 1, 2)

    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# 保存自动标记点
output_file = 'auto_detected_points_video_zhuozi5_EMA.txt'
with open(output_file, 'w') as f:
    for frame_id, points in auto_points:
        points_str = ', '.join([f'[{x:.2f}, {y:.2f}]' for x, y in points])
        f.write(f'frame_idx : {frame_id}, dst_points: {points_str}\n')

print(f"Detected points saved to {output_file}")
"""

import cv2
import numpy as np
from scipy.signal import savgol_filter

# 全局变量
manual_points = []  # 第一帧手动选定的点
auto_points = []  # 每帧自动计算的点
point_trajectories = []  # 每个点的历史轨迹（用于滤波）
ema_points = []  # 每个点的 EMA 值

# 鼠标点击回调函数
def mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to start processing.")

# 图像增强函数
def enhance_image(frame, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_frame

# 加载视频
video_path = 'data/video_zhuozi5.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 手动标记第一帧的点
ret, base_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

base_frame = cv2.resize(base_frame, (512, 512))
# base_frame = enhance_image(base_frame)  # 图像增强
cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_click)

while len(manual_points) < 4:
    temp_frame = base_frame.copy()
    for point in manual_points:
        cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)  # 绘制选点
    cv2.imshow("Select 4 Points", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# 转为 NumPy 数组
manual_points = np.array(manual_points, dtype=np.float32).reshape(-1, 1, 2)

# 初始化轨迹和 EMA
for point in manual_points:
    point_trajectories.append([(point[0][0], point[0][1])])  # 初始化为点的列表
    ema_points.append(point[0])  # 初始化 EMA

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()

# 在基准帧中检测特征点
keypoints1, descriptors1 = sift.detectAndCompute(base_frame, None)

# 初始化上一帧
prev_frame = base_frame
prev_points = manual_points.copy()

# EMA 参数
alpha = 0.6  # EMA 平滑因子
max_displacement = 5  # 限制点的最大移动距离

# 处理视频帧
frame_index = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (512, 512))
    # frame = enhance_image(frame)  # 图像增强

    # 检测当前帧的特征点
    keypoints2, descriptors2 = sift.detectAndCompute(frame, None)

    # 匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 使用比值测试过滤匹配（Lowe's ratio test）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用 RANSAC 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            mapped_points = cv2.perspectiveTransform(manual_points, H)

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame, frame, mapped_points, None,
                winSize=(21, 21), maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # 融合点并限制抖动幅度
            fused_points = 0.6 * next_points + 0.4 * mapped_points
            valid_points = []
            for i, point in enumerate(fused_points.reshape(-1, 2)):
                prev_point = prev_points[i, 0]
                displacement = np.linalg.norm(point - prev_point)
                if displacement > max_displacement:
                    corrected_point = (0.8 * prev_point + 0.2 * point)
                    valid_points.append(corrected_point)
                else:
                    valid_points.append(point)

            valid_points = np.array(valid_points, dtype=np.float32)

            # 更新轨迹并进行平滑
            smoothed_points = []
            for i, point in enumerate(valid_points):
                point_trajectories[i].append(point)
                if len(point_trajectories[i]) > 5:
                    smoothed_x = savgol_filter([p[0] for p in point_trajectories[i]], window_length=5, polyorder=2)
                    smoothed_y = savgol_filter([p[1] for p in point_trajectories[i]], window_length=5, polyorder=2)
                    smoothed_point = (smoothed_x[-1], smoothed_y[-1])
                else:
                    smoothed_point = point
                smoothed_points.append(smoothed_point)

            valid_points = np.array(smoothed_points, dtype=np.float32)

            # 保存当前帧点
            auto_points.append((frame_index, valid_points.tolist()))

            # 绘制平滑后的点
            for point in valid_points:
                cv2.circle(frame, tuple(int(x) for x in point), 5, (0, 255, 0), -1)

            prev_frame = frame.copy()
            prev_points = valid_points.reshape(-1, 1, 2)

    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# 保存自动标记点
output_file = 'txt/auto_detected_points_video_zhuozi5_EMA.txt'
with open(output_file, 'w') as f:
    for frame_id, points in auto_points:
        points_str = ', '.join([f'[{x:.2f}, {y:.2f}]' for x, y in points])
        f.write(f'frame_idx : {frame_id}, dst_points: {points_str}\n')

print(f"Detected points saved to {output_file}")
