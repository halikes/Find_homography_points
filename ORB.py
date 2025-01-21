# 速度是很快，精度低于SIFT
##############################
##ORB自动寻找homography目标点###
##############################

import cv2
import numpy as np

# 全局变量
manual_points = []  # 第一帧手动选定的点
auto_points = []  # 每帧自动计算的点

# 鼠标点击回调函数
def mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to start processing.")

# 加载视频
video_path = 'data/replace_original_1_1280.mp4'
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

# 调整第一帧大小为 512x512
base_frame = cv2.resize(base_frame, (512, 512))
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

# 初始化 ORB 特征检测器
orb = cv2.ORB_create()

# 在基准帧中检测特征点
keypoints1, descriptors1 = orb.detectAndCompute(base_frame, None)

# 初始化上一帧
prev_frame = base_frame
prev_points = manual_points.copy()

# 处理视频帧
frame_index = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧大小为 512x512
    frame = cv2.resize(frame, (512, 512))

    # 检测当前帧的特征点
    keypoints2, descriptors2 = orb.detectAndCompute(frame, None)

    # 匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点对
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        # 基于单应性变换预测当前帧的点
        mapped_points = cv2.perspectiveTransform(manual_points, H)

        # 使用光流调整预测点位置
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, frame, mapped_points, None, winSize=(15, 15), maxLevel=2,
                                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 将单应性和光流结果融合
        fused_points = 0.6 * next_points + 0.4 * mapped_points

        # 保存当前帧点
        valid_points = fused_points.reshape(-1, 2)
        auto_points.append((frame_index, valid_points.tolist()))

        # 在当前帧绘制点
        for point in valid_points:
            cv2.circle(frame, tuple(int(x) for x in point), 5, (0, 255, 0), -1)

        # 更新上一帧信息
        prev_frame = frame.copy()
        prev_points = valid_points.reshape(-1, 1, 2)

    # 显示处理结果
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# 保存自动标记点
output_file = 'txt/auto_detected_points_replace_1.txt'
with open(output_file, 'w') as f:
    for frame_id, points in auto_points:
        points_str = ', '.join([f'[{x:.2f}, {y:.2f}]' for x, y in points])
        f.write(f'frame_idx : {frame_id}, dst_points: [{points_str}]\n')

print(f"Detected points saved to {output_file}")
