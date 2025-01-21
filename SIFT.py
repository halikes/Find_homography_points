import cv2
import numpy as np
import json

# 全局变量
manual_points = []  # 手动选定的目标点
trajectories = []  # 保存目标点的轨迹
homography_frames = []  # 用于保存每帧的 Homography 矩阵

# 鼠标点击回调函数，用于手动标记第一帧
def mouse_click(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:  # 最多支持4个点
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
    return enhanced_gray

# 特征点匹配函数（SIFT）
def match_features(img1, img2):
    sift = cv2.SIFT_create()  # 创建SIFT检测器
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配的关键点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return pts1, pts2

# 加载视频
video_path = 'data/gongsiyuanben_biaozhun4s.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 读取第一帧并调整大小
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

prev_frame = cv2.resize(prev_frame, (1024, 1024))
prev_gray = enhance_image(prev_frame)  # 图像增强

# 显示第一帧并手动标记目标点
cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", mouse_click)

while len(manual_points) < 4:  # 等待用户标记4个点
    temp_frame = prev_frame.copy()
    for point in manual_points:
        cv2.circle(temp_frame, point, 2, (0, 0, 255), -1)  # 显示标记的点
    cv2.imshow("Select Points", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# 转为 NumPy 数组
manual_points = np.array(manual_points, dtype=np.float32).reshape(-1, 1, 2)
trajectories.append(manual_points.copy())  # 保存初始点到轨迹

# 第0帧的Homography矩阵为None
homography_frames.append({"frame_idx": 0, "matrix": None})

# 处理视频帧
frame_index = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 1024))
    curr_gray = enhance_image(frame)  # 图像增强

    # 特征点匹配并计算单应性矩阵
    pts1, pts2 = match_features(prev_gray, curr_gray)
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 保存单应性矩阵
    if H is not None:
        homography_frames.append({"frame_idx": frame_index, "matrix": H.tolist()})
        compensated_points = cv2.perspectiveTransform(manual_points, H)
        manual_points = compensated_points  # 更新补偿后的点
    else:
        print(f"Frame {frame_index}: Homography computation failed.")
        homography_frames.append({"frame_idx": frame_index, "matrix": None})

    # 保存轨迹
    trajectories.append(manual_points.copy())

    # 绘制轨迹和当前点
    for point in manual_points:
        cv2.circle(frame, tuple(int(x) for x in point[0]), 5, (0, 255, 0), -1)

    # 显示结果
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 更新上一帧
    prev_gray = curr_gray.copy()
    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# 保存轨迹
output_trajectory_file = 'txt/auto_detected_points_sift_gongsi.txt'
with open(output_trajectory_file, 'w') as f:
    for frame_id, traj in enumerate(trajectories):
        points_str = ', '.join([f'[{x[0][0]:.2f}, {x[0][1]:.2f}]' for x in traj])
        f.write(f'frame_idx : {frame_id}, dst_points : {points_str}\n')

print(f"Trajectories saved to {output_trajectory_file}")

# 保存Homography矩阵到JSON文件
output_homography_file = 'txt/homography_matrices.json'
with open(output_homography_file, 'w') as f:
    json.dump({"frames": homography_frames}, f, indent=4)

print(f"Homography matrices saved to {output_homography_file}")
