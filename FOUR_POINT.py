##################
#####纯手工标记#####
##################
import cv2
import json
import os

# 视频路径
video_path = "data/replace_original_1_1280.mp4"  # 替换为你的视频路径
output_json = "output/replace_original_1_1280.json"  # 输出 JSON 文件路径

# 检查视频文件是否存在
if not os.path.exists(video_path):
    print(f"视频文件 {video_path} 不存在！")
    exit()

# 视频加载
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件！")
    exit()

# 手动标记的点
dst_points = []
current_frame_idx = 0
frame_data = []
current_frame = None  # 保存当前帧用于实时更新

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global dst_points, current_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(dst_points) < 4:
        dst_points.append((x, y))
        print(f"已标记点: {x}, {y}")
        # 在当前帧上绘制标记点
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)  # 绘制红色圆点
        cv2.putText(current_frame, f"{len(dst_points)}", (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Video", current_frame)

# 创建窗口并绑定鼠标事件
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

print("使用鼠标左键单击标记4个点，按 'n' 键切换到下一帧，按 'q' 键退出。")

# 遍历每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频播放结束！")
        break

    # 保存当前帧，允许实时更新
    current_frame = frame.copy()

    # 在窗口中显示当前帧
    cv2.imshow("Video", current_frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # 按 'q' 键退出
        print("退出标记工具。")
        break
    elif key == ord('n'):  # 按 'n' 键切换到下一帧
        if len(dst_points) == 4:
            # 保存当前帧的标记数据
            frame_data.append({
                "frame_idx": current_frame_idx,
                "dst_points": dst_points
            })
            print(f"已保存第 {current_frame_idx} 帧的标记点: {dst_points}")
            dst_points = []  # 清空标记点
            current_frame_idx += 1
        else:
            print("请标记完整4个点后再切换到下一帧！")

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 将标记数据保存为 JSON
if frame_data:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"frames": frame_data}, f, indent=2, separators=(",", ":"))
    print(f"标记数据已保存到 {output_json}")
else:
    print("未标记任何帧，未生成 JSON 文件。")
