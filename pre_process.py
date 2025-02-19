import cv2
import numpy as np
import argparse
import os
import json
from PIL import Image
from scipy.signal import savgol_filter

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/Zoom_IN_OUT.mp4', help='Path to target video')
parser.add_argument('--object_file', type=str, default='data/source.png', help='Path to object image')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the first frame')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 全局变量 -----------------------
# 初始插入位置
obj_pos = [50, 50]
# 初始缩放因子
scale_factor = 1.0
# 当前鼠标坐标
mouse_coord = (0, 0)


# ----------------------- 鼠标回调（交互式对象插入） -----------------------
def nothing(x):
    pass


def insertion_mouse_callback(event, x, y, flags, param):
    global obj_pos, mouse_coord
    mouse_coord = (x, y)
    # param 中包含对象区域信息和拖动状态
    (ox, oy, ow, oh) = param['obj_bbox']
    if event == cv2.EVENT_LBUTTONDOWN:
        # 若点击在对象区域内则开始拖动
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


# ----------------------- 交互式对象插入模块 -----------------------
def interactive_insertion():
    """
    交互式调整物体插入目标视频第一帧：
      - 显示目标帧，允许用户拖动和调整物体图像的缩放
      - 不绘制对象边框，仅显示鼠标坐标
      - 结束后生成：
            fused_img: 融合后的目标帧（含插入对象）
            source_img: 黑色背景，仅保留插入对象区域
            mask_img: 与目标帧同尺寸的二值图（插入区域白，其余黑）
            final_params: 插入参数（位置和尺寸）
      - 同时返回原始视频第一帧，用于后续 Homography 计算
    """
    # 读取视频第一帧
    cap = cv2.VideoCapture(args.video_file)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video.")
        cap.release()
        exit()
    cap.release()
    # Resize 第一帧
    h0, w0 = first_frame.shape[:2]
    scale = args.resize_width / float(w0)
    first_frame = cv2.resize(first_frame, (args.resize_width, int(h0 * scale)))
    target_frame = first_frame.copy()  # 融合背景
    orig_frame = first_frame.copy()  # 原始第一帧（后续用于Homography计算）

    # 加载对象图像（转换为 BGR 格式）
    object_img = np.array(Image.open(args.object_file).convert('RGB'))
    object_img = cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR)

    # 初始化对象参数
    orig_obj_h, orig_obj_w = object_img.shape[:2]
    obj_pos = [50, 50]
    global scale_factor
    scale_factor = 1.0

    # 参数字典，用于共享对象位置和尺寸信息
    param = {
        'obj_pos': obj_pos,
        'obj_bbox': (obj_pos[0], obj_pos[1], orig_obj_w, orig_obj_h),
        'dragging': False,
        'drag_offset': [0, 0]
    }

    window_name = "Insertion Adjustment"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, insertion_mouse_callback, param)
    cv2.createTrackbar("Scale", window_name, 100, 200, nothing)  # 缩放范围50%-200%

    while True:
        scale_val = cv2.getTrackbarPos("Scale", window_name) / 100.0
        scale_factor = scale_val
        new_w = int(orig_obj_w * scale_factor)
        new_h = int(orig_obj_h * scale_factor)
        cur_obj_img = cv2.resize(object_img, (new_w, new_h))
        param['obj_bbox'] = (param['obj_pos'][0], param['obj_pos'][1], new_w, new_h)

        display_img = target_frame.copy()
        x, y = param['obj_pos']
        H, W = display_img.shape[:2]
        x = max(0, min(x, W - new_w))
        y = max(0, min(y, H - new_h))
        param['obj_pos'][0] = x
        param['obj_pos'][1] = y

        # 融合对象图像到目标帧，直接覆盖
        display_img[y:y + new_h, x:x + new_w] = cur_obj_img

        # 显示鼠标坐标
        cv2.putText(display_img, f"Mouse: {mouse_coord}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc确认结束
            break
    cv2.destroyAllWindows()

    # 融合结果即为 display_img
    fused_img = display_img.copy()

    # 生成 source_img：创建黑色背景，插入对象区域保持原样
    source_img = np.zeros_like(fused_img)
    source_img[y:y + new_h, x:x + new_w] = fused_img[y:y + new_h, x:x + new_w]

    # 生成 mask_img
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255

    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }
    # 保存结果
    cv2.imwrite(os.path.join(args.output_dir, "final_insertion_result.png"), fused_img)
    cv2.imwrite(os.path.join(args.output_dir, "source_img.png"), source_img)
    cv2.imwrite(os.path.join(args.output_dir, "mask_img.png"), mask_img)
    with open(os.path.join(args.output_dir, "insertion_params.json"), "w") as f:
        json.dump(final_params, f, indent=4)
    print("Final insertion parameters:", final_params)

    return fused_img, source_img, mask_img, final_params, orig_frame


# ----------------------- SIFT+边缘校正与全局 Homography 模块 -----------------------
manual_points = []  # 存放用户点击的四个点


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
    distances = np.sqrt((indices[:, 1] - point[0])**2 + (indices[:, 0] - point[1])**2)
    idx = np.argmin(distances)
    best_y, best_x = indices[idx]
    return (int(best_x), int(best_y))

def select_and_correct_points(frame):
    """
    在给定帧上，用户手动选取4个点，然后利用 Canny 边缘检测校正选点，
    返回校正后的点列表（作为源点）。
    """
    global manual_points
    manual_points = []
    disp = frame.copy()
    cv2.namedWindow("Select 4 Points for Homography")
    cv2.setMouseCallback("Select 4 Points for Homography", sift_mouse_click)
    while len(manual_points) < 4:
        temp = disp.copy()
        for pt in manual_points:
            cv2.circle(temp, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Points for Homography", temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Select 4 Points for Homography")
    # 利用 Canny 边缘检测对选取的点进行校正
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    corrected_points = []
    for pt in manual_points:
        cp = find_nearest_edge_point(pt, edges)
        corrected_points.append(cp)
        print(f"Original: {pt}, Corrected: {cp}")
    return corrected_points

def track_points(video_file, initial_points, resize_dim):
    """
    利用光流跟踪，将第一帧校正后的关键点在整个视频中跟踪，
    返回每一帧的关键点列表，格式为 list，每个元素为4个点的列表。
    """
    cap = cv2.VideoCapture(video_file)
    tracked_points = []
    prev_points = np.array(initial_points, dtype=np.float32).reshape(-1, 1, 2)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return tracked_points
    frame = cv2.resize(frame, resize_dim)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_points.append([tuple(pt[0]) for pt in prev_points])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None)
        # 选择有效跟踪点
        good_points = []
        for i, st in enumerate(status):
            if st[0] == 1:
                good_points.append(next_points[i])
            else:
                good_points.append(prev_points[i])
        prev_points = np.array(good_points, dtype=np.float32)
        tracked_points.append([tuple(pt[0]) for pt in prev_points])
        prev_gray = gray_frame.copy()
    cap.release()
    return tracked_points

def compute_homography_for_frames(src_points, tracked_points):
    """
    对于视频中每一帧，利用第一帧的校正点 src_points 作为源点，
    当前帧的跟踪点作为目标点，计算 Homography 矩阵，
    返回包含每一帧 Homography 的列表。
    """
    homography_results = []
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    for idx, pts in enumerate(tracked_points):
        dst = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": idx, "homography_matrix": H.tolist()})
    return homography_results


# ----------------------- 主程序 -----------------------
def main():
    # 第一步：交互式对象插入与后处理
    print("Starting interactive insertion...")
    fused_img, source_img, mask_img, ins_params, orig_frame = interactive_insertion()
    cv2.imshow("Final Insertion", fused_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # 第二步：在原始第一帧上选取4个点并校正
    print("Now select 4 points for global Homography computation on the original frame...")
    corrected_src_points = select_and_correct_points(orig_frame)
    print("Corrected source points:", corrected_src_points)

    # 获取视频第一帧尺寸（已resize）
    resize_dim = (orig_frame.shape[1], orig_frame.shape[0])
    # 利用光流跟踪，将这4个点在视频中跟踪
    tracked_points = track_points(args.video_file, corrected_src_points, resize_dim)
    print(f"Tracked points on {len(tracked_points)} frames obtained.")

    # 计算每一帧的 Homography，源点为第一帧校正后的点
    homography_results = compute_homography_for_frames(corrected_src_points, tracked_points)
    homography_json = os.path.join(args.output_dir, "global_homography_results.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)

    # 显示第一帧校正后的点
    disp = orig_frame.copy()
    for pt in corrected_src_points:
        cv2.circle(disp, pt, 5, (0, 255, 0), -1)
    cv2.imshow("Corrected Source Points", disp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
