import cv2
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_01.mp4', help='Path to target video')
parser.add_argument('--object_file', type=str, default='data/source.png', help='Path to object image')
parser.add_argument('--output_dir', type=str, default='results/insertion', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize the first frame')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 全局变量 -----------------------
obj_pos = [0, 0]
scale_factor = 1.0
mouse_coord = (0, 0)
manual_points = []  # use to SIFT


# ----------------------- 交互式对象插入模块 -----------------------
def nothing(x):
    pass


def insertion_mouse_callback(event, x, y, flags, param):
    global obj_pos, mouse_coord
    mouse_coord = (x, y)
    (ox, oy, ow, oh) = param['obj_bbox']
    if event == cv2.EVENT_LBUTTONDOWN:
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


def interactive_insertion(video_file, object_file, resize_width, output_dir):
    # 读取视频第一帧
    cap = cv2.VideoCapture(video_file)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read video file: " + video_file)
    cap.release()

    # Resize 第一帧
    h0, w0 = first_frame.shape[:2]
    scale = resize_width / float(w0)
    first_frame = cv2.resize(first_frame, (resize_width, int(h0 * scale)))
    target_frame = first_frame.copy()
    orig_frame = first_frame.copy()

    # 加载对象图像（转换为 BGR 格式）
    object_img = np.array(Image.open(object_file).convert('RGB'))
    object_img = cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR)

    orig_obj_h, orig_obj_w = object_img.shape[:2]
    obj_pos = [50, 50]
    global scale_factor
    scale_factor = 1.0

    param = {
        'obj_pos': obj_pos,
        'obj_bbox': (obj_pos[0], obj_pos[1], orig_obj_w, orig_obj_h),
        'dragging': False,
        'drag_offset': [0, 0]
    }

    window_name = "Insertion Adjustment"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, insertion_mouse_callback, param)
    cv2.createTrackbar("Scale", window_name, 100, 200, nothing)

    while True:
        scale_val = cv2.getTrackbarPos("Scale", window_name) / 100.0
        scale_factor = scale_val
        new_w = int(orig_obj_w * scale_factor)
        new_h = int(orig_obj_h * scale_factor)
        cur_obj_img = cv2.resize(object_img, (new_w, new_h))
        param['obj_bbox'] = (param['obj_pos'][0], param['obj_pos'][1], new_w, new_h)

        display_img = target_frame.copy()
        x, y = param['obj_pos']
        H_disp, W_disp = display_img.shape[:2]
        x = max(0, min(x, W_disp - new_w))
        y = max(0, min(y, H_disp - new_h))
        param['obj_pos'][0] = x
        param['obj_pos'][1] = y

        # 融合对象图像到目标帧（直接覆盖）
        display_img[y:y + new_h, x:x + new_w] = cur_obj_img
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

    fused_img = display_img.copy()

    # source_img
    source_img = np.zeros_like(fused_img)
    source_img[y:y + new_h, x:x + new_w] = cur_obj_img

    roi = source_img[y:y + new_h, x:x + new_w]
    roi_smoothed = cv2.bilateralFilter(roi, d=5, sigmaColor=75, sigmaSpace=75)
    source_img[y:y + new_h, x:x + new_w] = roi_smoothed

    # mask_img
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255


    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }

    cv2.imwrite(os.path.join(output_dir, "source_img_sample_video_01.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "mask_img_sample_video_01.png"), mask_img)

    return fused_img, source_img, mask_img, final_params, orig_frame


# ----------------------- SIFT+边缘校正与光流跟踪模块 -----------------------
manual_points = []  # 用于存放用户点击的四个点


def sift_mouse_click(event, x, y, flags, param):
    """
    鼠标回调：手动选取点
    """
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_points) < 4:
        manual_points.append((x, y))
        print(f"Selected point {len(manual_points)}: ({x}, {y})")
        if len(manual_points) == 4:
            print("4 points selected. Press any key to proceed.")


def find_nearest_edge_point(point, edge_image):
    """
    对于给定的点，在边缘图像中寻找最近的边缘点
    """
    indices = np.argwhere(edge_image > 0)
    if len(indices) == 0:
        return point
    distances = np.sqrt((indices[:, 1] - point[0]) ** 2 + (indices[:, 0] - point[1]) ** 2)
    idx = np.argmin(distances)
    best_y, best_x = indices[idx]
    return (int(best_x), int(best_y))


def select_and_correct_points(frame):
    """
    在给定帧上，用户手动选取4个点，然后利用 Canny 边缘检测对这些点进行校正，
    返回校正后的点列表。
    """
    global manual_points
    manual_points = []
    window_name = "Select 4 Points for Homography"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, sift_mouse_click)
    while len(manual_points) < 4:
        disp = frame.copy()
        for pt in manual_points:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        cv2.imshow(window_name, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)

    # 边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    corrected_points = []
    for pt in manual_points:
        cp = find_nearest_edge_point(pt, edges)
        corrected_points.append(cp)
        print(f"Original: {pt}, Corrected: {cp}")
    return corrected_points


def track_points(video_file, initial_points, resize_dim, visualize=False):
    """
    利用光流跟踪，将初始的4个校正点在整个视频中跟踪，
    返回一个列表，每个元素为当前帧跟踪到的4个点（格式：[(x,y), ...]）。
    如果 visualize=True，则实时显示窗口“Optical Flow Tracking”。
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

    if visualize:
        window_name = "Optical Flow Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resize_dim[0], resize_dim[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None)
        good_points = []
        for i, st in enumerate(status):
            if st[0] == 1:
                good_points.append(next_points[i])
            else:
                good_points.append(prev_points[i])
        prev_points = np.array(good_points, dtype=np.float32)
        tracked_points.append([tuple(pt[0]) for pt in prev_points])
        prev_gray = gray_frame.copy()

        if visualize:
            vis_frame = frame.copy()
            for pt in prev_points:
                cv2.circle(vis_frame, (int(pt[0][0]), int(pt[0][1])), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if visualize:
        cv2.destroyWindow(window_name)
    cap.release()
    return tracked_points


def smooth_trajectories(tracked_points, window_length=15, polyorder=2):
    """
    对每个选点的轨迹进行 Savitzky-Golay 平滑，返回平滑后的轨迹列表。
    tracked_points：list，每个元素为一帧跟踪到的4个点（格式：[(x,y), ...]）
    返回：list，每个元素为平滑后的4个点
    """
    num_frames = len(tracked_points)
    num_points = len(tracked_points[0])
    # 转换为 numpy 数组：形状 (num_frames, num_points, 2)
    traj = np.array(tracked_points)
    smoothed = np.zeros_like(traj)
    # 对每个点独立平滑 x 和 y 坐标
    for i in range(num_points):
        x_coords = traj[:, i, 0]
        y_coords = traj[:, i, 1]
        # 调整窗口长度：必须为奇数且不超过数据长度
        win_len = window_length if window_length <= len(x_coords) and window_length % 2 == 1 else (len(x_coords) // 2) * 2 + 1
        smooth_x = savgol_filter(x_coords, window_length=win_len, polyorder=polyorder)
        smooth_y = savgol_filter(y_coords, window_length=win_len, polyorder=polyorder)
        smoothed[:, i, 0] = smooth_x
        smoothed[:, i, 1] = smooth_y
    return smoothed


def plot_trajectories(original_traj, smoothed_traj):
    """
    original_traj: list，每个元素为一帧跟踪到的4个点，格式为 [(x,y), (x,y), (x,y), (x,y)]
    smoothed_traj: numpy数组，形状 (num_frames, 4, 2)，表示平滑后的轨迹
    """
    num_frames = len(original_traj)
    num_points = len(original_traj[0])

    # 将原始轨迹转换为 numpy 数组，形状 (num_frames, num_points, 2)
    orig_array = np.array(original_traj)

    plt.figure(figsize=(15, 15))
    colors = ['r', 'g', 'b', 'c']

    for i in range(num_points):
        # 提取第 i 个点的原始轨迹 (num_frames, 2)
        orig_points = orig_array[:, i, :]
        # 提取第 i 个点的平滑轨迹 (num_frames, 2)
        smooth_points = smoothed_traj[:, i, :]

        # 绘制平滑轨迹（用方块虚线连线）
        plt.plot(smooth_points[:, 0], smooth_points[:, 1], 's--', color=colors[i], alpha=0.5,
                 label=f"Smoothed Point {i + 1}")
        # 绘制原始轨迹（用圆点连线，颜色稍透明）
        plt.plot(orig_points[:, 0], orig_points[:, 1], 'o-', color= 'gray',
                 label=f"Original Point {i + 1}")


    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Original and Smoothed Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_homography_for_frames(src_points, smoothed_traj):
    """
    对于视频中每一帧，利用第一帧校正后的点 src_points 作为源点，
    当前帧平滑后的跟踪点作为目标点计算 Homography 矩阵，
    返回包含每一帧 Homography 的列表。
    """
    homography_results = []
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    num_frames = smoothed_traj.shape[0]
    for idx in range(num_frames):
        dst = smoothed_traj[idx, :, :].reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        homography_results.append({"frame_idx": idx, "homography_matrix": H.tolist()})
    return homography_results


# ----------------------- 主程序 -----------------------
def main():

    # 第一步：交互式对象插入（这里调用已有模块，返回融合图 fused_img, source_img, mask_img, 参数, 以及原始第一帧）
    print("Starting interactive insertion...")
    fused_img, source_img, mask_img, ins_params, orig_frame = interactive_insertion(
        args.video_file, args.object_file, args.resize_width, args.output_dir)
    cv2.imshow("Final Insertion", fused_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # 第二步：在原始第一帧上手动选取4个点并校正
    print("Now select 4 points for global Homography computation on the original frame...")
    corrected_src_points = select_and_correct_points(orig_frame)
    print("Corrected source points:", corrected_src_points)

    # 第三步：利用光流跟踪，将校正后的4个点在整个视频中跟踪
    resize_dim = (orig_frame.shape[1], orig_frame.shape[0])
    tracked_points = track_points(args.video_file, corrected_src_points, resize_dim, visualize=True)
    print(f"Tracked points obtained on {len(tracked_points)} frames.")

    # 第四步：对跟踪轨迹进行平滑处理
    smoothed_traj = smooth_trajectories(tracked_points, window_length=15, polyorder=2)
    # 可视化平滑后的轨迹
    plot_trajectories(tracked_points, smoothed_traj)

    # 第五步：计算每一帧的 Homography，源点为第一帧校正后的点，目标点为平滑后的轨迹点
    homography_results = compute_homography_for_frames(corrected_src_points, smoothed_traj)
    homography_json = os.path.join(args.output_dir, "global_homography_results_sample_video_01.json")
    with open(homography_json, "w") as f:
        json.dump(homography_results, f, indent=4)
    print("Homography results saved to", homography_json)


if __name__ == "__main__":
    main()