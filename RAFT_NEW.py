import cv2
import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# ----------------------- 参数解析 -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='data/sample_video_002.mp4', help='Path to target video')
parser.add_argument('--object_file', type=str, default='data/source.png', help='Path to object image')
parser.add_argument('--output_dir', type=str, default='results/vo', help='Directory for saving output')
parser.add_argument('--resize_width', type=int, default=512, help='Width to resize frames')
parser.add_argument('--sample_step', type=int, default=10, help='采样步长，用于从参考帧中均匀采样点')
parser.add_argument('--ransac_thresh', type=float, default=5.0, help='RANSAC重投影误差阈值')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------- 交互式对象插入模块 -----------------------
def nothing(x):
    pass

def insertion_mouse_callback(event, x, y, flags, param):
    # 用于交互式调整插入位置和大小
    global mouse_coord
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
    source_img[y:y + new_h, x:x + new_w] = roi_smoothed  # 注意这里确保区域尺寸一致
    mask_img = source_img.copy()
    mask_img[mask_img > 0] = 255

    final_params = {
        "insertion_position": {"x": x, "y": y},
        "scale": scale_factor,
        "object_size": {"w": new_w, "h": new_h}
    }

    cv2.imwrite(os.path.join(output_dir, "source_img_sample_video_06.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "mask_img_sample_video_06.png"), mask_img)

    return fused_img, source_img, mask_img, final_params, orig_frame

# ----------------------- 纹理增强模块 -----------------------
def enhance_texture(frame):
    """
    对输入BGR图像进行CLAHE和简单锐化，增强纹理信息
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    enhanced = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ----------------------- 全局Homography估计模块 -----------------------
def compute_global_homography(ref_frame, curr_frame, sample_step, ransac_thresh):
    """
    利用RAFT密集光流计算参考帧到当前帧的全局Homography：
      1. 对参考帧和当前帧进行纹理增强；
      2. 利用RAFT计算光流；
      3. 从参考帧中均匀采样像素，得到参考点集；
      4. 利用光流获得对应的目标点集；
      5. 使用RANSAC计算全局Homography矩阵。
    """
    # 增强图像
    ref_enh = enhance_texture(ref_frame)
    curr_enh = enhance_texture(curr_frame)
    
    # 转换为tensor并归一化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_tensor = torch.from_numpy(ref_enh).permute(2,0,1).unsqueeze(0).float() / 255.0
    curr_tensor = torch.from_numpy(curr_enh).permute(2,0,1).unsqueeze(0).float() / 255.0
    ref_tensor = ref_tensor.to(device)
    curr_tensor = curr_tensor.to(device)
    
    # 使用RAFT计算光流
    with torch.no_grad():
        flow_outputs = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()(ref_tensor, curr_tensor)
        flow = flow_outputs[-1]
    flow = flow.squeeze(0).cpu().numpy().transpose(1,2,0)  # (H, W, 2)
    
    H_img, W_img = ref_frame.shape[:2]
    # 从参考帧均匀采样点
    grid_y, grid_x = np.mgrid[0:H_img:sample_step, 0:W_img:sample_step]
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    src_points = np.stack([grid_x, grid_y], axis=1).astype(np.float32)
    
    # 得到对应的目标点：直接加上光流值
    # 注意：光流图的坐标顺序为 (row, col)
    flow_sampled = flow[grid_y, grid_x]  # (N, 2)
    dst_points = src_points + flow_sampled
    
    # 计算Homography
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_thresh)
    if H is None:
        H = np.eye(3, dtype=np.float32)
    return H

def track_global_homography(video_file, resize_dim, sample_step, ransac_thresh, visualize=False):
    """
    利用全局深度光流计算每一帧与参考帧之间的全局Homography，
    返回每一帧的Homography矩阵列表。参考帧为第一帧。
    """
    cap = cv2.VideoCapture(video_file)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read video file")
    ref_frame = cv2.resize(ref_frame, resize_dim)
    
    homographies = []
    # 第一帧的Homography设为单位矩阵
    homographies.append(np.eye(3, dtype=np.float32).tolist())
    
    frame_idx = 1
    if visualize:
        cv2.namedWindow("Global Homography Tracking", cv2.WINDOW_NORMAL)
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        H = compute_global_homography(ref_frame, frame, sample_step, ransac_thresh)
        homographies.append(H.tolist())
        
        if visualize:
            # 利用H将参考帧透视变换到当前帧
            ref_warp = cv2.warpPerspective(ref_frame, H, (resize_dim[0], resize_dim[1]))
            vis = cv2.addWeighted(frame, 0.5, ref_warp, 0.5, 0)
            cv2.imshow("Global Homography Tracking", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pbar.update(1)
    pbar.close()
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    return homographies

# ----------------------- 轨迹平滑与可视化 -----------------------
def smooth_trajectories(homographies, window_length=15, polyorder=2):
    homos = np.array(homographies)
    smoothed = savgol_filter(homos, window_length=window_length, polyorder=polyorder, axis=0)
    return smoothed

def plot_homography_trajectory(homographies, resize_dim):
    # 这里简单绘制每一帧Homography的平移部分
    traj = [H[0:2,2] for H in homographies]
    traj = np.array(traj)
    plt.figure(figsize=(8,8))
    plt.plot(traj[:,0], traj[:,1], 'o-')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Global Homography Translation")
    plt.grid(True)
    plt.show()

def compute_homography_for_frames(homographies):
    return [{"frame_idx": idx, "homography_matrix": H} for idx, H in enumerate(homographies)]

# ----------------------- 主程序 -----------------------
def main():
    # 交互式选择插入区域（保留原有插入模块，用于后续插入物体）
    print("Starting interactive insertion...")
    fused_img, source_img, mask_img, ins_params, orig_frame = interactive_insertion(
        args.video_file, args.object_file, args.resize_width, args.output_dir)
    cv2.imshow("Final Insertion", fused_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    # 此处暂不使用用户选点作为参考，纯使用深度光流计算全局Homography
    resize_dim = (args.resize_width, args.resize_width)
    homographies = track_global_homography(args.video_file, resize_dim, args.sample_step, args.ransac_thresh, visualize=True)
    print(f"Tracked global homographies obtained on {len(homographies)} frames.")
    
    final_homography = compute_homography_for_frames(homographies)
    homo_json = os.path.join(args.output_dir, "global_homography_results_dense.json")
    with open(homo_json, "w") as f:
        json.dump(final_homography, f, indent=4)
    print("Homography results saved to", homo_json)
    
    smoothed = smooth_trajectories(np.array(homographies), window_length=15, polyorder=2)
    plot_homography_trajectory(smoothed, resize_dim)

if __name__ == "__main__":
    main()
