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
from torchvision.transforms.functional import to_tensor
from torchvision.models.optical_flow.raft.utils.utils import InputPadder  # Import InputPadder from the RAFT utilities

def compute_optical_flow_raft(model, frame1, frame2):
    """
    计算帧 frame1 -> frame2 的光流，使用 RAFT。
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 预处理图像
    frame1_tensor = to_tensor(frame1).unsqueeze(0).to(device)
    frame2_tensor = to_tensor(frame2).unsqueeze(0).to(device)
    padder = InputPadder(frame1_tensor.shape)
    frame1_tensor, frame2_tensor = padder.pad(frame1_tensor, frame2_tensor)

    # 计算光流
    with torch.no_grad():
        flow_low, flow_up = model(frame1_tensor, frame2_tensor, iters=20, test_mode=True)
    
    # 转换为 numpy 格式
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
    return flow

def compute_pairwise_homography(flow, region_pts):
    """
    通过光流计算从前一帧到当前帧的 Homography
    """
    h, w = flow.shape[:2]
    src_pts = np.float32(region_pts).reshape(-1, 1, 2)
    
    # 计算光流位移
    displacement = flow[region_pts[:, 1].astype(int), region_pts[:, 0].astype(int)]
    dst_pts = src_pts + displacement.reshape(-1, 1, 2)

    # 计算 Homography
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return np.eye(3, dtype=np.float32)
    
    return H

def compute_global_homography(tracked_regions, flow_maps):
    """
    计算全局 Homography（相对于第一帧的变换）
    """
    num_frames = len(tracked_regions)
    H_matrices = [np.eye(3, dtype=np.float32)]  # 初始单位矩阵

    for i in range(1, num_frames):
        H_t = compute_pairwise_homography(flow_maps[i - 1], np.array(tracked_regions[i - 1]))
        
        # 累积 H_t = H_t-1 * H_t
        H_t_global = H_matrices[-1] @ H_t
        H_matrices.append(H_t_global)

    return H_matrices

def compute_sift_homography(frames):
    """
    计算 SIFT 估计的 Homography（仅用于融合）
    """
    sift = cv2.SIFT_create()
    H_matrices = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(frames)):
        # 计算 SIFT 特征点和描述符
        kp1, des1 = sift.detectAndCompute(frames[i-1], None)
        kp2, des2 = sift.detectAndCompute(frames[i], None)

        # 进行特征点匹配
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # 选取优质匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            H = np.eye(3, dtype=np.float32)

        # 累积计算 H
        H_t_global = H_matrices[-1] @ H
        H_matrices.append(H_t_global)

    return H_matrices

def refine_homography_sift_raft(H_matrices, sift_H_matrices):
    """
    结合 RAFT 计算的 H_matrices 和 SIFT 计算的 H_matrices
    """
    alpha = 0.7  # 70% 采用 RAFT 结果，30% 采用 SIFT 结果
    refined_H = []

    for i in range(len(H_matrices)):
        H_refined = alpha * H_matrices[i] + (1 - alpha) * sift_H_matrices[i]
        refined_H.append(H_refined)

    return refined_H

def smooth_homography(H_matrices, window_size=5, polyorder=2):
    """
    对 Homography 进行 Savitzky-Golay 平滑，减少抖动
    """
    H_smoothed = []

    for i in range(9):  # H 矩阵是 3x3，共 9 个元素
        data = [H[i // 3, i % 3] for H in H_matrices]
        smoothed_data = savgol_filter(data, window_size, polyorder)
        for j in range(len(H_matrices)):
            if j >= len(H_smoothed):
                H_smoothed.append(np.eye(3, dtype=np.float32))
            H_smoothed[j][i // 3, i % 3] = smoothed_data[j]

    return H_smoothed

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    model = RAFT()  # 加载预训练的 RAFT
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    frames = []
    tracked_regions = []
    flow_maps = []

    # 读取所有帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # 转为灰度

    cap.release()

    # 计算光流
    for i in range(len(frames) - 1):
        flow = compute_optical_flow_raft(model, frames[i], frames[i + 1])
        flow_maps.append(flow)

    # 计算 Homography
    H_raft = compute_global_homography(tracked_regions, flow_maps)
    H_sift = compute_sift_homography(frames)
    H_combined = refine_homography_sift_raft(H_raft, H_sift)
    H_smoothed = smooth_homography(H_combined)

    return H_smoothed

