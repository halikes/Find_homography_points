import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def analyze_dst_points(json_file_path, output_file_path):
    # Load data from JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Check if 'frames' key exists
    if "frames" not in data:
        print("Error: 'frames' key not found in the JSON file.")
        return

    frames = data["frames"]

    # Extract frame indices and dst_points
    dst_points_list = []

    for frame in frames:
        if "frame_idx" not in frame or "dst_points" not in frame:
            print(f"Warning: Frame data missing 'frame_idx' or 'dst_points'. Skipping frame: {frame}")
            continue

        dst_points = np.array(frame["dst_points"])
        dst_points_list.append(dst_points)

    # Convert to numpy array for easier analysis
    if not dst_points_list:
        print("Error: No valid 'dst_points' data found in the JSON file.")
        return

    dst_points_array = np.array(dst_points_list)  # Shape: (num_frames, num_points, 2)

    # Debug: Check shape of dst_points_array
    if dst_points_array.ndim != 3 or dst_points_array.shape[1:] != (4, 2):
        print("Error: Unexpected shape of dst_points_array. Expected shape (num_frames, 4, 2).")
        return

    # Separate and smooth x and y coordinates for each point
    smoothed_dst_points_list = []

    for frame_idx in range(dst_points_array.shape[0]):
        smoothed_frame_points = []
        for i in range(4):  # Iterate over the 4 points
            x_coords = dst_points_array[:, i, 0]
            y_coords = dst_points_array[:, i, 1]

            smoothed_x = savgol_filter(x_coords, window_length=5, polyorder=2)
            smoothed_y = savgol_filter(y_coords, window_length=5, polyorder=2)

            smoothed_frame_points.append([
                round(smoothed_x[frame_idx], 2),
                round(smoothed_y[frame_idx], 2)
            ])

        smoothed_dst_points_list.append(smoothed_frame_points)

    # Create a new data structure with smoothed dst_points
    smoothed_data = {"frames": []}
    src_point =  [[184,170],[305,167],[195,452],[300,445]]
    for frame, smoothed_points in zip(frames, smoothed_dst_points_list):
        smoothed_frame = {
            "frame_idx": frame["frame_idx"],
            "src_points": src_point,
            # "src_points": frame["src_points"],
            "dst_points": smoothed_points
        }
        smoothed_data["frames"].append(smoothed_frame)
    # src [184,170],[305,167],[202,457],[293,456]
    # Save the smoothed data to a new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(smoothed_data, output_file, indent=4)

    # Plot the original and smoothed x and y coordinates for each point
    for i in range(4):
        x_coords = dst_points_array[:, i, 0]
        y_coords = dst_points_array[:, i, 1]

        smoothed_x = savgol_filter(x_coords, window_length=5, polyorder=2)
        smoothed_y = savgol_filter(y_coords, window_length=5, polyorder=2)

        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, y_coords, marker='o', color='red', label=f"Original Path - Point {i+1}", alpha=0.5)
        plt.plot(smoothed_x, smoothed_y, marker='o', color='gray', label=f"Smoothed Path - Point {i+1}", linewidth=2)
        plt.title(f"Coordinate Changes for Point {i+1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid()
        plt.show()

    print(f"Smoothed data saved to {output_file_path}")

# data/points.json output/output_new.json
json_file_path = 'output/auto_detected_points_sift_gongsi.json'
output_file_path = 'auto_detected_points_sift_gongsi.json'
analyze_dst_points(json_file_path, output_file_path)
