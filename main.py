import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

class HomographyTool:
    def __init__(self, source_img_path, target_img_path, output_dir="output"):
        self.src_points = []
        self.dst_points = []
        self.output_dir = output_dir

        self.source_img = self.load_image(source_img_path)
        self.target_img = self.load_image(target_img_path)
        os.makedirs(self.output_dir, exist_ok=True)

        self.initialize_plot()

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def initialize_plot(self):
        self.fig, (self.ax_src, self.ax_tgt) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_src.imshow(self.source_img)
        self.ax_src.set_title("Source Image (Click to select points)")
        self.ax_tgt.imshow(self.target_img)
        self.ax_tgt.set_title("Target Image (Click to select points)")

        self.add_compute_button()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        plt.show()

    def add_compute_button(self):
        button_ax = plt.axes([0.85, 0.05, 0.1, 0.075])
        self.compute_button = Button(button_ax, "Compute Homography")
        self.compute_button.on_clicked(self.compute_homography)

    def on_click(self, event):
        if event.inaxes == self.ax_src and len(self.src_points) < 4:
            self.add_point(self.src_points, event.xdata, event.ydata, self.ax_src, "red")
        elif event.inaxes == self.ax_tgt and len(self.dst_points) < 4:
            self.add_point(self.dst_points, event.xdata, event.ydata, self.ax_tgt, "blue")
        self.update_status()

    def add_point(self, points_list, x, y, axis, color):
        points_list.append([x, y])
        axis.plot(x, y, marker="o", color=color, markersize=8)
        axis.text(x, y, str(len(points_list)), color="white", fontsize=10)
        self.fig.canvas.draw()

    def update_status(self):
        self.ax_src.set_title(f"Source Image ({len(self.src_points)} points selected)")
        self.ax_tgt.set_title(f"Target Image ({len(self.dst_points)} points selected)")
        self.fig.canvas.draw()

    def compute_homography(self, event):
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            print("Please select exactly 4 points on both images!")
            return

        src_points = np.array(self.src_points, dtype=np.float32)
        dst_points = np.array(self.dst_points, dtype=np.float32)

        h_matrix, _ = cv2.findHomography(src_points, dst_points)
        warped_img = self.apply_homography(h_matrix)
        self.save_results(warped_img, h_matrix)
        self.visualize_results(warped_img)

    def apply_homography(self, h_matrix):
        height, width, _ = self.target_img.shape
        return cv2.warpPerspective(self.source_img, h_matrix, (width, height))

    def save_results(self, warped_img, h_matrix):
        warped_path = os.path.join(self.output_dir, "warped_image.png")
        matrix_path = os.path.join(self.output_dir, "homography_matrix.txt")
        cv2.imwrite(warped_path, cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))
        np.savetxt(matrix_path, h_matrix, fmt="%.6f")
        print(f"Warped image saved to {warped_path}")
        print(f"Homography matrix saved to {matrix_path}")

    def visualize_results(self, warped_img):
        plt.figure(figsize=(8, 6))
        plt.imshow(warped_img)
        plt.title("Warped Source Image")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    source_img_path = "data/7_source.png"
    target_img_path = "data/video_frame/video_frame_0.png"
    HomographyTool(source_img_path, target_img_path)
