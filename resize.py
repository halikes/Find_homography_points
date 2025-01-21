import cv2

# 加载原始图片
input_image_path = 'data/mask.png'  # 替换为你的图片路径
output_image_path = 'mask_512.png'  # 保存的图片路径

# 读取图片
image = cv2.imread(input_image_path)

# 检查图片是否成功加载
if image is None:
    print("Error: Could not load image.")
    exit()

# 调整大小到 512x512
resized_image = cv2.resize(image, (512, 512))

# 保存调整后的图片
cv2.imwrite(output_image_path, resized_image)

print(f"Resized image saved to {output_image_path}")
