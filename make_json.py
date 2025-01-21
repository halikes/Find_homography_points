import json
import re  # 引入正则表达式库

# 固定的 src_points
src_points = [[184, 169], [306, 169], [193, 449], [299, 449]]

# 输入文件路径
input_file = "txt/auto_detected_points_sift_gongsi.txt"  # 替换为你的输入文件路径
output_file = "output/auto_detected_points_sift_gongsi.json"  # 输出 JSON 文件路径

# 定义正则表达式，匹配 frame_idx 和 dst_points
pattern = r"frame_idx\s*:\s*(\d+),\s*dst_points\s*:\s*(.*)"

frames = []
with open(input_file, "r") as f:
    for line in f:
        match = re.search(pattern, line.strip())
        if match:
            frame_idx = int(match.group(1))  # 提取 frame_idx
            dst_points_str = match.group(2)  # 提取 dst_points 内容

            # 去掉开头的 `[` 和结尾的 `]`，再用 `split` 拆分每组坐标
            dst_points_str = dst_points_str.strip("[]")
            dst_points = [
                [float(x) for x in point.split(", ")]
                for point in dst_points_str.split("], [")
            ]

            # 构建单帧数据
            frames.append({
                "frame_idx": frame_idx,
                "src_points": src_points,
                "dst_points": dst_points
            })
        else:
            print(f"无法解析行：{line.strip()}")  # 输出无法解析的行，便于调试

# 将数据写入 JSON 文件
output_data = {"frames": frames}
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2, separators=(",", ":"))

print(f"JSON 文件已生成：{output_file}")
