
import os
import pandas as pd
from colorthief import ColorThief
from PIL import Image

# 配置：图片文件夹路径 + 输出CSV路径
IMAGE_FOLDER = "dunhuang_caisson_images"  # 本地敦煌藻井图片文件夹
OUTPUT_CSV = "caisson_colors.csv"

# 批量处理函数
def batch_extract_colors():
    color_data = []
    # 遍历文件夹中所有图片
    for img_name in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue  # 跳过非图片文件

        try:
            # 提取主色调（前5种）
            color_thief = ColorThief(img_path)
            palette = color_thief.get_palette(color_count=5)  # 提取5种主色调
            # 提取图片基础信息
            with Image.open(img_path) as img:
                width, height = img.size

            # 存入数据
            color_data.append({
                "图片名称": img_name,
                "宽度": width,
                "高度": height,
                "主色调1(RGB)": palette[0],
                "主色调2(RGB)": palette[1],
                "主色调3(RGB)": palette[2]
            })
            print(f"处理完成：{img_name}")
        except Exception as e:
            print(f"处理失败 {img_name}：{str(e)}")

    # 保存为CSV
    df = pd.DataFrame(color_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"批量提取完成！结果已保存至：{OUTPUT_CSV}")

# 执行批量提取
if __name__ == "__main__":
    batch_extract_colors()