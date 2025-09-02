import cv2
import numpy as np

def clean_captcha(image_path, output_path="cleaned.png"):
    # 读取原图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（Otsu 自动阈值）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去除细线：形态学开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算：让字符更连贯
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2, iterations=1)

    # 连通域分析，过滤小面积噪声
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    mask = np.zeros_like(cleaned)

    for i in range(1, num_labels):  # 从1开始跳过背景
        x, y, w, h, area = stats[i]
        # 过滤小面积、非文字区域
        if area > 50 and h > 10 and w > 5:
            mask[labels == i] = 255

    # 保存结果
    cv2.imwrite(output_path, mask)
    print(f"处理完成，已保存到 {output_path}")

# 使用方法
clean_captcha("dataset/val/2SKZTH_0511.png", "captcha_cleaned.png")