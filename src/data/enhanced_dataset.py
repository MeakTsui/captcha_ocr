import cv2
import numpy as np
import torch
import os
from PIL import Image
from .dataset import CaptchaDataset

class EnhancedCaptchaDataset(CaptchaDataset):
    def __init__(self, data_dir, char_to_idx, captcha_length, transform=None, preprocess_cfg: dict | None = None):
        super().__init__(data_dir, char_to_idx, captcha_length, transform)
        # 预处理配置（保持默认与原来一致）
        self.pre_cfg = preprocess_cfg or {}
        self.pre_cfg.setdefault('use_otsu', True)
        self.pre_cfg.setdefault('invert_back', True)
        # 形态学
        self.pre_cfg.setdefault('morph_open', {'enable': False, 'ksize': 2, 'iterations': 1})
        self.pre_cfg.setdefault('morph_close', {'enable': False, 'ksize': 3, 'iterations': 1})
        # 连通域过滤
        self.pre_cfg.setdefault('cc_filter', {'enable': False, 'min_area': 50, 'min_h': 10, 'min_w': 5})
        # 去线（HoughLinesP）
        self.pre_cfg.setdefault('hough_remove_lines', {
            'enable': False, 'threshold': 50, 'minLineLength': 30, 'maxLineGap': 10, 'thickness': 1
        })
        
    def preprocess_image(self, image):
        """预处理图片（可配置）。返回 RGB PIL 图像。"""
        # PIL -> numpy (RGB)
        rgb = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 1) Otsu 二值化（默认开启）
        if self.pre_cfg.get('use_otsu', True):
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # 不使用 Otsu 时，直接使用灰度的反相阈值简单化（尽量接近旧流程）
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 2) Hough 去线（可选）
        hough = self.pre_cfg.get('hough_remove_lines', {})
        if hough.get('enable', False):
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=int(hough.get('threshold', 50)),
                minLineLength=int(hough.get('minLineLength', 30)),
                maxLineGap=int(hough.get('maxLineGap', 10))
            )
            if lines is not None:
                # 在线条位置画白线到二值图（先前是黑底白字）
                thickness = int(hough.get('thickness', 1))
                for l in lines.reshape(-1, 4):
                    x1, y1, x2, y2 = map(int, l)
                    cv2.line(binary, (x1, y1), (x2, y2), color=255, thickness=thickness)

        # 3) 形态学开/闭（可选）
        mo = self.pre_cfg.get('morph_open', {})
        if mo.get('enable', False):
            k = int(mo.get('ksize', 2))
            it = int(mo.get('iterations', 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, k), max(1, k)))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=max(1, it))

        mc = self.pre_cfg.get('morph_close', {})
        if mc.get('enable', False):
            k = int(mc.get('ksize', 3))
            it = int(mc.get('iterations', 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, k), max(1, k)))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, it))

        # 4) 连通域过滤（可选）
        cc = self.pre_cfg.get('cc_filter', {})
        if cc.get('enable', False):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            mask = np.zeros_like(binary)
            min_area = int(cc.get('min_area', 50))
            min_h = int(cc.get('min_h', 10))
            min_w = int(cc.get('min_w', 5))
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area >= min_area and h >= min_h and w >= min_w:
                    mask[labels == i] = 255
            binary = mask

        # 5) 反相（保持与旧流程视觉一致）
        if self.pre_cfg.get('invert_back', True):
            processed = cv2.bitwise_not(binary)
        else:
            processed = binary

        # 返回 RGB
        processed_image = Image.fromarray(processed).convert('RGB')
        return processed_image
    
    def __getitem__(self, idx):
        """重写获取数据的方法"""
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # 添加预处理步骤
        # image = self.preprocess_image(image)
        
        if self.transform:
            image = self.transform(image)
            
        # 从文件名获取标签
        name = os.path.splitext(img_file)[0]
        parts = name.split('_')
        if len(parts) >= 2:
            label = parts[0]
        else:
            label = name
        # label = img_file[:-4]  # 移除.png后缀
        label_tensor = torch.zeros(self.captcha_length, dtype=torch.long)
        for i, char in enumerate(label):
            label_tensor[i] = self.char_to_idx[char]
            
        return image, label_tensor 