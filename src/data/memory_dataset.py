import cv2
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
from .dataset import CaptchaDataset

class MemoryCaptchaDataset(CaptchaDataset):
    def __init__(self, data_dir, char_to_idx, captcha_length, transform=None):
        super().__init__(data_dir, char_to_idx, captcha_length, transform)
        
        # 预加载所有图片到内存
        self.images = {}
        self.logger = self._setup_logger()
        self._preload_images()
        
    def _setup_logger(self):
        """设置简单的日志记录"""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger
    
    def _preload_images(self):
        """预加载所有图片到内存"""
        self.logger.info(f"开始预加载 {len(self.image_files)} 张图片到内存...")
        
        for img_file in tqdm(self.image_files, desc="加载图片"):
            img_path = os.path.join(self.data_dir, img_file)
            # 读取并预处理图片
            image = Image.open(img_path).convert('RGB')
            processed_image = self.preprocess_image(image)
            # 存储到内存
            self.images[img_file] = processed_image
            
        total_memory = sum(img.size[0] * img.size[1] * 3 for img in self.images.values()) / (1024 * 1024)
        self.logger.info(f"图片加载完成! 预计内存占用: {total_memory:.2f} MB")
    
    def preprocess_image(self, image):
        """预处理图片"""
        # 将PIL图像转换为OpenCV格式
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
        # Otsu's 阈值处理
        _, binary_image = cv2.threshold(
            img_array, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # 反转图像
        processed = cv2.bitwise_not(binary_image)
        
        # 转回PIL格式
        processed_image = Image.fromarray(processed)
        
        # 转为RGB模式以匹配原始数据集格式
        return processed_image.convert('RGB')
    
    def __getitem__(self, idx):
        """从内存中获取预处理后的图片"""
        img_file = self.image_files[idx]
        
        # 直接从内存获取预处理后的图片
        image = self.images[img_file]
        
        if self.transform:
            image = self.transform(image)
            
        # 从文件名获取标签
        label = img_file[:-4]  # 移除.png后缀
        label_tensor = torch.zeros(self.captcha_length, dtype=torch.long)
        for i, char in enumerate(label):
            label_tensor[i] = self.char_to_idx[char]
            
        return image, label_tensor
    
    def __del__(self):
        """清理内存"""
        self.images.clear() 