import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, char_to_idx, captcha_length, transform=None):
        self.data_dir = data_dir
        self.char_to_idx = char_to_idx
        self.captcha_length = captcha_length
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = img_file[:-4]
        label_tensor = torch.zeros(self.captcha_length, dtype=torch.long)
        for i, char in enumerate(label):
            label_tensor[i] = self.char_to_idx[char]
            
        return image, label_tensor

def get_data_transforms(config):
    return transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]) 