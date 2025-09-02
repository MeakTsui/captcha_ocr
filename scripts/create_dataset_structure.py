import os
from pathlib import Path

def create_dataset_structure():
    """创建数据集目录结构"""
    # 创建主目录
    dataset_root = Path("dataset")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    
    # 创建所需的目录
    for dir_path in [train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print("数据集目录结构已创建:")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    print("\n请将训练图片放入train目录，验证图片放入val目录")

if __name__ == "__main__":
    create_dataset_structure() 