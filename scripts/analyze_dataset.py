import os
from pathlib import Path
from collections import Counter
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

class DatasetAnalyzer:
    def __init__(self, config_path='config/config.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化计数器
        self.train_counter = Counter()
        self.val_counter = Counter()
        self.position_counter = {i: Counter() for i in range(self.config['captcha']['length'])}
        
    def analyze_directory(self, dir_path, counter):
        """分析指定目录中的字符分布"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"目录不存在: {dir_path}")
            return
            
        for file_name in tqdm(os.listdir(dir_path), desc=f"分析 {dir_path.name}"):
            if file_name.endswith('.png'):
                label = file_name[:-4]  # 移除.png后缀
                # 更新总体计数
                counter.update(label)
                # 更新位置计数
                for i, char in enumerate(label):
                    self.position_counter[i].update(char)
    
    def analyze(self):
        """分析整个数据集"""
        print("开始分析数据集...")
        
        # 分析训练集
        self.analyze_directory(self.config['data']['train_dir'], self.train_counter)
        
        # 分析验证集
        self.analyze_directory(self.config['data']['val_dir'], self.val_counter)
        
    def plot_distribution(self, counter, title, save_path=None):
        """绘制字符分布图"""
        plt.figure(figsize=(15, 5))
        chars = list(counter.keys())
        counts = list(counter.values())
        
        plt.bar(chars, counts)
        plt.title(title)
        plt.xlabel('字符')
        plt.ylabel('出现次数')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n=== 数据集统计信息 ===")
        
        # 获取所有出现的字符
        all_chars = set(self.train_counter.keys()) | set(self.val_counter.keys())
        all_chars_str = ''.join(sorted(all_chars))
        print(f"\n所有出现的字符 ({len(all_chars)}个):")
        print(f"'{all_chars_str}'")
        
        # 检查是否与配置文件中的字符集一致
        config_chars = set(self.config['captcha']['charset'])
        missing_chars = config_chars - all_chars
        extra_chars = all_chars - config_chars
        
        if missing_chars:
            print(f"\n配置文件中定义但未出现的字符: {''.join(sorted(missing_chars))}")
        if extra_chars:
            print(f"\n数据集中出现但未在配置文件中定义的字符: {''.join(sorted(extra_chars))}")
        
        # 训练集统计
        train_total = sum(self.train_counter.values())
        print(f"\n训练集总样本数: {train_total // self.config['captcha']['length']}")
        print("\n训练集字符分布:")
        for char, count in sorted(self.train_counter.items()):
            print(f"{char}: {count} ({count/train_total*100:.2f}%)")
        
        # 验证集统计
        val_total = sum(self.val_counter.values())
        print(f"\n验证集总样本数: {val_total // self.config['captcha']['length']}")
        print("\n验证集字符分布:")
        for char, count in sorted(self.val_counter.items()):
            print(f"{char}: {count} ({count/val_total*100:.2f}%)")
        
        # 位置统计
        print("\n各位置字符分布:")
        for pos, counter in self.position_counter.items():
            print(f"\n位置 {pos+1}:")
            total = sum(counter.values())
            for char, count in sorted(counter.items()):
                print(f"{char}: {count} ({count/total*100:.2f}%)")
    
    def save_report(self, output_path='dataset_analysis.txt'):
        """保存分析报告"""
        with open(output_path, 'w') as f:
            # 重定向print输出到文件
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            self.print_statistics()
            
            sys.stdout = original_stdout
        
        print(f"\n分析报告已保存至: {output_path}")
    
    def generate_visualizations(self, output_dir='analysis_plots'):
        """生成可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制训练集分布
        self.plot_distribution(
            self.train_counter,
            "训练集字符分布",
            output_dir / "train_distribution.png"
        )

        # 绘制验证集分布
        self.plot_distribution(
            self.val_counter,
            "验证集字符分布",
            output_dir / "val_distribution.png"
        )
        
        #  绘制每个位置的分布
        for pos, counter in self.position_counter.items():
            self.plot_distribution(
                counter,
                f"位置 {pos+1} 字符分布",
                output_dir / f"position_{pos+1}_distribution.png"
            )

def main():
    analyzer = DatasetAnalyzer()
    analyzer.analyze()
    analyzer.print_statistics()
    analyzer.save_report()
    analyzer.generate_visualizations()

if __name__ == '__main__':
    main() 