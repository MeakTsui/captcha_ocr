from src.trainer import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', default='checkpoints/best_model.pth', help='继续训练的模型检查点路径')
    # parser.add_argument('--resume', default=None, help='继续训练的模型检查点路径')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train(resume_path=args.resume)

if __name__ == '__main__':
    main()
