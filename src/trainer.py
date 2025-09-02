import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import resource

from .data.dataset import get_data_transforms
from .data.enhanced_dataset import EnhancedCaptchaDataset
from .models.cnn import LightCaptchaModel
from .models.advanced_models import CRNNModel, ResNetCaptcha, DenseNetCaptcha
from .utils.logger import setup_logger
from .utils.metrics import CaptchaMetrics

class Trainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.logger = setup_logger(__name__, "logs/training.log")
        
        # 更新设备检测逻辑
        device = self.config['training']['device'].lower()
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            # 直接使用配置文件指定的设备
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        self.setup()
        
    def setup(self):
        # 检查数据集目录
        train_path = Path(self.config['data']['train_dir'])
        val_path = Path(self.config['data']['val_dir'])
        # 测试集路径：优先读取配置，否则回退到默认目录
        test_dir_cfg = self.config['data'].get('test_dir', 'dataset/test')
        test_path = Path(test_dir_cfg)
        
        if not train_path.exists():
            self.logger.error(f"训练集目录不存在: {train_path}")
            raise FileNotFoundError(f"训练集目录不存在: {train_path}")
        if not val_path.exists():
            self.logger.error(f"验证集目录不存在: {val_path}")
            raise FileNotFoundError(f"验证集目录不存在: {val_path}")
        if not test_path.exists():
            self.logger.warning(f"测试集目录不存在(跳过测试): {test_path}")
        
        # 设置字符映射
        chars = self.config['captcha']['charset']
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        # 数据加载
        transform = get_data_transforms(self.config)
        train_dataset = EnhancedCaptchaDataset(
            train_path,
            self.char_to_idx,
            self.config['captcha']['length'],
            transform,
            preprocess_cfg=self.config.get('preprocessing', {})
        )
        val_dataset = EnhancedCaptchaDataset(
            val_path,
            self.char_to_idx,
            self.config['captcha']['length'],
            transform,
            preprocess_cfg=self.config.get('preprocessing', {})
        )
        # 可选：测试集
        test_dataset = None
        if test_path.exists():
            test_dataset = EnhancedCaptchaDataset(
                test_path,
                self.char_to_idx,
                self.config['captcha']['length'],
                transform,
                preprocess_cfg=self.config.get('preprocessing', {})
            )
        
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"验证集大小: {len(val_dataset)}")
        if test_dataset is not None:
            self.logger.info(f"测试集大小: {len(test_dataset)}")
        
        # 检查系统限制并调整num_workers
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        recommended_workers = min(self.config['data']['num_workers'], (hard_limit - 100) // 100)
        
        if recommended_workers != self.config['data']['num_workers']:
            self.logger.warning(
                f"由于系统文件描述符限制({soft_limit})，"
                f"num_workers已从{self.config['data']['num_workers']}调整为{recommended_workers}"
            )
            self.config['data']['num_workers'] = recommended_workers
        
        # 数据加载
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            persistent_workers=True if self.config['data']['num_workers'] > 0 else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            persistent_workers=True if self.config['data']['num_workers'] > 0 else False
        )
        # 测试集 DataLoader（如存在）
        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                persistent_workers=True if self.config['data']['num_workers'] > 0 else False
            )
        
        # 模型初始化
        model_type = self.config['model'].get('type', 'light')
        if model_type == 'crnn':
            self.model = CRNNModel(self.config)
        elif model_type == 'resnet':
            self.model = ResNetCaptcha(self.config)
        elif model_type == 'densenet':
            self.model = DenseNetCaptcha(self.config)
        else:
            self.model = LightCaptchaModel(self.config)
        
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-5
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 创建模型保存目录
        self.checkpoint_dir = Path(self.config['training']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def decode_seq(self, seq):
        """将索引序列解码为字符串"""
        return ''.join(self.idx_to_char[int(i)] for i in seq)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 确保优化器状态在正确的设备上
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        # 返回上次训练的epoch和最佳准确率
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def train_epoch(self, epoch):
        self.model.train()
        metrics = CaptchaMetrics(self.config)  # 使用指标类
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = 0
            position_losses = []  # 记录每个位置的损失
            
            for i in range(self.config['captcha']['length']):
                pos_loss = self.criterion(output[:, i], target[:, i])
                position_losses.append(pos_loss.item())
                loss += pos_loss
                
            loss = loss / self.config['captcha']['length']
            
            # 如果损失异常大，打印详细信息
            if loss.item() > 3:
                self.logger.warning(f"高损失警告 - Batch:")
                self.logger.warning(f"位置损失: {position_losses}")
                
            loss.backward()
            self.optimizer.step()
            
            # 更新指标
            pred = output.argmax(dim=2)
            metrics.update(pred.cpu(), target.cpu(), loss.cpu())
            
            # 获取当前指标
            current_metrics = metrics.get_metrics()
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_metrics["accuracy"]:.4f}',
                'avg_loss': f'{current_metrics["loss"]:.4f}'
            })
        
        return metrics.get_metrics()  # 返回完整的指标字典
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 收集样本用于打印
        correct_samples = []  # [(pred_str, label_str)]
        incorrect_samples = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = 0
                for i in range(self.config['captcha']['length']):
                    loss += self.criterion(output[:, i], target[:, i])
                loss = loss / self.config['captcha']['length']
                total_loss += loss.item()
                
                # 计算准确率
                pred = output.argmax(dim=2)
                correct += (pred == target).all(dim=1).sum().item()
                total += target.size(0)
                
                # 收集样本（仅收集到满足5个正确和5个错误为止，避免日志过多）
                need_more = len(correct_samples) < 5 or len(incorrect_samples) < 5
                if need_more:
                    pred_cpu = pred.detach().cpu()
                    tgt_cpu = target.detach().cpu()
                    match_mask = (pred_cpu == tgt_cpu).all(dim=1)
                    for j in range(tgt_cpu.size(0)):
                        if len(correct_samples) >= 5 and len(incorrect_samples) >= 5:
                            break
                        pred_str = self.decode_seq(pred_cpu[j])
                        label_str = self.decode_seq(tgt_cpu[j])
                        if match_mask[j].item():
                            if len(correct_samples) < 5:
                                correct_samples.append((pred_str, label_str))
                        else:
                            if len(incorrect_samples) < 5:
                                incorrect_samples.append((pred_str, label_str))
                
        # 打印样本到日志
        if correct_samples:
            self.logger.info("验证样本 - 正确预测(最多5个):")
            for p, l in correct_samples:
                self.logger.info(f"pred={p} | label={l}")
        if incorrect_samples:
            self.logger.info("验证样本 - 错误预测(最多5个):")
            for p, l in incorrect_samples:
                self.logger.info(f"pred={p} | label={l}")
                
        return total_loss / len(self.val_loader), correct / total
    
    def test(self):
        """在测试集上评估（若无测试集将直接返回None）。"""
        if self.test_loader is None:
            self.logger.warning("未加载测试集，跳过测试阶段。")
            return None, None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = 0
                for i in range(self.config['captcha']['length']):
                    loss += self.criterion(output[:, i], target[:, i])
                loss = loss / self.config['captcha']['length']
                total_loss += loss.item()
                pred = output.argmax(dim=2)
                correct += (pred == target).all(dim=1).sum().item()
                total += target.size(0)
        return total_loss / len(self.test_loader), correct / total
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 保存最新的checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            self.logger.info(f'保存最佳模型: {best_model_path}')
    
    def train(self, resume_path=None):
        self.logger.info("开始训练...")
        start_epoch = 0
        best_acc = 0
        
        # 如果提供了检查点路径，加载模型状态
        if resume_path:
            try:
                start_epoch, best_acc = self.load_checkpoint(resume_path)
                self.logger.info(f"从epoch {start_epoch}继续训练")
                self.logger.info(f"之前的最佳准确率: {best_acc:.4f}")
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}")
                self.logger.info("将从头开始训练")
                start_epoch = 0
                best_acc = 0
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)  # 现在返回的是字典
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            
            # 记录训练信息
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["training"]["epochs"]} - '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}'
            )
            
            # 保存模型
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            self.save_checkpoint(epoch + 1, val_acc, is_best)
            
            # 每 N 个 epoch 在测试集上评估一次（默认 10）
            test_interval = int(self.config.get('training', {}).get('test_interval', 10))
            if self.test_loader is not None and test_interval > 0 and (epoch + 1) % test_interval == 0:
                test_loss, test_acc = self.test()
                if test_loss is not None:
                    self.logger.info(
                        f'Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}'
                    )
            
            # 每5个epoch删除旧的checkpoint，只保留最新的
            if epoch > 0 and epoch % 5 == 0:
                old_checkpoint = self.checkpoint_dir / f'checkpoint_epoch_{epoch-4}.pth'
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
        
        # 训练结束，记录总用时
        total_time = time.time() - start_time
        self.logger.info(f'训练完成! 总用时: {total_time/60:.2f}分钟')
        self.logger.info(f'最佳验证准确率: {best_acc:.4f}')