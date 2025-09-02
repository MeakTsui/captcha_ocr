class CaptchaMetrics:
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.total_samples = 0
        self.correct_samples = 0
        self.position_correct = [0] * self.config['captcha']['length']
        self.char_correct = {char: 0 for char in self.config['captcha']['charset']}
        self.total_loss = 0
    
    def update(self, predictions, targets, loss):
        """更新指标"""
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        self.total_loss += loss.item() * batch_size
        
        # 计算整体准确率
        correct = (predictions == targets).all(dim=1)
        self.correct_samples += correct.sum().item()
        
        # 计算每个位置的准确率
        for i in range(self.config['captcha']['length']):
            self.position_correct[i] += (predictions[:, i] == targets[:, i]).sum().item()
        
        # 计算每个字符的准确率
        for i, char in enumerate(self.config['captcha']['charset']):
            mask = (targets == i)
            if mask.any():
                self.char_correct[char] += ((predictions == targets) & mask).sum().item()
    
    def get_metrics(self):
        """获取所有指标"""
        metrics = {
            'loss': self.total_loss / self.total_samples,
            'accuracy': self.correct_samples / self.total_samples,
            'position_accuracy': [
                correct / self.total_samples 
                for correct in self.position_correct
            ],
            'char_accuracy': {
                char: correct / self.total_samples 
                for char, correct in self.char_correct.items()
            }
        }
        return metrics 