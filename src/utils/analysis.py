def analyze_high_loss_samples(model, dataloader, device, threshold=2.0):
    """分析高损失样本"""
    model.eval()
    problem_cases = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 计算每个样本的损失
            for i in range(len(data)):
                sample_loss = 0
                for pos in range(target.size(1)):
                    pos_loss = F.cross_entropy(
                        output[i:i+1, pos], 
                        target[i:i+1, pos]
                    )
                    sample_loss += pos_loss.item()
                
                if sample_loss > threshold:
                    problem_cases.append({
                        'loss': sample_loss,
                        'target': target[i],
                        'prediction': output[i].argmax(dim=1)
                    })
    
    return problem_cases 