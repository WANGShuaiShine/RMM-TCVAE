import torch
import torch.optim as optim
import numpy as np
import argparse
import logging
from models.tmdc import TMDC

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='训练TMDC模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/', help='数据路径')
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'iemocap'], help='数据集名称')
    
    # 模型参数
    parser.add_argument('--input_dims', type=list, default=[1024, 512, 1024], help='各模态输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--num_modalities', type=int, default=3, help='模态数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--beta', type=float, default=0.01, help='VIB损失权重')
    parser.add_argument('--epochs_imd', type=int, default=80, help='IMD阶段训练轮数')
    parser.add_argument('--epochs_imc', type=int, default=100, help='IMC阶段训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    # 其他参数
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='模型保存路径')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')
    
    args = parser.parse_args()
    return args

def load_data(data_path, dataset, input_dims):
    """加载数据集
    
    Args:
        data_path: 数据路径
        dataset: 数据集名称
        input_dims: 各模态输入特征维度列表
        
    Returns:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
    """
    # 这里是一个示例，实际需要根据数据集格式实现
    logger.info(f'加载{dataset}数据集...')
    
    # 假设数据格式为：每个样本是一个元组 (x_list, y_true)
    # x_list 是各模态表示的列表，每个元素形状为 [seq_len, input_dim]
    # y_true 是真实标签
    
    # 生成随机数据作为示例
    def generate_random_data(num_samples, seq_len, input_dims):
        data = []
        for _ in range(num_samples):
            x_list = []
            for dim in input_dims:
                # 随机生成序列数据
                x = torch.randn(seq_len, dim)
                x_list.append(x)
            # 随机生成标签
            y = torch.randn(1)
            data.append((x_list, y))
        return data
    
    # 根据数据集设置不同的参数
    if dataset == 'mosi':
        num_train = 1600
        num_val = 200
        num_test = 400
        seq_len = 50
    elif dataset == 'mosei':
        num_train = 18000
        num_val = 2000
        num_test = 2856
        seq_len = 100
    elif dataset == 'iemocap':
        num_train = 4000
        num_val = 500
        num_test = 1031
        seq_len = 75
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    train_data = generate_random_data(num_train, seq_len, input_dims)
    val_data = generate_random_data(num_val, seq_len, input_dims)
    test_data = generate_random_data(num_test, seq_len, input_dims)
    
    logger.info(f'数据集加载完成: 训练集 {len(train_data)} 样本, 验证集 {len(val_data)} 样本, 测试集 {len(test_data)} 样本')
    
    return train_data, val_data, test_data

def create_dataloader(data, batch_size, shuffle=True):
    """创建数据加载器
    
    Args:
        data: 数据列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        dataloader: 数据加载器
    """
    # 自定义数据加载逻辑
    def collate_fn(batch):
        x_list_batch = []
        y_batch = []
        
        # 初始化各模态的列表
        num_modalities = len(batch[0][0])
        for i in range(num_modalities):
            x_list_batch.append([])
        
        # 处理每个样本
        for x_list, y in batch:
            for i in range(num_modalities):
                x_list_batch[i].append(x_list[i])
            y_batch.append(y)
        
        # 堆叠成批次
        for i in range(num_modalities):
            x_list_batch[i] = torch.stack(x_list_batch[i], dim=0)
        y_batch = torch.stack(y_batch, dim=0)
        
        return x_list_batch, y_batch
    
    # 创建数据集和数据加载器
    from torch.utils.data import Dataset, DataLoader
    
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return dataloader

def train_imd(model, train_loader, val_loader, optimizer, args):
    """训练模态内去噪阶段（IMD）
    
    Args:
        model: TMDC模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        args: 参数
        
    Returns:
        best_model: 最佳模型
    """
    logger.info('开始模态内去噪阶段（IMD）训练...')
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(args.epochs_imd):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x_list_batch, y_true) in enumerate(train_loader):
            # 移动到设备
            x_list_batch = [x.to(args.device) for x in x_list_batch]
            y_true = y_true.to(args.device)
            
            # 前向传播
            y_preds_s, y_preds_c, _, _, mus_s, logvars_s, mus_c, logvars_c = model(x_list_batch, stage='imd')
            
            # 计算损失
            loss, loss_dict = model.get_loss_imd(y_preds_s, y_preds_c, mus_s, logvars_s, mus_c, logvars_c, y_true, args.beta)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印日志
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f'IMD Epoch [{epoch+1}/{args.epochs_imd}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 平均训练损失
        train_loss /= len(train_loader)
        
        # 验证
        val_loss = evaluate_imd(model, val_loader, args)
        
        logger.info(f'IMD Epoch [{epoch+1}/{args.epochs_imd}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            torch.save(best_model, f'{args.save_path}/tmdc_imd_best.pth')
            logger.info(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
    
    logger.info('模态内去噪阶段（IMD）训练完成！')
    return best_model

def evaluate_imd(model, val_loader, args):
    """评估模态内去噪阶段（IMD）
    
    Args:
        model: TMDC模型
        val_loader: 验证数据加载器
        args: 参数
        
    Returns:
        val_loss: 验证损失
    """
    # 评估模式
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x_list_batch, y_true in val_loader:
            # 移动到设备
            x_list_batch = [x.to(args.device) for x in x_list_batch]
            y_true = y_true.to(args.device)
            
            # 前向传播
            y_preds_s, y_preds_c, _, _, mus_s, logvars_s, mus_c, logvars_c = model(x_list_batch, stage='imd')
            
            # 计算损失
            loss, _ = model.get_loss_imd(y_preds_s, y_preds_c, mus_s, logvars_s, mus_c, logvars_c, y_true, args.beta)
            
            val_loss += loss.item()
    
    # 平均验证损失
    val_loss /= len(val_loader)
    
    return val_loss

def train_imc(model, train_loader, val_loader, optimizer, args):
    """训练模态间补全阶段（IMC）
    
    Args:
        model: TMDC模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        args: 参数
        
    Returns:
        best_model: 最佳模型
    """
    logger.info('开始模态间补全阶段（IMC）训练...')
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(args.epochs_imc):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x_list_batch, y_true) in enumerate(train_loader):
            # 移动到设备
            x_list_batch = [x.to(args.device) for x in x_list_batch]
            y_true = y_true.to(args.device)
            
            # 随机生成缺失模态掩码
            # 这里简单实现：每个模态有25%的概率缺失
            missing_mask = torch.randint(0, 2, (args.num_modalities,), device=args.device, dtype=torch.float32)
            
            # 前向传播
            y_pred, _ = model(x_list_batch, stage='imc', missing_mask=missing_mask)
            
            # 计算损失
            loss = model.get_loss_imc(y_pred, y_true)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印日志
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f'IMC Epoch [{epoch+1}/{args.epochs_imc}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 平均训练损失
        train_loss /= len(train_loader)
        
        # 验证
        val_loss = evaluate_imc(model, val_loader, args)
        
        logger.info(f'IMC Epoch [{epoch+1}/{args.epochs_imc}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            torch.save(best_model, f'{args.save_path}/tmdc_imc_best.pth')
            logger.info(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
    
    logger.info('模态间补全阶段（IMC）训练完成！')
    return best_model

def evaluate_imc(model, val_loader, args):
    """评估模态间补全阶段（IMC）
    
    Args:
        model: TMDC模型
        val_loader: 验证数据加载器
        args: 参数
        
    Returns:
        val_loss: 验证损失
    """
    # 评估模式
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x_list_batch, y_true in val_loader:
            # 移动到设备
            x_list_batch = [x.to(args.device) for x in x_list_batch]
            y_true = y_true.to(args.device)
            
            # 随机生成缺失模态掩码
            missing_mask = torch.randint(0, 2, (args.num_modalities,), device=args.device, dtype=torch.float32)
            
            # 前向传播
            y_pred, _ = model(x_list_batch, stage='imc', missing_mask=missing_mask)
            
            # 计算损失
            loss = model.get_loss_imc(y_pred, y_true)
            
            val_loss += loss.item()
    
    # 平均验证损失
    val_loss /= len(val_loader)
    
    return val_loss

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    import os
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载数据
    train_data, val_data, test_data = load_data(args.data_path, args.dataset, args.input_dims)
    
    # 创建数据加载器
    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, args.batch_size, shuffle=False)
    
    # 初始化模型
    model = TMDC(
        input_dims=args.input_dims,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_modalities=args.num_modalities
    ).to(args.device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 1. 训练模态内去噪阶段（IMD）
    best_imd_model = train_imd(model, train_loader, val_loader, optimizer, args)
    
    # 加载最佳IMD模型
    model.load_state_dict(best_imd_model)
    
    # 2. 训练模态间补全阶段（IMC）
    best_imc_model = train_imc(model, train_loader, val_loader, optimizer, args)
    
    # 加载最佳IMC模型
    model.load_state_dict(best_imc_model)
    
    # 测试
    logger.info('开始测试...')
    test_loss = evaluate_imc(model, test_loader, args)
    logger.info(f'测试损失: {test_loss:.4f}')
    
    logger.info('所有训练和测试完成！')

if __name__ == '__main__':
    main()
