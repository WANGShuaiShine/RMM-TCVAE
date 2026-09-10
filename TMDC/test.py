import torch
import argparse
from models.tmdc import TMDC

"""
TMDC模型测试脚本

该脚本用于测试TMDC模型的基本功能，包括模型初始化、前向传播等。
"""

def parse_args():
    """解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='测试TMDC模型')
    
    # 模型参数
    parser.add_argument('--input_dims', type=list, default=[1024, 512, 1024], help='各模态输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--num_modalities', type=int, default=3, help='模态数量')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=50, help='序列长度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    return args

def test_model(args):
    """测试模型
    
    Args:
        args: 参数
    """
    print("=== 测试TMDC模型 ===")
    
    # 初始化模型
    model = TMDC(
        input_dims=args.input_dims,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_modalities=args.num_modalities
    ).to(args.device)
    
    print(f"模型初始化成功，设备: {args.device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 生成测试数据
    print("\n=== 生成测试数据 ===")
    x_list = []
    for dim in args.input_dims:
        x = torch.randn(args.batch_size, args.seq_len, dim, device=args.device)
        x_list.append(x)
        print(f"模态 {len(x_list)} 形状: {x.shape}")
    
    # 测试IMD阶段
    print("\n=== 测试IMD阶段 ===")
    y_preds_s, y_preds_c, x_s_list, x_c_list, mus_s, logvars_s, mus_c, logvars_c = model(x_list, stage='imd')
    
    print(f"IMD阶段输出:")
    print(f"- 模态特定预测数量: {len(y_preds_s)}, 形状: {y_preds_s[0].shape}")
    print(f"- 模态不变预测数量: {len(y_preds_c)}, 形状: {y_preds_c[0].shape}")
    print(f"- 模态特定表示数量: {len(x_s_list)}, 形状: {x_s_list[0].shape}")
    print(f"- 模态不变表示数量: {len(x_c_list)}, 形状: {x_c_list[0].shape}")
    
    # 测试IMC阶段
    print("\n=== 测试IMC阶段 ===")
    
    # 测试所有模态可用的情况
    missing_mask_all = torch.zeros(args.num_modalities, device=args.device)
    y_pred_all, x_fused_all = model(x_list, stage='imc', missing_mask=missing_mask_all)
    print(f"所有模态可用时:")
    print(f"- 预测形状: {y_pred_all.shape}")
    print(f"- 融合表示形状: {x_fused_all.shape}")
    
    # 测试缺少一个模态的情况
    missing_mask_one = torch.zeros(args.num_modalities, device=args.device)
    missing_mask_one[0] = 1  # 缺少第一个模态
    y_pred_one, x_fused_one = model(x_list, stage='imc', missing_mask=missing_mask_one)
    print(f"缺少一个模态时:")
    print(f"- 预测形状: {y_pred_one.shape}")
    print(f"- 融合表示形状: {x_fused_one.shape}")
    
    # 测试缺少两个模态的情况
    missing_mask_two = torch.zeros(args.num_modalities, device=args.device)
    missing_mask_two[0] = 1  # 缺少第一个模态
    missing_mask_two[1] = 1  # 缺少第二个模态
    y_pred_two, x_fused_two = model(x_list, stage='imc', missing_mask=missing_mask_two)
    print(f"缺少两个模态时:")
    print(f"- 预测形状: {y_pred_two.shape}")
    print(f"- 融合表示形状: {x_fused_two.shape}")
    
    print("\n=== 测试完成 ===")
    print("TMDC模型功能正常！")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 测试模型
    test_model(args)

if __name__ == '__main__':
    main()
