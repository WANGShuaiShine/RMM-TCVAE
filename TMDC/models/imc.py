import torch
import torch.nn as nn
import torch.nn.functional as F

class InterModalityComplementation(nn.Module):
    """模态间补全阶段（IMC）
    
    该模块用于处理具有缺失模态的不完整数据。
    利用可用模态的表示来补全缺失模态的信息。
    """
    def __init__(self, hidden_dim, output_dim, num_heads=4, dropout=0.5):
        """初始化模态间补全模块
        
        Args:
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super(InterModalityComplementation, self).__init__()
        
        # 模态内注意力层，用于增强单模态表示
        self.intra_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 跨模态注意力层，用于建模可用模态之间的依赖关系
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 残差全连接层
        self.fc_residual = nn.Linear(hidden_dim, hidden_dim)
        
        # 最终融合层
        self.fc_fusion = nn.Linear(hidden_dim * 3, hidden_dim)  # 假设最多处理3个模态
        
        # 预测层
        self.fc_pred = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x_s_list, x_c_list, missing_mask):
        """前向传播
        
        Args:
            x_s_list: 模态特定表示列表，每个元素形状为 [batch_size, seq_len, hidden_dim]
            x_c_list: 模态不变表示列表，每个元素形状为 [batch_size, seq_len, hidden_dim]
            missing_mask: 缺失模态掩码，形状为 [num_modalities]，1表示缺失，0表示可用
            
        Returns:
            y_pred: 预测标签
            x_fused: 融合后的多模态表示
        """
        num_modalities = len(x_s_list)
        available_modalities = [i for i in range(num_modalities) if missing_mask[i] == 0]
        num_available = len(available_modalities)
        
        # 1. 增强单模态表示
        enhanced_reps = []
        for i in available_modalities:
            # 使用模态不变表示作为键和值，模态特定表示作为查询
            attn_output, _ = self.intra_attn(x_s_list[i], x_c_list[i], x_c_list[i])
            x_enhanced = x_s_list[i] + attn_output
            x_enhanced = self.layer_norm(x_enhanced)
            
            # 残差全连接
            x_residual = self.fc_residual(x_enhanced)
            x_residual = F.relu(x_residual)
            x_residual = self.dropout(x_residual)
            x_enhanced = x_enhanced + x_residual
            x_enhanced = self.layer_norm(x_enhanced)
            
            enhanced_reps.append(x_enhanced)
        
        # 2. 跨模态补全
        if num_available == 1:
            # 只有一个模态可用，使用自注意力进行补全
            available_idx = available_modalities[0]
            x_s = x_s_list[available_idx]
            x_c = x_c_list[available_idx]
            
            # 使用模态特定表示作为查询，模态不变表示作为键和值
            attn_output, _ = self.intra_attn(x_s, x_c, x_c)
            x_compensated = x_s + attn_output
            x_compensated = self.layer_norm(x_compensated)
            
            # 残差全连接
            x_residual = self.fc_residual(x_compensated)
            x_residual = F.relu(x_residual)
            x_residual = self.dropout(x_residual)
            x_compensated = x_compensated + x_residual
            x_compensated = self.layer_norm(x_compensated)
            
            # 重复补全后的表示以匹配模态数量
            fused_reps = [enhanced_reps[0]]
            for _ in range(num_modalities - 1):
                fused_reps.append(x_compensated)
        else:
            # 多个模态可用，使用双向注意力进行补全
            fused_reps = []
            
            # 首先添加所有可用模态的增强表示
            for i in range(num_modalities):
                if i in available_modalities:
                    idx = available_modalities.index(i)
                    fused_reps.append(enhanced_reps[idx])
                else:
                    fused_reps.append(None)
            
            # 对于每个缺失模态，使用所有可用模态进行补全
            for i in range(num_modalities):
                if missing_mask[i] == 1:
                    # 计算所有可用模态的加权和作为补全表示
                    compensated_list = []
                    for j in available_modalities:
                        # 使用缺失模态的模态不变表示作为查询，可用模态的模态特定表示作为键和值
                        # 注意：这里我们使用所有可用模态的平均作为查询
                        query = torch.mean(torch.stack([x_c_list[j] for j in available_modalities]), dim=0)
                        key = x_s_list[j]
                        value = x_s_list[j]
                        
                        attn_output, _ = self.cross_attn(query, key, value)
                        compensated_list.append(attn_output)
                    
                    # 平均所有补全表示
                    x_compensated = torch.mean(torch.stack(compensated_list), dim=0)
                    fused_reps[i] = x_compensated
        
        # 3. 融合多模态表示
        # 对每个模态的表示进行平均池化，得到 [batch_size, hidden_dim]
        pooled_reps = [torch.mean(rep, dim=1) for rep in fused_reps]
        
        # 拼接所有模态的表示
        x_concat = torch.cat(pooled_reps, dim=1)
        
        # 融合层
        x_fused = self.fc_fusion(x_concat)
        x_fused = F.relu(x_fused)
        x_fused = self.dropout(x_fused)
        
        # 预测
        y_pred = self.fc_pred(x_fused)
        
        return y_pred, x_fused
