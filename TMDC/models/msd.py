import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalitySpecificDenoising(nn.Module):
    """模态特定去噪模块（MSD）
    
    该模块用于减少每个模态内的噪声并提取模态特定表示。
    每个模态有自己独立的网络，参数不共享。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.5):
        """初始化模态特定去噪模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super(ModalitySpecificDenoising, self).__init__()
        
        # 1D时间卷积层，用于标准化模态维度和序列长度
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # VIB模块
        self.vib_mu = nn.Linear(hidden_dim, hidden_dim)
        self.vib_logvar = nn.Linear(hidden_dim, hidden_dim)
        
        # 多头注意力层
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 残差全连接层
        self.fc_residual = nn.Linear(hidden_dim, hidden_dim)
        
        # 预测层
        self.fc_pred = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入模态表示，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            y_pred: 预测标签
            x_denoised: 去噪后的模态表示
            mu: 均值
            logvar: 对数方差
        """
        # 1D卷积，标准化维度
        # 调整输入形状为 [batch_size, input_dim, seq_len] 以适应Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # 恢复形状为 [batch_size, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        
        # VIB模块
        mu = self.vib_mu(x)
        logvar = self.vib_logvar(x)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x_denoised = mu + eps * std
        
        # 多头注意力
        attn_output, _ = self.mha(x_denoised, x_denoised, x_denoised)
        x_attn = x_denoised + attn_output
        x_attn = self.layer_norm(x_attn)
        
        # 残差全连接
        x_residual = self.fc_residual(x_attn)
        x_residual = F.relu(x_residual)
        x_residual = self.dropout(x_residual)
        x_final = x_attn + x_residual
        x_final = self.layer_norm(x_final)
        
        # 预测
        y_pred = self.fc_pred(x_final.mean(dim=1))
        
        return y_pred, x_denoised, mu, logvar
