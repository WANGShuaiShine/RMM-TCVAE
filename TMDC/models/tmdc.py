import torch
import torch.nn as nn
import torch.nn.functional as F
from models.msd import ModalitySpecificDenoising
from models.mcd import ModalityCommonDenoising
from models.imc import InterModalityComplementation

class TMDC(nn.Module):
    """两阶段模态去噪和补全框架（TMDC）
    
    该框架包括两个训练阶段：
    1. 模态内去噪阶段（IMD）：使用完整数据集训练，学习模态特定和模态不变表示
    2. 模态间补全阶段（IMC）：处理缺失模态的情况，利用可用模态补全缺失信息
    """
    def __init__(self, input_dims, hidden_dim, output_dim, num_heads=4, dropout=0.5, num_modalities=3):
        """初始化TMDC框架
        
        Args:
            input_dims: 各模态的输入特征维度列表，形状为 [num_modalities]
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
            num_modalities: 模态数量，默认为3（文本、音频、视频）
        """
        super(TMDC, self).__init__()
        
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # 1. 模态内去噪阶段（IMD）
        # 模态特定去噪模块（MSD），每个模态有自己的独立网络
        self.msd_modules = nn.ModuleList()
        for input_dim in input_dims:
            self.msd_modules.append(ModalitySpecificDenoising(input_dim, hidden_dim, output_dim, num_heads, dropout))
        
        # 模态公共去噪模块（MCD），所有模态共享参数
        # 注意：这里假设所有模态经过卷积后维度相同
        self.mcd_module = ModalityCommonDenoising(input_dims[0], hidden_dim, output_dim, num_heads, dropout)
        
        # 2. 模态间补全阶段（IMC）
        self.imc_module = InterModalityComplementation(hidden_dim, output_dim, num_heads, dropout)
    
    def forward_imd(self, x_list):
        """模态内去噪阶段前向传播
        
        Args:
            x_list: 各模态输入列表，每个元素形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            y_preds_s: 模态特定表示的预测标签列表
            y_preds_c: 模态不变表示的预测标签列表
            x_s_list: 去噪后的模态特定表示列表
            x_c_list: 去噪后的模态不变表示列表
            mus_s: 模态特定表示的均值列表
            logvars_s: 模态特定表示的对数方差列表
            mus_c: 模态不变表示的均值列表
            logvars_c: 模态不变表示的对数方差列表
        """
        y_preds_s = []
        y_preds_c = []
        x_s_list = []
        x_c_list = []
        mus_s = []
        logvars_s = []
        mus_c = []
        logvars_c = []
        
        # 处理每个模态
        for i in range(self.num_modalities):
            x = x_list[i]
            
            # 模态特定去噪
            y_pred_s, x_s, mu_s, logvar_s = self.msd_modules[i](x)
            y_preds_s.append(y_pred_s)
            x_s_list.append(x_s)
            mus_s.append(mu_s)
            logvars_s.append(logvar_s)
            
            # 模态公共去噪
            y_pred_c, x_c, mu_c, logvar_c = self.mcd_module(x)
            y_preds_c.append(y_pred_c)
            x_c_list.append(x_c)
            mus_c.append(mu_c)
            logvars_c.append(logvar_c)
        
        return y_preds_s, y_preds_c, x_s_list, x_c_list, mus_s, logvars_s, mus_c, logvars_c
    
    def forward_imc(self, x_list, missing_mask):
        """模态间补全阶段前向传播
        
        Args:
            x_list: 各模态输入列表，每个元素形状为 [batch_size, seq_len, input_dim]
            missing_mask: 缺失模态掩码，形状为 [num_modalities]，1表示缺失，0表示可用
            
        Returns:
            y_pred: 预测标签
            x_fused: 融合后的多模态表示
        """
        # 首先获取各模态的去噪表示
        _, _, x_s_list, x_c_list, _, _, _, _ = self.forward_imd(x_list)
        
        # 然后进行模态间补全和融合
        y_pred, x_fused = self.imc_module(x_s_list, x_c_list, missing_mask)
        
        return y_pred, x_fused
    
    def forward(self, x_list, stage='imd', missing_mask=None):
        """前向传播
        
        Args:
            x_list: 各模态输入列表，每个元素形状为 [batch_size, seq_len, input_dim]
            stage: 训练阶段，'imd' 或 'imc'
            missing_mask: 缺失模态掩码，仅在 'imc' 阶段使用
            
        Returns:
            根据阶段返回不同的结果
        """
        if stage == 'imd':
            return self.forward_imd(x_list)
        elif stage == 'imc':
            if missing_mask is None:
                # 默认所有模态可用
                missing_mask = torch.zeros(self.num_modalities, device=x_list[0].device)
            return self.forward_imc(x_list, missing_mask)
        else:
            raise ValueError(f"Invalid stage: {stage}, must be 'imd' or 'imc'")
    
    def get_loss_imd(self, y_preds_s, y_preds_c, mus_s, logvars_s, mus_c, logvars_c, y_true, beta=0.01):
        """计算模态内去噪阶段的损失
        
        Args:
            y_preds_s: 模态特定表示的预测标签列表
            y_preds_c: 模态不变表示的预测标签列表
            mus_s: 模态特定表示的均值列表
            logvars_s: 模态特定表示的对数方差列表
            mus_c: 模态不变表示的均值列表
            logvars_c: 模态不变表示的对数方差列表
            y_true: 真实标签
            beta: VIB损失的权重
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典，包含各部分损失
        """
        # 任务损失
        task_loss_s = 0.0
        task_loss_c = 0.0
        
        # VIB损失
        vib_loss_s = 0.0
        vib_loss_c = 0.0
        
        for i in range(self.num_modalities):
            # 任务损失
            task_loss_s += F.mse_loss(y_preds_s[i], y_true)
            task_loss_c += F.mse_loss(y_preds_c[i], y_true)
            
            # VIB损失（KL散度）
            # KL(p||q) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            vib_loss_s += -0.5 * torch.sum(1 + logvars_s[i] - mus_s[i].pow(2) - logvars_s[i].exp())
            vib_loss_c += -0.5 * torch.sum(1 + logvars_c[i] - mus_c[i].pow(2) - logvars_c[i].exp())
        
        # 平均损失
        task_loss_s /= self.num_modalities
        task_loss_c /= self.num_modalities
        vib_loss_s /= self.num_modalities
        vib_loss_c /= self.num_modalities
        
        # 总损失
        total_loss = task_loss_s + task_loss_c + beta * (vib_loss_s + vib_loss_c)
        
        loss_dict = {
            'total_loss': total_loss,
            'task_loss_s': task_loss_s,
            'task_loss_c': task_loss_c,
            'vib_loss_s': vib_loss_s,
            'vib_loss_c': vib_loss_c
        }
        
        return total_loss, loss_dict
    
    def get_loss_imc(self, y_pred, y_true):
        """计算模态间补全阶段的损失
        
        Args:
            y_pred: 预测标签
            y_true: 真实标签
            
        Returns:
            loss: 损失
        """
        return F.mse_loss(y_pred, y_true)
