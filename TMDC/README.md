# TMDC: 两阶段模态去噪和补全框架

## 1. 项目介绍

TMDC（Two-stage Modality Denoising and Complementation）是一个用于处理多模态情感分析中缺失和噪声模态的框架。该框架包括两个训练阶段：

1. **模态内去噪阶段（IMD）**：从完整数据中学习去噪的模态特定和模态不变表示
2. **模态间补全阶段（IMC）**：利用可用模态来补全缺失模态的信息

## 2. 论文引用

如果您使用本代码，请引用以下论文：

```
@article{zhuang2025tmdc,
  title={TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis with Missing and Noisy Modalities},
  author={Zhuang, Yan and Liu, Minhao and Zhang, Yanru and Deng, Jiawen and Ren, Fuji},
  journal={arXiv preprint arXiv:2511.10325},
  year={2025}
}
```

## 3. 项目结构

```
TMDC/
├── models/             # 模型文件
│   ├── msd.py          # 模态特定去噪模块（MSD）
│   ├── mcd.py          # 模态公共去噪模块（MCD）
│   ├── imc.py          # 模态间补全模块（IMC）
│   └── tmdc.py         # 完整的TMDC框架
├── utils/              # 工具函数
├── data/               # 数据集目录
├── checkpoints/        # 模型保存目录
├── train.py            # 训练脚本
├── test.py             # 测试脚本
└── README.md           # 项目说明
```

## 4. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy
pip install argparse
pip install logging
```

## 5. 训练模型

### 5.1 模态内去噪阶段（IMD）

```bash
python train.py --stage imd --dataset mosi --epochs_imd 80
```

### 5.2 模态间补全阶段（IMC）

```bash
python train.py --stage imc --dataset mosi --epochs_imc 100
```

### 5.3 完整训练

```bash
python train.py --dataset mosi
```

## 6. 测试模型

```bash
python test.py
```

## 7. 模型参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --input_dims | 各模态输入特征维度 | [1024, 512, 1024] |
| --hidden_dim | 隐藏层维度 | 256 |
| --output_dim | 输出维度 | 1 |
| --num_heads | 多头注意力头数 | 4 |
| --dropout | Dropout概率 | 0.5 |
| --num_modalities | 模态数量 | 3 |
| --batch_size | 批次大小 | 32 |
| --lr | 学习率 | 1e-4 |
| --beta | VIB损失权重 | 0.01 |
| --epochs_imd | IMD阶段训练轮数 | 80 |
| --epochs_imc | IMC阶段训练轮数 | 100 |
| --device | 设备 | cuda（如果可用） |

## 8. 数据集支持

当前支持以下数据集：

- MOSI（Multimodal Opinion Sentiment Intensity）
- MOSEI（Multimodal Opinion Sentiment and Emotion Intensity）
- IEMOCAP（Interactive Emotional Dyadic Motion Capture）

## 9. 代码说明

### 9.1 模态特定去噪模块（MSD）

该模块用于减少每个模态内的噪声并提取模态特定表示。每个模态有自己独立的网络，参数不共享。

### 9.2 模态公共去噪模块（MCD）

该模块用于提取跨所有模态共享的去噪、模态不变特征。Conv1D、VIB和Attention层的参数是共享的。

### 9.3 模态间补全模块（IMC）

该模块用于处理具有缺失模态的不完整数据，利用可用模态的表示来补全缺失模态的信息。

### 9.4 完整的TMDC框架

整合了上述三个模块，实现了两个训练阶段的逻辑。

## 10. 注意事项

1. 本代码仅实现了论文中的核心算法，实际应用时可能需要根据具体数据集进行调整
2. 数据加载部分需要根据实际数据集格式进行修改
3. 建议在GPU上运行代码以获得更好的性能

## 11. 联系方式

如果您有任何问题或建议，请联系：

- 作者：Yan Zhuang
- 邮箱：202211081370@std.uestc.edu.cn
