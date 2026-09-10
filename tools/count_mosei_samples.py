import os
import h5py
import numpy as np
from mmsdk import mmdatasdk as md

def analyze_sample_structure(file_path, modality_name):
    """
    分析CSD文件中样本的数据结构，包括特征维度、时间戳等
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0, []
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 获取顶层键（通常是模态名称）
            root_key = list(f.keys())[0]
            print(f"文件顶层键: {root_key}")
            
            # 获取数据键
            data_keys = list(f[root_key]['data'].keys())
            count = len(data_keys)
            print(f"{modality_name} 模态样本总数: {count}")
            
            # 分析前3个样本的数据结构作为示例
            sample_structures = []
            for i, sample_key in enumerate(data_keys[:3]):
                print(f"\n示例样本 {i+1}/{min(3, count)}: {sample_key}")
                sample_data = f[root_key]['data'][sample_key]
                
                sample_info = {
                    'key': sample_key,
                    'components': {}
                }
                
                # 分析样本下的所有组件
                for component_name in sample_data.keys():
                    component_data = sample_data[component_name][()]
                    print(f"  组件 '{component_name}':")
                    print(f"    数据类型: {type(component_data).__name__}")
                    print(f"    形状: {component_data.shape}")
                    
                    # 对于数组类型，显示更多信息
                    if isinstance(component_data, np.ndarray):
                        print(f"    维度: {component_data.ndim}")
                        print(f"    数据类型: {component_data.dtype}")
                        # 如果维度较小，可以显示部分数据示例
                        if component_data.size <= 100:
                            print(f"    数据示例: {component_data}")
                        else:
                            print(f"    数据示例（前5个元素）: {component_data.flatten()[:5]}")
                    
                    sample_info['components'][component_name] = {
                        'shape': component_data.shape,
                        'dtype': str(component_data.dtype) if isinstance(component_data, np.ndarray) else type(component_data).__name__
                    }
                
                sample_structures.append(sample_info)
            
            return count, sample_structures
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0, []

def count_samples_in_csd(file_path, modality_name):
    """
    计算给定csd文件中的样本数量
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 获取顶层键（通常是模态名称）
            root_key = list(f.keys())[0]
            # 获取数据键
            data_keys = list(f[root_key]['data'].keys())
            count = len(data_keys)
            print(f"{modality_name} 模态样本数: {count}")
            return count
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")
        return 0

def analyze_mosei_sample_structures():
    """
    分析MOSEI数据集中各模态的样本结构
    """
    # 定义所有模态的文件路径
    modalities = {
        'language': './cmumosei/CMU_MOSEI_TimestampedWordVectors.csd',
        'visual': './cmumosei/CMU_MOSEI_VisualOpenFace2.csd', 
        'acoustic': './cmumosei/CMU_MOSEI_COVAREP.csd',
        'labels': './cmumosei/CMU_MOSEI_Labels.csd',
        'words': './cmumosei/CMU_MOSEI_TimestampedWords.csd',
        'phones': './cmumosei/CMU_MOSEI_TimestampedPhones.csd',
        'visual_facet': './cmumosei/CMU_MOSEI_VisualFacet42.csd'
    }
    
    # 确保使用绝对路径
    for key in modalities:
        modalities[key] = os.path.abspath(modalities[key])
    
    # 如果路径不存在，跳过该模态
    valid_modalities = {}
    for name, path in modalities.items():
        if os.path.exists(path):
            valid_modalities[name] = path
        else:
            print(f"警告: {name} 模态文件不存在: {path}")
    
    print("\n开始分析MOSEI数据集中各模态的样本结构...")
    
    modality_structures = {}
    
    for modality_name, file_path in valid_modalities.items():
        print(f"\n{'-' * 60}")
        print(f"分析 {modality_name.upper()} 模态 ({file_path})")
        print(f"{'-' * 60}")
        
        count, sample_structures = analyze_sample_structure(file_path, modality_name)
        modality_structures[modality_name] = {
            'count': count,
            'samples': sample_structures
        }
    
    print(f"\n{'-' * 60}")
    print("MOSEI数据集样本结构分析完成")
    print(f"{'-' * 60}")
    
    # 打印汇总信息
    print("\n各模态样本数量汇总:")
    for modality_name, info in modality_structures.items():
        print(f"{modality_name.upper()}: {info['count']} 样本")
    
    return modality_structures

def count_mosei_modalities():
    """
    计算MOSEI数据集中三种模态的样本数量
    """
    # 定义三种模态的文件路径
    modalities = {
        'language': './cmumosei/CMU_MOSEI_TimestampedWordVectors.csd',
        'visual': './cmumosei/CMU_MOSEI_VisualOpenFace2.csd', 
        'acoustic': './cmumosei/CMU_MOSEI_COVAREP.csd'
    }
    
    # 确保使用绝对路径
    for key in modalities:
        modalities[key] = os.path.abspath(modalities[key])
    
    counts = {}
    
    print("正在计算MOSEI数据集中各模态的样本数量...")
    print("=" * 50)
    
    # 计算每种模态的样本数
    for modality_name, file_path in modalities.items():
        if os.path.exists(file_path):
            count = count_samples_in_csd(file_path, modality_name)
            counts[modality_name] = count
        else:
            print(f"警告: {modality_name} 模态文件不存在: {file_path}")
    
    print("=" * 50)
    print("结果汇总:")
    for modality_name, count in counts.items():
        print(f"{modality_name.upper()} 模态样本数: {count}")
    
    return counts

def count_using_mmdatasdk():
    """
    使用mmdatasdk方法计算样本数
    """
    print("\n使用mmdatasdk方法计算样本数...")
    print("=" * 50)
    
    try:
        # 尝试从cmumosei目录加载数据集
        dataset = md.mmdataset('./cmumosei/')
        
        counts = {}
        
        # 语言模态
        if 'glove_vectors' in dataset.computational_sequences:
            language_count = len(dataset.computational_sequences['glove_vectors'].data.keys())
            counts['language'] = language_count
            print(f"language 模态样本数: {language_count}")
        
        # 视觉模态
        if 'OpenFace_2' in dataset.computational_sequences:
            visual_count = len(dataset.computational_sequences['OpenFace_2'].data.keys())
            counts['visual'] = visual_count
            print(f"visual 模态样本数: {visual_count}")
        
        # 声学模态
        if 'COVAREP' in dataset.computational_sequences:
            acoustic_count = len(dataset.computational_sequences['COVAREP'].data.keys())
            counts['acoustic'] = acoustic_count
            print(f"acoustic 模态样本数: {acoustic_count}")
            
        print("=" * 50)
        print("结果汇总:")
        for modality_name, count in counts.items():
            print(f"{modality_name.upper()} 模态样本数: {count}")
            
        return counts
    except Exception as e:
        print(f"使用mmdatasdk方法时出错: {e}")
        return {}

if __name__ == "__main__":
    print("CMU-MOSEI数据集样本结构分析工具")
    print("数据集路径:", os.path.abspath('./cmumosei/'))
    print()
    
    # 分析样本结构
    analyze_mosei_sample_structures()
    
    # 原有的计数功能保留但作为可选
    # print("\n" + "=" * 60)
    # print("额外信息: 各模态样本数量统计")
    # print("=" * 60)
    # counts1 = count_mosei_modalities()
