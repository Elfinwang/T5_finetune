import pandas as pd
import numpy as np

def print_qerror(task, preds_unnorm, labels_unnorm):
    """
    计算并打印 q-error 统计信息
    参数:
    task: 任务名称
    preds_unnorm: 预测值列表
    labels_unnorm: 真实值列表
    """
    qerror = []
    
    for i in range(len(preds_unnorm)):
        try:
            pred_val = float(preds_unnorm[i])
            label_val = float(labels_unnorm[i])
            
            # 确保预测值大于0
            if pred_val > 0 and label_val > 0:
                if pred_val > label_val:
                    qerror.append(pred_val / label_val)
                else:
                    qerror.append(label_val / pred_val)
        except (ValueError, TypeError):
            # 跳过无法转换为数值的数据
            continue
    
    # 打印统计信息，基于 #selectedCode 中的指标
    print("*"*50)
    print(task)
    print("Mean: {}".format(np.mean(qerror)))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
   
    print("*"*50)

# 读取 CSV 文件
file_path = "./test_result/test_t5_large_stats_ft_results_269000_1e-5.csv"

try:
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    
    # 检查文件是否包含 truth 和 prediction 列
    if 'truth' in data.columns and 'prediction' in data.columns:
        # 提取真实值和预测值
        labels_unnorm = data['truth'].values
        preds_unnorm = data['prediction'].values
        
        # 计算并打印 q-error 统计信息
        print_qerror(file_path, preds_unnorm, labels_unnorm)
        
    else:
        print("文件中没有找到 'truth' 或 'prediction' 列")
        print("文件列名:", data.columns.tolist())
        
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"处理文件时出错: {e}")