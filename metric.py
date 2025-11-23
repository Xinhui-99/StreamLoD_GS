import json
import os
from pathlib import Path

def process_metrics():
    """
    读取多个路径下的avg_metrics.json文件，计算指定属性的均值
    """
    
    # 定义要访问的路径列表
    paths = [
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/coffee_martini3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/cook_spinach3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/cut_roasted_beef3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/flame_salmon_13_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/flame_steak3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/sear_steak3_r0.0_llff6_eprest6",
    ]
    
    # 定义要提取的属性
    target_metrics = [
        "PSNR (Test)",
        "SSIM (Test)", 
        "LPIPS (Test)",
        "Size (MB)",
        "Frame time",
    ]
    
    # 初始化用于存储所有值的字典
    all_values = {metric: [] for metric in target_metrics}
    
    # 记录成功读取的文件
    successful_files = []
    failed_files = []
    
    # 循环访问每个路径
    for path in paths:
        json_file_path = os.path.join(path, "avg_metrics.json")
        
        try:
            # 读取JSON文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
            # 提取目标属性的值
            for metric in target_metrics:
                if metric in data:
                    all_values[metric].append(data[metric])
                else:
                    print(f"警告: 在文件 {json_file_path} 中找不到属性 '{metric}'")
            
            successful_files.append(json_file_path)
            print(f"成功读取: {json_file_path}")
            
        except FileNotFoundError:
            print(f"错误: 文件不存在 - {json_file_path}")
            failed_files.append(json_file_path)
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 - {json_file_path}: {e}")
            failed_files.append(json_file_path)
        except Exception as e:
            print(f"错误: 读取文件失败 - {json_file_path}: {e}")
            failed_files.append(json_file_path)
    
    # 计算均值
    average_metrics = {}
    for metric in target_metrics:
        if all_values[metric]:  # 如果有值
            average_metrics[metric] = sum(all_values[metric]) / len(all_values[metric])
        else:
            average_metrics[metric] = None
            print(f"警告: 无法计算 '{metric}' 的均值（没有有效数据）")
    
    # 添加统计信息
    result = {
        "average_metrics": average_metrics,
        "statistics": {
            "total_files": len(paths),
            "successful_reads": len(successful_files),
            "failed_reads": len(failed_files),
            "source_files": successful_files
        }
    }
    
    # 定义输出路径（你可以根据需要修改这个路径）
    output_path = "output_dynerf_llff6_eprest6.json"
    
    # 保存结果到JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\n结果已成功保存到: {output_path}")
        
        # 打印计算结果
        print("\n计算的平均值:")
        for metric, avg_value in average_metrics.items():
            if avg_value is not None:
                print(f"  {metric}: {avg_value:.4f}")
            else:
                print(f"  {metric}: N/A")
                
    except Exception as e:
        print(f"错误: 无法保存结果文件 - {e}")
        
    return result


def process_metrics_with_custom_output(output_file_path):
    """
    带自定义输出路径的版本
    
    Args:
        output_file_path: 输出JSON文件的完整路径
    """
    
    # 定义要访问的路径列表
    paths = [
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/coffee_martini3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/cook_spinach3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/cut_roasted_beef3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/flame_salmon_13_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/flame_steak3_r0.0_llff6_eprest6",
        "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp/output_sparse30/sear_steak3_r0.0_llff6_eprest6",
    ]
    
    # 定义要提取的属性
    target_metrics = [
        "PSNR (Test)",
        "SSIM (Test)", 
        "LPIPS (Test)",
        "Size (MB)",
        "Frame time",
    ]
    # 初始化用于存储所有值的字典
    all_values = {metric: [] for metric in target_metrics}
    
    # 循环访问每个路径
    for path in paths:
        json_file_path = os.path.join(path, "avg_metrics.json")
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
            # 提取目标属性的值
            for metric in target_metrics:
                if metric in data:
                    all_values[metric].append(data[metric])
                    
        except Exception as e:
            print(f"处理文件 {json_file_path} 时出错: {e}")
            continue
    
    # 计算均值（只保存均值）
    average_metrics = {}
    for metric in target_metrics:
        if all_values[metric]:
            average_metrics[metric] = sum(all_values[metric]) / len(all_values[metric])
    
    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(average_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file_path}")
    return average_metrics


if __name__ == "__main__":
    # 运行主函数
    process_metrics()
    
    # 如果需要指定特定的输出路径，可以使用：
    # process_metrics_with_custom_output("/your/custom/path/output_dynerf_exp4.json")