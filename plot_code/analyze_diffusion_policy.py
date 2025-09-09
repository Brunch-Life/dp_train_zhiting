#!/usr/bin/env python3
"""
Diffusion Policy分析脚本
基于analy_reference.py，适配当前的Diffusion Policy训练代码
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# 添加当前项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入我们的组件
from data_adapter import DiffusionPolicyDataAdapter
from model_wrapper import DiffusionPolicyWrapper
from visualization_tools import DiffusionPolicyVisualizer

class DiffusionPolicyAnalyzer:
    """
    Diffusion Policy性能分析器
    分析训练后的模型在训练集和测试集上的性能
    """
    
    def __init__(self, 
                 data_root,
                 model_path,
                 output_dir,
                 num_seeds=500,
                 chunk_size=20,
                 batch_size=128,
                 max_train_batches=None,
                 max_test_batches=None):
        """
        Args:
            data_root: 数据根目录路径
            model_path: 模型检查点路径
            output_dir: 输出目录路径
            num_seeds: 数据集种子数量
            chunk_size: 训练时使用的chunk大小
            batch_size: 批次大小
            max_train_batches: 最大训练批次数，None表示使用所有数据
            max_test_batches: 最大测试批次数，None表示使用所有数据
        """
        self.data_root = Path(data_root)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_seeds = num_seeds
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_train_batches = max_train_batches
        self.max_test_batches = max_test_batches
        
        print(f"分析参数:")
        print(f"  数据根目录: {self.data_root}")
        print(f"  模型路径: {self.model_path}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  种子数量: {self.num_seeds}")
        print(f"  块大小: {self.chunk_size}")
        print(f"  批次大小: {self.batch_size}")
        
        # 初始化组件
        self._setup_data_adapter()
        self._setup_model_wrapper()
        self._setup_visualizer()
        
        # 数据存储
        self.train_data = None
        self.test_data = None
        self.train_predictions = None
        self.test_predictions = None
    
    def _setup_data_adapter(self):
        """设置数据适配器"""
        print("正在设置数据适配器...")
        
        # 数据配置（与训练脚本一致）
        camera_names = ["third", "wrist"]
        usages = ["obs"]
        
        data_config = {
            "data_roots": [str(self.data_root)],
            "num_seeds": self.num_seeds,
            "camera_names": camera_names,
            "usages": usages,
            "chunk_size": self.chunk_size,
            "train_batch_size": self.batch_size,
            "val_batch_size": self.batch_size,
            "ckpt_dir": str(self.output_dir)  # 用于复制归一化参数
        }
        
        # 归一化参数路径
        norm_stats_filename = f"norm_stats_1_epsnum_{self.num_seeds}.pkl"
        norm_stats_path = self.data_root / norm_stats_filename
        
        # 检查归一化参数文件是否存在
        if not norm_stats_path.exists():
            print(f"警告: 归一化参数文件不存在: {norm_stats_path}")
            print("将使用训练数据计算归一化参数...")
            norm_stats_path = None
        
        self.data_adapter = DiffusionPolicyDataAdapter(
            data_config=data_config,
            norm_stats_path=str(norm_stats_path) if norm_stats_path else None
        )
        
        print("数据适配器设置完成")
    
    def _setup_model_wrapper(self):
        """设置模型包装器"""
        print("正在设置模型包装器...")
        
        # 归一化参数路径
        norm_stats_filename = f"norm_stats_1_epsnum_{self.num_seeds}.pkl"
        norm_stats_path = self.output_dir / norm_stats_filename
        
        # 如果输出目录中没有归一化参数，使用原始路径
        if not norm_stats_path.exists():
            norm_stats_path = self.data_root / norm_stats_filename
        
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"找不到归一化参数文件: {norm_stats_path}")
        
        self.model_wrapper = DiffusionPolicyWrapper(
            model_path=str(self.model_path),
            norm_stats_path=str(norm_stats_path),
            device='cuda'
        )
        
        print("模型包装器设置完成")
        
        # 打印模型信息
        model_info = self.model_wrapper.get_model_info()
        print(f"模型信息:")
        print(f"  参数总数: {model_info['num_parameters']:,}")
        print(f"  可训练参数: {model_info['num_trainable_parameters']:,}")
        print(f"  设备: {model_info['device']}")
    
    def _setup_visualizer(self):
        """设置可视化工具"""
        self.visualizer = DiffusionPolicyVisualizer(self.output_dir)
        print("可视化工具设置完成")
    
    def load_data(self):
        """加载数据"""
        print("="*50)
        print("开始加载数据")
        print("="*50)
        
        start_time = time.time()
        
        # 加载训练数据
        print("正在加载训练数据...")
        self.train_data = self.data_adapter.get_train_data(
            max_batches=self.max_train_batches
        )
        
        # 加载测试数据（验证数据）
        print("正在加载测试数据...")
        self.test_data = self.data_adapter.get_val_data(
            max_batches=self.max_test_batches
        )
        
        load_time = time.time() - start_time
        print(f"数据加载完成，耗时: {load_time:.1f}s")
        
        print(f"训练数据: {len(self.train_data['actions'])} 样本")
        print(f"测试数据: {len(self.test_data['actions'])} 样本")
    
    def predict_actions(self):
        """生成模型预测"""
        print("="*50)
        print("开始模型预测")
        print("="*50)
        
        start_time = time.time()
        
        # 训练集预测
        if self.train_data is not None:
            print("正在预测训练集...")
            self.train_predictions = self.model_wrapper.predict_batch(
                images=self.train_data['images'],
                states=self.train_data['states'],
                batch_size=self.batch_size
            )
            print(f"训练集预测完成，形状: {self.train_predictions.shape}")
        
        # 测试集预测
        if self.test_data is not None:
            print("正在预测测试集...")
            self.test_predictions = self.model_wrapper.predict_batch(
                images=self.test_data['images'], 
                states=self.test_data['states'],
                batch_size=self.batch_size
            )
            print(f"测试集预测完成，形状: {self.test_predictions.shape}")
        
        predict_time = time.time() - start_time
        print(f"模型预测完成，耗时: {predict_time:.1f}s")
    
    def analyze_and_visualize(self):
        """分析结果并创建可视化"""
        print("="*50)
        print("开始分析和可视化")
        print("="*50)
        
        start_time = time.time()
        
        # 创建所有可视化图表
        metrics = self.visualizer.create_all_visualizations(
            train_data=self.train_data,
            test_data=self.test_data,
            train_predictions=self.train_predictions,
            test_predictions=self.test_predictions
        )
        
        viz_time = time.time() - start_time
        print(f"分析和可视化完成，耗时: {viz_time:.1f}s")
        
        return metrics
    
    def print_summary(self, metrics):
        """打印分析总结"""
        print("="*50)
        print("分析总结")
        print("="*50)
        
        if 'train' in metrics and 'test' in metrics:
            print(f"训练集样本数: {len(self.train_data['actions'])}")
            print(f"测试集样本数: {len(self.test_data['actions'])}")
            print()
            
            print("总体性能:")
            print(f"  训练集 MSE: {metrics['train']['mse_total']:.6f}")
            print(f"  测试集 MSE: {metrics['test']['mse_total']:.6f}")
            print(f"  训练集 MAE: {metrics['train']['mae_total']:.6f}")
            print(f"  测试集 MAE: {metrics['test']['mae_total']:.6f}")
            print()
            
            print("各维度 MSE:")
            action_labels = self.data_adapter.get_action_labels()
            for label in action_labels:
                train_mse = metrics['train']['mse_per_dim'][label]
                test_mse = metrics['test']['mse_per_dim'][label]
                print(f"  {label:>8}: 训练集 {train_mse:.6f}, 测试集 {test_mse:.6f}")
            
            print()
            print("各维度相关系数:")
            for label in action_labels:
                train_corr = metrics['train']['correlation_per_dim'][label]
                test_corr = metrics['test']['correlation_per_dim'][label]
                print(f"  {label:>8}: 训练集 {train_corr:.4f}, 测试集 {test_corr:.4f}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("开始Diffusion Policy性能分析")
        print("="*60)
        
        total_start_time = time.time()
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 模型预测
            self.predict_actions()
            
            # 3. 分析和可视化
            metrics = self.analyze_and_visualize()
            
            # 4. 打印总结
            self.print_summary(metrics)
            
            total_time = time.time() - total_start_time
            
            print("="*60)
            print("分析完成!")
            print(f"总耗时: {total_time:.1f}s")
            print(f"结果保存在: {self.output_dir}")
            print("="*60)
            
            return metrics
            
        except Exception as e:
            print(f"分析过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析Diffusion Policy训练后的模型性能')
    
    # 数据相关参数
    parser.add_argument('--data_root', type=str,
                       default="dataset/processed/nonstop_plate_wooden_new",
                       help='数据根目录路径')
    parser.add_argument('--num_seeds', type=int, default=500,
                       help='数据集种子数量')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str,
                       required=True,
                       help='模型检查点路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str,
                       default="analysis_results_dp",
                       help='输出目录路径')
    
    # 训练配置参数
    parser.add_argument('--chunk_size', type=int, default=20,
                       help='训练时使用的chunk大小')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='分析时的批次大小')
    
    # 数据量控制参数
    parser.add_argument('--max_train_batches', type=int, default=None,
                       help='最大训练批次数，None表示使用所有数据')
    parser.add_argument('--max_test_batches', type=int, default=None,
                       help='最大测试批次数，None表示使用所有数据')
    
    args = parser.parse_args()
    
    # 打印参数信息
    print("运行参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # 创建分析器
    analyzer = DiffusionPolicyAnalyzer(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches
    )
    
    # 运行分析
    metrics = analyzer.run_analysis()
    
    if metrics is not None:
        print("分析成功完成！")
        return 0
    else:
        print("分析失败！")
        return 1

if __name__ == "__main__":
    exit(main())







