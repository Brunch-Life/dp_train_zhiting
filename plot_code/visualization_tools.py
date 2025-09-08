#!/usr/bin/env python3
"""
可视化工具
复用analy_reference.py的可视化函数，适配10维动作空间
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DiffusionPolicyVisualizer:
    """
    Diffusion Policy分析可视化工具
    基于analy_reference.py的可视化函数，适配10维动作空间
    """
    
    def __init__(self, output_dir):
        """
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 10维动作标签
        self.action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 
                             'rot_2', 'rot_3', 'rot_4', 'rot_5', 'gripper']
    
    def calculate_metrics(self, train_data, test_data, train_predictions, test_predictions):
        """
        计算各种评估指标
        Args:
            train_data: 训练数据字典
            test_data: 测试数据字典  
            train_predictions: 训练预测结果
            test_predictions: 测试预测结果
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 训练集指标
        if train_predictions is not None and 'actions' in train_data:
            train_true = train_data['actions'].numpy()
            train_pred = train_predictions
            
            train_metrics = {}
            # MSE 损失 (每个维度)
            train_mse = np.mean((train_true - train_pred) ** 2, axis=0)
            train_metrics['mse_per_dim'] = dict(zip(self.action_labels, train_mse))
            train_metrics['mse_total'] = np.mean(train_mse)
            
            # MAE 损失 (每个维度)
            train_mae = np.mean(np.abs(train_true - train_pred), axis=0)
            train_metrics['mae_per_dim'] = dict(zip(self.action_labels, train_mae))
            train_metrics['mae_total'] = np.mean(train_mae)
            
            # 相关系数 (每个维度)
            train_corr = [np.corrcoef(train_true[:, i], train_pred[:, i])[0, 1] 
                         for i in range(len(self.action_labels))]
            train_metrics['correlation_per_dim'] = dict(zip(self.action_labels, train_corr))
            
            metrics['train'] = train_metrics
        
        # 测试集指标
        if test_predictions is not None and 'actions' in test_data:
            test_true = test_data['actions'].numpy()
            test_pred = test_predictions
            
            test_metrics = {}
            # MSE 损失 (每个维度)
            test_mse = np.mean((test_true - test_pred) ** 2, axis=0)
            test_metrics['mse_per_dim'] = dict(zip(self.action_labels, test_mse))
            test_metrics['mse_total'] = np.mean(test_mse)
            
            # MAE 损失 (每个维度)
            test_mae = np.mean(np.abs(test_true - test_pred), axis=0)
            test_metrics['mae_per_dim'] = dict(zip(self.action_labels, test_mae))
            test_metrics['mae_total'] = np.mean(test_mae)
            
            # 相关系数 (每个维度)
            test_corr = [np.corrcoef(test_true[:, i], test_pred[:, i])[0, 1] 
                        for i in range(len(self.action_labels))]
            test_metrics['correlation_per_dim'] = dict(zip(self.action_labels, test_corr))
            
            metrics['test'] = test_metrics
        
        # 保存指标到文件
        with open(self.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        return metrics
    
    def plot_error_metrics(self, metrics):
        """绘制MSE和MAE对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE对比
        if 'train' in metrics and 'test' in metrics:
            train_mse = [metrics['train']['mse_per_dim'][label] for label in self.action_labels]
            test_mse = [metrics['test']['mse_per_dim'][label] for label in self.action_labels]
            
            x = np.arange(len(self.action_labels))
            width = 0.35
            
            ax1.bar(x - width/2, train_mse, width, label='Train', alpha=0.8)
            ax1.bar(x + width/2, test_mse, width, label='Test', alpha=0.8)
            ax1.set_xlabel('Action Dimensions')
            ax1.set_ylabel('MSE')
            ax1.set_title('Mean Squared Error (MSE) Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(self.action_labels, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # MAE对比
        if 'train' in metrics and 'test' in metrics:
            train_mae = [metrics['train']['mae_per_dim'][label] for label in self.action_labels]
            test_mae = [metrics['test']['mae_per_dim'][label] for label in self.action_labels]
            
            ax2.bar(x - width/2, train_mae, width, label='Train', alpha=0.8)
            ax2.bar(x + width/2, test_mae, width, label='Test', alpha=0.8)
            ax2.set_xlabel('Action Dimensions')
            ax2.set_ylabel('MAE')
            ax2.set_title('Mean Absolute Error (MAE) Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(self.action_labels, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_metrics(self, metrics):
        """绘制相关系数对比图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'train' in metrics and 'test' in metrics:
            train_corr = [metrics['train']['correlation_per_dim'][label] for label in self.action_labels]
            test_corr = [metrics['test']['correlation_per_dim'][label] for label in self.action_labels]
            
            x = np.arange(len(self.action_labels))
            width = 0.35
            
            ax.bar(x - width/2, train_corr, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, test_corr, width, label='Test', alpha=0.8)
            ax.set_xlabel('Action Dimensions')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Prediction vs Ground Truth Correlation Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(self.action_labels, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_scatter(self, train_data, test_data, train_predictions, test_predictions):
        """绘制预测vs真实值散点图"""
        # 需要更多子图来容纳10个维度
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, label in enumerate(self.action_labels):
            ax = axes[i]
            
            # 训练集散点图
            if train_predictions is not None and 'actions' in train_data:
                train_true = train_data['actions'].numpy()[:, i]
                train_pred = train_predictions[:, i]
                
                # 随机采样以避免图表过于密集
                n_samples = min(1000, len(train_true))
                indices = np.random.choice(len(train_true), n_samples, replace=False)
                
                ax.scatter(train_true[indices], train_pred[indices], 
                          alpha=0.6, s=10, label='Train', color='blue')
            
            # 测试集散点图
            if test_predictions is not None and 'actions' in test_data:
                test_true = test_data['actions'].numpy()[:, i]
                test_pred = test_predictions[:, i]
                
                # 随机采样
                n_samples = min(1000, len(test_true))
                indices = np.random.choice(len(test_true), n_samples, replace=False)
                
                ax.scatter(test_true[indices], test_pred[indices], 
                          alpha=0.6, s=10, label='Test', color='red')
            
            # 画对角线 (完美预测)
            all_values = []
            if train_predictions is not None and 'actions' in train_data:
                all_values.extend([train_true.min(), train_true.max()])
            if test_predictions is not None and 'actions' in test_data:
                all_values.extend([test_true.min(), test_true.max()])
            
            if all_values:
                min_val, max_val = min(all_values), max(all_values)
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
            
            ax.set_xlabel(f'Ground Truth - {label}')
            ax.set_ylabel(f'Prediction - {label}')
            ax.set_title(f'{label} Prediction Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.action_labels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series(self, train_data, test_data, train_predictions, test_predictions, n_samples=300):
        """绘制时间序列对比图"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, label in enumerate(self.action_labels):
            ax = axes[i]
            
            x = np.arange(n_samples)
            
            # 训练集时间序列
            if (train_predictions is not None and 'actions' in train_data and 
                len(train_data['actions']) >= n_samples):
                train_true = train_data['actions'].numpy()[:n_samples, i]
                train_pred = train_predictions[:n_samples, i]
                
                ax.plot(x, train_true, label='Train Ground Truth', alpha=0.8, linewidth=2)
                ax.plot(x, train_pred, label='Train Prediction', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'{label}')
            ax.set_title(f'{label} Time Series Comparison (First {n_samples} Samples)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.action_labels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self, train_data, test_data, train_predictions, test_predictions):
        """绘制误差分布直方图"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, label in enumerate(self.action_labels):
            ax = axes[i]
            
            # 训练集误差分布
            if train_predictions is not None and 'actions' in train_data:
                train_error = train_data['actions'].numpy()[:, i] - train_predictions[:, i]
                ax.hist(train_error, bins=50, alpha=0.6, label='Train Error', density=True)
            
            # 测试集误差分布
            if test_predictions is not None and 'actions' in test_data:
                test_error = test_data['actions'].numpy()[:, i] - test_predictions[:, i]
                ax.hist(test_error, bins=50, alpha=0.6, label='Test Error', density=True)
            
            ax.set_xlabel(f'Error - {label}')
            ax.set_ylabel('Density')
            ax.set_title(f'{label} Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.action_labels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_summary(self, metrics, train_data, test_data):
        """绘制总结图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总体MSE对比
        if 'train' in metrics and 'test' in metrics:
            categories = ['Train', 'Test']
            mse_values = [metrics['train']['mse_total'], metrics['test']['mse_total']]
            
            ax1.bar(categories, mse_values, color=['skyblue', 'lightcoral'])
            ax1.set_ylabel('MSE')
            ax1.set_title('Overall Mean Squared Error Comparison')
            ax1.grid(True, alpha=0.3)
            
            # 在柱状图上显示数值
            for i, v in enumerate(mse_values):
                ax1.text(i, v + max(mse_values) * 0.01, f'{v:.6f}', 
                        ha='center', va='bottom')
        
        # 2. 总体MAE对比
        if 'train' in metrics and 'test' in metrics:
            mae_values = [metrics['train']['mae_total'], metrics['test']['mae_total']]
            
            ax2.bar(categories, mae_values, color=['skyblue', 'lightcoral'])
            ax2.set_ylabel('MAE')
            ax2.set_title('Overall Mean Absolute Error Comparison')
            ax2.grid(True, alpha=0.3)
            
            # 在柱状图上显示数值
            for i, v in enumerate(mae_values):
                ax2.text(i, v + max(mae_values) * 0.01, f'{v:.6f}', 
                        ha='center', va='bottom')
        
        # 3. 数据集大小对比
        dataset_info = {
            'Train': len(train_data.get('actions', [])),
            'Test': len(test_data.get('actions', []))
        }
        
        ax3.bar(dataset_info.keys(), dataset_info.values(), color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Dataset Size')
        ax3.grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for i, (k, v) in enumerate(dataset_info.items()):
            ax3.text(i, v + max(dataset_info.values()) * 0.01, f'{v}', 
                    ha='center', va='bottom')
        
        # 4. 模型性能总结表
        ax4.axis('tight')
        ax4.axis('off')
        
        if 'train' in metrics and 'test' in metrics:
            summary_data = [
                ['Metric', 'Train', 'Test'],
                ['MSE', f"{metrics['train']['mse_total']:.6f}", f"{metrics['test']['mse_total']:.6f}"],
                ['MAE', f"{metrics['train']['mae_total']:.6f}", f"{metrics['test']['mae_total']:.6f}"],
                ['Samples', f"{len(train_data['actions'])}", f"{len(test_data['actions'])}"]
            ]
            
            table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            ax4.set_title('Model Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all_visualizations(self, train_data, test_data, train_predictions, test_predictions):
        """创建所有可视化图表"""
        print("正在创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        
        # 1. 计算指标
        metrics = self.calculate_metrics(train_data, test_data, train_predictions, test_predictions)
        
        # 2. MSE/MAE对比图
        self.plot_error_metrics(metrics)
        
        # 3. 相关系数对比图
        self.plot_correlation_metrics(metrics)
        
        # 4. 预测vs真实值散点图
        self.plot_prediction_scatter(train_data, test_data, train_predictions, test_predictions)
        
        # 5. 时间序列对比图 (前300个样本)
        self.plot_time_series(train_data, test_data, train_predictions, test_predictions)
        
        # 6. 误差分布直方图
        self.plot_error_distribution(train_data, test_data, train_predictions, test_predictions)
        
        # 7. 总结图
        self.plot_summary(metrics, train_data, test_data)
        
        print(f"所有图表已保存到: {self.output_dir}")
        return metrics





