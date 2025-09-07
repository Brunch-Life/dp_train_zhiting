#!/usr/bin/env python3
"""
分析SFT训练后的模型性能与数据集的对比
将训练集和测试集分开统计和可视化
"""

import os
import sys
import glob
import time
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import matplotlib.pyplot as plt
# import seaborn as sns  # 注释掉，使用基本matplotlib
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

# 添加路径以导入 SimplerEnv 模块
sys.path.append('/mnt/public/chenyinuo/RL4VLA/SimplerEnv')
from simpler_env.policies.MLP.MLP_train import MLPPolicy

# 导入角度后处理工具
from angle_utils import postprocess_action_for_env

# 配置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局函数，用于多进程
def _load_single_file_global(file_path, image_size=(480, 640)):
    """
    multi-process safe single file loading function
    Args:
        file_path: NPZ path
        image_size: target image size (H, W)
    Returns:
        tuple: (processed_images, actions, states) or None
    """
    try:
        data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
        
        # parse action data
        position = data["action"]["end"]["position"].squeeze(1)  # (T, 3)
        orientation_quat = data["action"]["end"]["orientation"].squeeze(1)  # (T, 4) quaternion
        gripper = data["action"]["effector"]["position_gripper"]  # (T, 1)
        
        # Convert quaternion to euler angles
        # orientation_quat is in (x, y, z, w) format, scipy expects (x, y, z, w)
        euler_angles = R.from_quat(orientation_quat).as_euler('xyz', degrees=False)  # (T, 3)
        
        # Concatenate: position (3) + euler (3) + gripper (1) = 7D action
        actions = np.concatenate([
            position,      # (T, 3)
            euler_angles,  # (T, 3) 
            gripper        # (T, 1)
        ], axis=1).astype(np.float32)  # (T, 7)
        
        # parse state data (current robot state)
        state_position = data["state"]["end"]["position"].squeeze(1)  # (T, 3)
        state_orientation_quat = data["state"]["end"]["orientation"].squeeze(1)  # (T, 4)
        state_gripper = data["state"]["effector"]["position_gripper"]  # (T, 1)
        
        # Concatenate: position (3) + quaternion (4) + gripper (1) = 8D state  
        states = np.concatenate([
            state_position,        # (T, 3)
            state_orientation_quat, # (T, 4)
            state_gripper          # (T, 1)
        ], axis=1).astype(np.float32)  # (T, 8)
        
        # parse image data
        raw_images = data["observation"]["rgb"]
        
        # preprocess images (resize)
        processed_images = []
        for img in raw_images:
            img_array = np.asarray(img)
            
            # ensure image is uint8 format and has 3 channels
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                if img_array.shape[:2] != image_size:
                    # use PIL to resize
                    from PIL import Image
                    img_array = np.array(Image.fromarray(img_array).resize(
                        (image_size[1], image_size[0])  # PIL use (width, height)
                    ))
                processed_images.append(img_array)
            else:
                print(f"Warning: Skipping invalid image shape: {img_array.shape}")
                continue
        
        if not processed_images:
            return None
            
        processed_images = np.stack(processed_images, axis=0)  # (N, H, W, C)
        
        # 确保数据长度一致
        min_len = min(len(actions), len(processed_images), len(states))
        actions = actions[:min_len]
        processed_images = processed_images[:min_len]
        states = states[:min_len]
        
        return processed_images, actions, states
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# MockArgs 类定义在全局级别，匹配训练时的模型配置
class MockArgs:
    def __init__(self, mlp_embedding_size=512, alg_lr=1e-4, action_dim=7, use_state=True):
        self.mlp_embedding_size = mlp_embedding_size  # 512维嵌入，匹配训练时配置
        self.alg_lr = alg_lr
        self.action_dim = action_dim
        self.use_state = use_state  # 是否使用状态信息，训练时使用了这个参数

class SFTAnalyzer:
    def __init__(self, 
                 data_dir="/home/chenyinuo/data/dataset/bingwen/data_for_success/green_bell_pepper_plate_wooden/success",
                 model_path="/mnt/public/chenyinuo/RL4VLA/runs/mlp_sft_steps_50000/step_010000",
                 output_dir="/mnt/public/chenyinuo/RL4VLA/analysis_results",
                 max_files=None,
                 debug=False):
        
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_files = max_files  # 最大文件数量，None表示使用所有文件
        self.debug = debug
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 数据存储
        self.train_data = {}
        self.test_data = {}
        self.train_predictions = {}
        self.test_predictions = {}
        
    def _load_model(self):
        """加载训练好的MLP模型"""
        print("正在加载模型...")
        
        # 使用全局的MockArgs类
        args = MockArgs()
        model = MLPPolicy(args, self.device)
        model.load(self.model_path)
        model.eval()
        
        print("模型加载完成!")
        return model
    
    def _load_single_file(self, file_path, image_size=(480, 640)):
        """加载单个NPZ文件"""
        try:
            data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
            
            # 解析动作数据
            position = data["action"]["end"]["position"].squeeze(1)  # (T, 3)
            orientation_quat = data["action"]["end"]["orientation"].squeeze(1)  # (T, 4) 
            gripper = data["action"]["effector"]["position_gripper"]  # (T, 1)
            
            # 将四元数转换为欧拉角
            euler_angles = R.from_quat(orientation_quat).as_euler('xyz', degrees=False)  # (T, 3)
            
            # 拼接: position (3) + euler (3) + gripper (1) = 7D action
            actions = np.concatenate([
                position,      # (T, 3)
                euler_angles,  # (T, 3) 
                gripper        # (T, 1)
            ], axis=1).astype(np.float32)  # (T, 7)
            
            # 解析状态数据 (当前机器人状态)
            state_position = data["state"]["end"]["position"].squeeze(1)  # (T, 3)
            state_orientation_quat = data["state"]["end"]["orientation"].squeeze(1)  # (T, 4)
            state_gripper = data["state"]["effector"]["position_gripper"]  # (T, 1)
            
            # 拼接: position (3) + quaternion (4) + gripper (1) = 8D state  
            states = np.concatenate([
                state_position,        # (T, 3)
                state_orientation_quat, # (T, 4)
                state_gripper          # (T, 1)
            ], axis=1).astype(np.float32)  # (T, 8)
            
            # 解析图像数据
            raw_images = data["observation"]["rgb"]
            
            # 预处理图像 (调整大小)
            processed_images = []
            for img in raw_images:
                img_array = np.asarray(img)
                
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    if img_array.shape[:2] != image_size:
                        img_array = np.array(Image.fromarray(img_array).resize(
                            (image_size[1], image_size[0])  # PIL 使用 (width, height)
                        ))
                    processed_images.append(img_array)
                else:
                    print(f"警告: 跳过无效图像形状: {img_array.shape}")
                    continue
            
            if not processed_images:
                return None
                
            processed_images = np.stack(processed_images, axis=0)  # (N, H, W, C)
            
            # 确保数据长度一致
            min_len = min(len(actions), len(processed_images), len(states))
            actions = actions[:min_len]
            processed_images = processed_images[:min_len]
            states = states[:min_len]
            
            return processed_images, actions, states
            
        except Exception as e:
            print(f"加载文件错误 {file_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load and split dataset (multithreaded version)"""
        print("Loading dataset...")
        
        # Find all NPZ files
        npz_files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        print(f"Found {len(npz_files)} NPZ files in {self.data_dir}")
        
        if len(npz_files) == 0:
            raise ValueError(f"No NPZ files found in {self.data_dir}")
        
        # 如果设置了最大文件数量，则随机选取
        if self.max_files is not None and self.max_files < len(npz_files):
            print(f"随机选取 {self.max_files} 个文件进行分析...")
            random.seed(42)  # 设置随机种子以确保结果可重现
            npz_files = random.sample(npz_files, self.max_files)
            npz_files = sorted(npz_files)  # 重新排序以保持一致性
            print(f"已选取 {len(npz_files)} 个文件")
        
        # 80/20 split
        split_idx = int(0.8 * len(npz_files))
        train_files = npz_files[:split_idx]
        test_files = npz_files[split_idx:]
        
        print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
        
        # Get CPU core count
        num_workers = min(mp.cpu_count(), 16)  # Limit to max 16 processes
        print(f"Using {num_workers} processes for parallel loading")
        
        # Multithreaded loading of training data
        print("Loading training data in parallel...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all training file tasks
            future_to_file = {
                executor.submit(_load_single_file_global, file_path): file_path 
                for file_path in train_files
            }
            
            train_images, train_actions, train_states = [], [], []
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        images, actions, states = result
                        train_images.append(images)
                        train_actions.append(actions)
                        train_states.append(states)
                        print(f"  Loaded train file {i+1}/{len(train_files)}: {len(actions)} samples")
                except Exception as e:
                    print(f"Warning: Failed to load train file {file_path}: {e}")
                    continue
        
        if train_images:
            print("Merging training data...")
            self.train_data = {
                'images': np.concatenate(train_images, axis=0),
                'actions': np.concatenate(train_actions, axis=0),
                'states': np.concatenate(train_states, axis=0)
            }
            print(f"Train data shape: images {self.train_data['images'].shape}, actions {self.train_data['actions'].shape}, states {self.train_data['states'].shape}")
        
        train_load_time = time.time() - start_time
        print(f"Training data load time: {train_load_time:.1f}s")
        
        # Multithreaded loading of test data
        print("Loading test data in parallel...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all test file tasks
            future_to_file = {
                executor.submit(_load_single_file_global, file_path): file_path 
                for file_path in test_files
            }
            
            test_images, test_actions, test_states = [], [], []
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        images, actions, states = result
                        test_images.append(images)
                        test_actions.append(actions)
                        test_states.append(states)
                        print(f"  Loaded test file {i+1}/{len(test_files)}: {len(actions)} samples")
                except Exception as e:
                    print(f"Warning: Failed to load test file {file_path}: {e}")
                    continue
        
        if test_images:
            print("Merging test data...")
            self.test_data = {
                'images': np.concatenate(test_images, axis=0),
                'actions': np.concatenate(test_actions, axis=0),
                'states': np.concatenate(test_states, axis=0)
            }
            print(f"Test data shape: images {self.test_data['images'].shape}, actions {self.test_data['actions'].shape}, states {self.test_data['states'].shape}")
        
        test_load_time = time.time() - start_time
        print(f"Test data load time: {test_load_time:.1f}s")
        print(f"Total load time: {train_load_time + test_load_time:.1f}s")
    
    def _preprocess_images_for_model(self, images):
        """为模型预处理图像"""
        # 转换为tensor并归一化
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        images = images.float() / 255.0  # 归一化到 [0,1]
        
        # 转换格式: BHWC -> BCHW
        if len(images.shape) == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        return images.to(self.device)
    
    def predict_actions(self, batch_size=32):
        """Predict actions for training and test sets"""
        print("Making model predictions...")
        
        with torch.no_grad():
            # Training set predictions
            if 'images' in self.train_data:
                train_preds = []
                train_values = []
                
                num_train = len(self.train_data['images'])
                for i in tqdm(range(0, num_train, batch_size), desc="Train prediction"):
                    end_idx = min(i + batch_size, num_train)
                    batch_images = self.train_data['images'][i:end_idx]
                    batch_states = self.train_data['states'][i:end_idx]
                    
                    # debug: load image from debug_image.png
                    if self.debug:
                        from PIL import Image
                        import os
                        
                        debug_image_path = "/mnt/public/chenyinuo/RL4VLA/SimplerEnv/debug_image.png"
                        if os.path.exists(debug_image_path):
                            # 加载 debug 图像
                            debug_img = Image.open(debug_image_path)
                            debug_img_array = np.array(debug_img)
                            
                            # 复制 debug 图像来匹配当前 batch 的大小
                            current_batch_size = len(batch_images)
                            batch_images = [debug_img_array for _ in range(current_batch_size)]
                            batch_images = np.stack(batch_images, axis=0)
                            print(f"Debug mode: Using debug_image.png replicated {current_batch_size} times")
                        else:
                            print(f"Warning: {debug_image_path} not found, using original batch_images")

                    processed_images = self._preprocess_images_for_model(batch_images)
                    processed_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)

                    obs = {"image": processed_images, "state": processed_states}
                    
                    value, action, _ = self.model.get_action(obs, deterministic=True)
                    # 对预测结果进行后处理，从训练空间转换回原始角度空间
                    action_postprocessed = postprocess_action_for_env(action)
                    train_preds.append(action_postprocessed.cpu().numpy())
                    train_values.append(value.cpu().numpy())
                
                self.train_predictions = {
                    'actions': np.concatenate(train_preds, axis=0),
                    'values': np.concatenate(train_values, axis=0)
                }
                print(f"Train prediction completed: {self.train_predictions['actions'].shape}")
            
            # Test set predictions
            if 'images' in self.test_data:
                test_preds = []
                test_values = []
                
                num_test = len(self.test_data['images'])
                for i in tqdm(range(0, num_test, batch_size), desc="Test prediction"):
                    end_idx = min(i + batch_size, num_test)
                    batch_images = self.test_data['images'][i:end_idx]
                    batch_states = self.test_data['states'][i:end_idx]
                    
                    processed_images = self._preprocess_images_for_model(batch_images)
                    processed_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
                    
                    obs = {"image": processed_images, "state": processed_states}
                    
                    value, action, _ = self.model.get_action(obs, deterministic=True)
                    # 对预测结果进行后处理，从训练空间转换回原始角度空间
                    action_postprocessed = postprocess_action_for_env(action)
                    test_preds.append(action_postprocessed.cpu().numpy())
                    test_values.append(value.cpu().numpy())
                
                self.test_predictions = {
                    'actions': np.concatenate(test_preds, axis=0),
                    'values': np.concatenate(test_values, axis=0)
                }
                print(f"Test prediction completed: {self.test_predictions['actions'].shape}")
    
    def calculate_metrics(self):
        """计算各种评估指标"""
        metrics = {}
        
        # 动作维度标签
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        # 训练集指标
        if 'actions' in self.train_predictions:
            train_true = self.train_data['actions']
            train_pred = self.train_predictions['actions']
            
            train_metrics = {}
            # MSE 损失 (每个维度)
            train_mse = np.mean((train_true - train_pred) ** 2, axis=0)
            train_metrics['mse_per_dim'] = dict(zip(action_labels, train_mse))
            train_metrics['mse_total'] = np.mean(train_mse)
            
            # MAE 损失 (每个维度)
            train_mae = np.mean(np.abs(train_true - train_pred), axis=0)
            train_metrics['mae_per_dim'] = dict(zip(action_labels, train_mae))
            train_metrics['mae_total'] = np.mean(train_mae)
            
            # 相关系数 (每个维度)
            train_corr = [np.corrcoef(train_true[:, i], train_pred[:, i])[0, 1] 
                         for i in range(len(action_labels))]
            train_metrics['correlation_per_dim'] = dict(zip(action_labels, train_corr))
            
            metrics['train'] = train_metrics
        
        # 测试集指标
        if 'actions' in self.test_predictions:
            test_true = self.test_data['actions']
            test_pred = self.test_predictions['actions']
            
            test_metrics = {}
            # MSE 损失 (每个维度)
            test_mse = np.mean((test_true - test_pred) ** 2, axis=0)
            test_metrics['mse_per_dim'] = dict(zip(action_labels, test_mse))
            test_metrics['mse_total'] = np.mean(test_mse)
            
            # MAE 损失 (每个维度)
            test_mae = np.mean(np.abs(test_true - test_pred), axis=0)
            test_metrics['mae_per_dim'] = dict(zip(action_labels, test_mae))
            test_metrics['mae_total'] = np.mean(test_mae)
            
            # 相关系数 (每个维度)
            test_corr = [np.corrcoef(test_true[:, i], test_pred[:, i])[0, 1] 
                        for i in range(len(action_labels))]
            test_metrics['correlation_per_dim'] = dict(zip(action_labels, test_corr))
            
            metrics['test'] = test_metrics
        
        self.metrics = metrics
        
        # 保存指标到文件
        with open(self.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        return metrics
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("Creating visualization charts...")
        
        # Set chart style
        plt.style.use('default')
        
        # 1. MSE/MAE comparison chart
        self._plot_error_metrics()
        
        # 2. Correlation coefficient comparison chart
        self._plot_correlation_metrics()
        
        # 3. Prediction vs true value scatter plot
        self._plot_prediction_scatter()
        
        # 4. Time series comparison chart (first 100 samples)
        self._plot_time_series()
        
        # 5. Error distribution histogram
        self._plot_error_distribution()
        
        # 6. Summary chart
        self._plot_summary()
        
        print(f"All charts saved to: {self.output_dir}")
    
    def _plot_error_metrics(self):
        """Plot MSE and MAE comparison chart"""
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE comparison
        if 'train' in self.metrics and 'test' in self.metrics:
            train_mse = [self.metrics['train']['mse_per_dim'][label] for label in action_labels]
            test_mse = [self.metrics['test']['mse_per_dim'][label] for label in action_labels]
            
            x = np.arange(len(action_labels))
            width = 0.35
            
            ax1.bar(x - width/2, train_mse, width, label='Train', alpha=0.8)
            ax1.bar(x + width/2, test_mse, width, label='Test', alpha=0.8)
            ax1.set_xlabel('Action Dimensions')
            ax1.set_ylabel('MSE')
            ax1.set_title('Mean Squared Error (MSE) Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(action_labels, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # MAE comparison
        if 'train' in self.metrics and 'test' in self.metrics:
            train_mae = [self.metrics['train']['mae_per_dim'][label] for label in action_labels]
            test_mae = [self.metrics['test']['mae_per_dim'][label] for label in action_labels]
            
            ax2.bar(x - width/2, train_mae, width, label='Train', alpha=0.8)
            ax2.bar(x + width/2, test_mae, width, label='Test', alpha=0.8)
            ax2.set_xlabel('Action Dimensions')
            ax2.set_ylabel('MAE')
            ax2.set_title('Mean Absolute Error (MAE) Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(action_labels, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_metrics(self):
        """Plot correlation coefficient comparison chart"""
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train' in self.metrics and 'test' in self.metrics:
            train_corr = [self.metrics['train']['correlation_per_dim'][label] for label in action_labels]
            test_corr = [self.metrics['test']['correlation_per_dim'][label] for label in action_labels]
            
            x = np.arange(len(action_labels))
            width = 0.35
            
            ax.bar(x - width/2, train_corr, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, test_corr, width, label='Test', alpha=0.8)
            ax.set_xlabel('Action Dimensions')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Prediction vs Ground Truth Correlation Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(action_labels, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self):
        """绘制预测vs真实值散点图"""
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(action_labels):
            ax = axes[i]
            
            # 训练集散点图
            if 'actions' in self.train_predictions:
                train_true = self.train_data['actions'][:, i]
                train_pred = self.train_predictions['actions'][:, i]
                
                # 随机采样以避免图表过于密集
                n_samples = min(1000, len(train_true))
                indices = np.random.choice(len(train_true), n_samples, replace=False)
                
                ax.scatter(train_true[indices], train_pred[indices], 
                          alpha=0.6, s=10, label='Train', color='blue')
            
            # 测试集散点图
            if 'actions' in self.test_predictions:
                test_true = self.test_data['actions'][:, i]
                test_pred = self.test_predictions['actions'][:, i]
                
                # 随机采样
                n_samples = min(1000, len(test_true))
                indices = np.random.choice(len(test_true), n_samples, replace=False)
                
                ax.scatter(test_true[indices], test_pred[indices], 
                          alpha=0.6, s=10, label='Test', color='red')
            
            # 画对角线 (完美预测)
            if i < len(action_labels):
                all_values = []
                if 'actions' in self.train_predictions:
                    all_values.extend([train_true.min(), train_true.max()])
                if 'actions' in self.test_predictions:
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
        if len(action_labels) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series(self):
        """绘制时间序列对比图 (前300个样本)"""
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        n_samples = 300
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(action_labels):
            ax = axes[i]
            
            x = np.arange(n_samples)
            
            # 训练集时间序列
            if 'actions' in self.train_predictions and len(self.train_data['actions']) >= n_samples:
                train_true = self.train_data['actions'][:n_samples, i]
                train_pred = self.train_predictions['actions'][:n_samples, i]
                
                ax.plot(x, train_true, label='Train Ground Truth', alpha=0.8, linewidth=2)
                ax.plot(x, train_pred, label='Train Prediction', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'{label}')
            ax.set_title(f'{label} Time Series Comparison (First {n_samples} Samples)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        if len(action_labels) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self):
        """绘制误差分布直方图"""
        action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(action_labels):
            ax = axes[i]
            
            # 训练集误差分布
            if 'actions' in self.train_predictions:
                train_error = self.train_data['actions'][:, i] - self.train_predictions['actions'][:, i]
                ax.hist(train_error, bins=50, alpha=0.6, label='Train Error', density=True)
            
            # 测试集误差分布
            if 'actions' in self.test_predictions:
                test_error = self.test_data['actions'][:, i] - self.test_predictions['actions'][:, i]
                ax.hist(test_error, bins=50, alpha=0.6, label='Test Error', density=True)
            
            ax.set_xlabel(f'Error - {label}')
            ax.set_ylabel('Density')
            ax.set_title(f'{label} Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        if len(action_labels) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary(self):
        """绘制总结图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总体MSE对比
        if 'train' in self.metrics and 'test' in self.metrics:
            categories = ['Train', 'Test']
            mse_values = [self.metrics['train']['mse_total'], self.metrics['test']['mse_total']]
            
            ax1.bar(categories, mse_values, color=['skyblue', 'lightcoral'])
            ax1.set_ylabel('MSE')
            ax1.set_title('Overall Mean Squared Error Comparison')
            ax1.grid(True, alpha=0.3)
            
            # 在柱状图上显示数值
            for i, v in enumerate(mse_values):
                ax1.text(i, v + max(mse_values) * 0.01, f'{v:.6f}', 
                        ha='center', va='bottom')
        
        # 2. 总体MAE对比
        if 'train' in self.metrics and 'test' in self.metrics:
            mae_values = [self.metrics['train']['mae_total'], self.metrics['test']['mae_total']]
            
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
            'Train': len(self.train_data.get('actions', [])),
            'Test': len(self.test_data.get('actions', []))
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
        
        if 'train' in self.metrics and 'test' in self.metrics:
            summary_data = [
                ['Metric', 'Train', 'Test'],
                ['MSE', f"{self.metrics['train']['mse_total']:.6f}", f"{self.metrics['test']['mse_total']:.6f}"],
                ['MAE', f"{self.metrics['train']['mae_total']:.6f}", f"{self.metrics['test']['mae_total']:.6f}"],
                ['Samples', f"{len(self.train_data['actions'])}", f"{len(self.test_data['actions'])}"]
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
    
    def run_analysis(self):
        """运行完整分析"""
        print("="*50)
        print("开始SFT模型性能分析")
        print("="*50)
        
        # 1. 加载数据集
        self.load_dataset()
        
        # 2. 进行预测
        self.predict_actions()
        
        # 3. 计算指标
        metrics = self.calculate_metrics()
        
        # 4. 创建可视化
        self.create_visualizations()
        
        # 5. 打印总结
        self.print_summary(metrics)
        
        print("="*50)
        print("分析完成!")
        print(f"结果保存在: {self.output_dir}")
        print("="*50)
    
    def print_summary(self, metrics):
        """打印分析总结"""
        print("\n分析总结:")
        print("-" * 40)
        
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
            action_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析SFT训练后的模型性能')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/chenyinuo/data/dataset/bingwen/data_for_success/green_bell_pepper_plate_wooden/success",
                       help='数据集目录路径')
    parser.add_argument('--model_path', type=str,
                       default="/mnt/public/chenyinuo/RL4VLA/runs/mlp_sft_steps_50000/step_010000",
                       help='模型路径')
    parser.add_argument('--output_dir', type=str,
                       default="/mnt/public/chenyinuo/RL4VLA/analysis_results",
                       help='输出目录路径')
    parser.add_argument('--max_files', type=int,
                       default=None,
                       help='最大使用的npz文件数量，设置为None使用所有文件。例如: --max_files 10')
    parser.add_argument('--debug', type=bool,
                       default=False,
                       help='是否使用debug模式，使用debug模式时，会使用debug_image.png作为输入')
    
    args = parser.parse_args()
    
    # 打印参数信息
    print("="*50)
    print("分析参数:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  模型路径: {args.model_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  最大文件数: {args.max_files if args.max_files else '使用所有文件'}")
    print("="*50)
    
    # 创建分析器
    analyzer = SFTAnalyzer(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_files=args.max_files,
        debug=args.debug
    )
    
    # 运行分析
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
