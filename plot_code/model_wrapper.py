#!/usr/bin/env python3
"""
模型加载和推理包装器
适配DiffusionPolicy模型，实现与eval代码一致的推理流程
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加当前项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入模型
from models.policy import DiffusionPolicy

class DiffusionPolicyWrapper:
    """
    DiffusionPolicy模型包装器
    提供与eval代码一致的推理接口
    """
    
    def __init__(self, model_path, norm_stats_path, device='cuda'):
        """
        Args:
            model_path: 模型检查点路径
            norm_stats_path: 归一化参数文件路径  
            device: 设备类型
        """
        self.model_path = Path(model_path)
        self.norm_stats_path = norm_stats_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"模型路径: {self.model_path}")
        print(f"归一化参数路径: {self.norm_stats_path}")
        
        # 设置归一化参数
        self._setup_normalization()
        
        # 加载模型
        self._load_model()
    
    def _setup_normalization(self):
        """设置归一化参数（与eval代码一致）"""
        if self.norm_stats_path and os.path.exists(self.norm_stats_path):
            print("正在加载归一化参数...")
            with open(self.norm_stats_path, "rb") as f:
                self.obs_normalize_params = pickle.load(f)
        else:
            raise FileNotFoundError(f"归一化参数文件不存在: {self.norm_stats_path}")
        
        # 提取归一化参数（与eval代码完全一致）
        self.pose_gripper_mean = np.concatenate([
            self.obs_normalize_params[key]["mean"]
            for key in ["pose", "gripper_width"]
        ])
        self.pose_gripper_scale = np.concatenate([
            self.obs_normalize_params[key]["scale"]
            for key in ["pose", "gripper_width"]
        ])
        self.proprio_gripper_mean = np.concatenate([
            self.obs_normalize_params[key]["mean"]
            for key in ["proprio_state", "gripper_width"]
        ])
        self.proprio_gripper_scale = np.concatenate([
            self.obs_normalize_params[key]["scale"]
            for key in ["proprio_state", "gripper_width"]
        ])
        
        print("归一化参数加载完成")
    
    def _load_model(self):
        """加载模型（基于eval代码的配置）"""
        print("正在加载模型...")
        
        # 模型配置（与eval代码一致）
        self.policy_config = {
            'lr': 1e-5,
            'num_images': 2,  # third, wrist
            'action_dim': 10,
            'observation_horizon': 1,
            'action_horizon': 1,
            'prediction_horizon': 20,  # 训练时使用的chunk_size
            'global_obs_dim': 10,
            'num_inference_timesteps': 10,
            'ema_power': 0.75,
            'vq': False,
        }
        
        # 创建模型
        self.policy = DiffusionPolicy(self.policy_config)
        
        # 加载检查点
        if self.model_path.suffix == '.ckpt':
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.policy.deserialize(checkpoint)
        else:
            # 尝试寻找最佳检查点
            best_ckpt_path = self.model_path / 'policy_best.ckpt'
            if best_ckpt_path.exists():
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')
                self.policy.deserialize(checkpoint)
            else:
                raise FileNotFoundError(f"找不到模型检查点: {self.model_path}")
        
        # 设置为评估模式并移到GPU
        self.policy.eval()
        self.policy.to(self.device)
        
        print("模型加载完成")
    
    def denormalize_action(self, action):
        """
        反归一化动作（与eval代码一致）
        Args:
            action: torch.Tensor, shape (B, 10)
        Returns:
            denormalized_action: np.ndarray, shape (B, 10)
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.asarray(action)
        action = action * self.pose_gripper_scale[None, :] + self.pose_gripper_mean[None, :]
        return action
    
    def normalize_states(self, states):
        """
        归一化状态（与eval代码一致）
        Args:
            states: np.ndarray 或 torch.Tensor, shape (B, 10)
        Returns:
            normalized_states: torch.Tensor, shape (B, 10)
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        states = np.asarray(states)
        if states.ndim == 1:
            states = states[None, :]
        
        # 应用mask（与eval代码一致）
        masks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        states = np.where(masks, states, np.zeros_like(states))
        
        # 归一化
        states = (states - self.proprio_gripper_mean[None, :]) / self.proprio_gripper_scale[None, :]
        
        return torch.from_numpy(states).float()
    
    def predict_batch(self, images, states, batch_size=32):
        """
        批量预测动作
        Args:
            images: torch.Tensor, shape (N, M, C, H, W), [0,1]范围
            states: torch.Tensor, shape (N, 10)
            batch_size: 批次大小
        Returns:
            predictions: np.ndarray, shape (N, 10), 反归一化后的动作
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        
        num_samples = images.shape[0]
        all_predictions = []
        
        print(f"开始批量预测，总样本数: {num_samples}, 批次大小: {batch_size}")
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="模型预测"):
                end_idx = min(i + batch_size, num_samples)
                
                # 获取当前批次
                batch_images = images[i:end_idx].to(self.device)  # (B, M, C, H, W)
                batch_states = states[i:end_idx]  # (B, 10)
                
                # 归一化状态
                normalized_states = self.normalize_states(batch_states).to(self.device)
                
                # 模型推理
                pred_actions = self.policy(normalized_states, batch_images)  # (B, chunk_size, 10)
                
                # 取第一个时间步的动作
                if pred_actions.dim() == 3:
                    pred_actions = pred_actions[:, 0, :]  # (B, 10)
                
                # 反归一化
                denormalized_actions = self.denormalize_action(pred_actions)
                all_predictions.append(denormalized_actions)
        
        # 合并所有预测结果
        predictions = np.concatenate(all_predictions, axis=0)
        
        print(f"批量预测完成，输出形状: {predictions.shape}")
        return predictions
    
    def predict_single(self, image, state):
        """
        单次预测（用于调试）
        Args:
            image: torch.Tensor, shape (M, C, H, W) 或 (1, M, C, H, W)
            state: torch.Tensor, shape (10,) 或 (1, 10)
        Returns:
            prediction: np.ndarray, shape (10,)
        """
        # 确保是批次格式
        if image.dim() == 4:  # (M, C, H, W)
            image = image.unsqueeze(0)  # (1, M, C, H, W)
        if state.dim() == 1:  # (10,)
            state = state.unsqueeze(0)  # (1, 10)
        
        prediction = self.predict_batch(image, state, batch_size=1)
        return prediction[0]  # 返回单个样本
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_path': str(self.model_path),
            'policy_config': self.policy_config,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.policy.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        }
    
    def save_predictions(self, images, states, output_path, batch_size=32):
        """
        保存预测结果到文件
        Args:
            images: 输入图像
            states: 输入状态  
            output_path: 输出文件路径
            batch_size: 批次大小
        """
        predictions = self.predict_batch(images, states, batch_size)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, predictions)
        print(f"预测结果已保存到: {output_path}")
        
        return predictions





