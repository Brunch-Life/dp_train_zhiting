#!/usr/bin/env python3
"""
数据加载和预处理适配器
适配当前的Diffusion Policy训练数据格式，实现与eval代码一致的预处理流程
"""

import os
import sys
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加当前项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入训练代码的数据加载器
from datasets.dataset_array import load_sim2sim_data

class DiffusionPolicyDataAdapter:
    """
    适配Diffusion Policy的数据加载和预处理
    与训练代码和eval代码保持一致
    """
    
    def __init__(self, data_config, norm_stats_path=None):
        """
        Args:
            data_config: 数据配置字典，包含data_roots, num_seeds等
            norm_stats_path: 归一化参数文件路径
        """
        self.data_config = data_config
        self.norm_stats_path = norm_stats_path
        
        # 图像预处理变换（与eval代码一致）
        self.image_transforms = transforms.Compose([
            transforms.RandomCrop((int(224 * 0.95), int(224 * 0.95))),  # 95%裁剪
            transforms.Resize((224, 224), antialias=True),
        ])
        
        # 加载数据和归一化参数
        self._load_data()
        self._setup_normalization()
    
    def _load_data(self):
        """加载训练和验证数据"""
        print("正在加载数据...")
        
        # 使用训练代码的数据加载函数
        (self.train_dataloader, 
         self.val_dataloader, 
         self.train_norm_stats, 
         self.val_norm_stats) = load_sim2sim_data(**self.data_config)
        
        print(f"训练数据加载完成，批次数: {len(self.train_dataloader)}")
        print(f"验证数据加载完成，批次数: {len(self.val_dataloader)}")
    
    def _setup_normalization(self):
        """设置归一化参数（与eval代码一致）"""
        if self.norm_stats_path and os.path.exists(self.norm_stats_path):
            print(f"从文件加载归一化参数: {self.norm_stats_path}")
            with open(self.norm_stats_path, "rb") as f:
                obs_normalize_params = pickle.load(f)
        else:
            print("使用训练数据的归一化参数")
            obs_normalize_params = self.train_norm_stats
        
        # 提取归一化参数（与eval代码完全一致）
        self.pose_gripper_mean = np.concatenate([
            obs_normalize_params[key]["mean"]
            for key in ["pose", "gripper_width"]
        ])
        self.pose_gripper_scale = np.concatenate([
            obs_normalize_params[key]["scale"]
            for key in ["pose", "gripper_width"]
        ])
        self.proprio_gripper_mean = np.concatenate([
            obs_normalize_params[key]["mean"]
            for key in ["proprio_state", "gripper_width"]
        ])
        self.proprio_gripper_scale = np.concatenate([
            obs_normalize_params[key]["scale"]
            for key in ["proprio_state", "gripper_width"]
        ])
        
        print(f"归一化参数设置完成:")
        print(f"  pose_gripper_mean shape: {self.pose_gripper_mean.shape}")
        print(f"  pose_gripper_scale shape: {self.pose_gripper_scale.shape}")
        print(f"  proprio_gripper_mean shape: {self.proprio_gripper_mean.shape}")
        print(f"  proprio_gripper_scale shape: {self.proprio_gripper_scale.shape}")
    
    def process_images_like_eval(self, images):
        """
        按照eval代码的方式预处理图像
        Args:
            images: torch.Tensor, shape (B, M, C, H, W) 或 (B, M, H, W, C)
        Returns:
            processed_images: torch.Tensor, shape (B, M, C, 224, 224), [0,1]范围
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # 确保格式是 (B, M, C, H, W)
        if images.dim() == 5 and images.shape[-1] == 3:  # (B, M, H, W, C)
            images = images.permute(0, 1, 4, 2, 3)  # -> (B, M, C, H, W)
        
        B, M, C, H, W = images.shape
        
        # 重塑为 (B*M, C, H, W) 以便批量处理
        images = images.reshape(B * M, C, H, W)
        
        # 确保是float类型并归一化到[0,1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # 应用变换
        try:
            images = self.image_transforms(images)
        except Exception as e:
            print(f"图像变换失败: {e}")
            # 如果变换失败，直接resize
            images = transforms.Resize((224, 224), antialias=True)(images)
        
        # 重塑回 (B, M, C, 224, 224)
        images = images.view(B, M, C, 224, 224)
        
        return images
    
    def process_states_like_eval(self, states):
        """
        按照eval代码的方式预处理状态
        Args:
            states: np.ndarray 或 torch.Tensor, shape (B, 10)
        Returns:
            processed_states: torch.Tensor, shape (B, 10), 归一化后
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        states = np.asarray(states)
        if states.ndim == 1:
            states = states[None, :]  # (1, 10)
        
        # 应用mask（与eval代码一致）
        masks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        states = np.where(masks, states, np.zeros_like(states))
        
        # 归一化（与eval代码一致）
        states = (states - self.proprio_gripper_mean[None, :]) / self.proprio_gripper_scale[None, :]
        
        return torch.from_numpy(states).float()
    
    def denormalize_action(self, action):
        """
        反归一化动作（与eval代码一致）
        Args:
            action: torch.Tensor 或 np.ndarray, shape (B, 10)
        Returns:
            denormalized_action: np.ndarray, shape (B, 10)
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.asarray(action)
        action = action * self.pose_gripper_scale[None, :] + self.pose_gripper_mean[None, :]
        return action
    
    def extract_data_for_analysis(self, dataloader, max_batches=None, split_name="train"):
        """
        从dataloader中提取数据用于分析
        Args:
            dataloader: 数据加载器
            max_batches: 最大批次数，None表示使用所有数据
            split_name: 数据集分割名称，用于日志
        Returns:
            dict: 包含images, actions, states的字典
        """
        print(f"正在从{split_name}数据中提取样本...")
        
        all_images = []
        all_actions = []
        all_states = []
        
        total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"提取{split_name}数据")):
                if max_batches and i >= max_batches:
                    break
                
                # 提取数据
                images = batch["images"]  # (B, M, C, H, W)
                actions = batch["action"]  # (B, chunk_size, 10)
                states = batch["proprio_state"]  # (B, 10)
                is_pad = batch["is_pad"]  # (B, chunk_size)
                
                # 只使用第一个时间步的动作（与训练时一致）
                actions = actions[:, 0, :]  # (B, 10)
                
                # 预处理图像（按eval方式）
                processed_images = self.process_images_like_eval(images)
                
                # 预处理状态（按eval方式）
                processed_states = self.process_states_like_eval(states)
                
                # 保存反归一化的动作（用于与预测结果比较）
                original_actions = self.denormalize_action(actions)
                
                all_images.append(processed_images.cpu())
                all_actions.append(torch.from_numpy(original_actions))
                all_states.append(processed_states.cpu())
        
        # 合并所有批次
        result = {
            'images': torch.cat(all_images, dim=0),     # (N, M, C, 224, 224)
            'actions': torch.cat(all_actions, dim=0),   # (N, 10)
            'states': torch.cat(all_states, dim=0)      # (N, 10)
        }
        
        print(f"{split_name}数据提取完成:")
        print(f"  图像形状: {result['images'].shape}")
        print(f"  动作形状: {result['actions'].shape}")
        print(f"  状态形状: {result['states'].shape}")
        
        return result
    
    def get_train_data(self, max_batches=None):
        """获取训练数据"""
        return self.extract_data_for_analysis(
            self.train_dataloader, max_batches, "训练"
        )
    
    def get_val_data(self, max_batches=None):
        """获取验证数据"""
        return self.extract_data_for_analysis(
            self.val_dataloader, max_batches, "验证"
        )
    
    def get_camera_names(self):
        """获取相机名称"""
        return ["third", "wrist"]  # 与训练配置一致
    
    def get_action_dim(self):
        """获取动作维度"""
        return 10
    
    def get_action_labels(self):
        """获取动作维度标签"""
        return ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'rot_4', 'rot_5', 'gripper']







