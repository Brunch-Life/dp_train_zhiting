#!/usr/bin/env python3
"""
简单的组件测试脚本
用于验证各个组件是否能正常导入和基本功能
"""

import sys
from pathlib import Path

# 添加当前项目路径
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """测试导入"""
    print("测试导入...")
    
    try:
        from data_adapter import DiffusionPolicyDataAdapter
        print("✓ data_adapter 导入成功")
    except Exception as e:
        print(f"✗ data_adapter 导入失败: {e}")
        return False
    
    try:
        from model_wrapper import DiffusionPolicyWrapper
        print("✓ model_wrapper 导入成功")
    except Exception as e:
        print(f"✗ model_wrapper 导入失败: {e}")
        return False
    
    try:
        from visualization_tools import DiffusionPolicyVisualizer
        print("✓ visualization_tools 导入成功")
    except Exception as e:
        print(f"✗ visualization_tools 导入失败: {e}")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    # 测试可视化工具初始化
    try:
        from visualization_tools import DiffusionPolicyVisualizer
        visualizer = DiffusionPolicyVisualizer("test_output")
        action_labels = visualizer.action_labels
        print(f"✓ 可视化工具初始化成功，动作标签: {action_labels}")
    except Exception as e:
        print(f"✗ 可视化工具初始化失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("="*50)
    print("Diffusion Policy 分析工具 - 组件测试")
    print("="*50)
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败，请检查依赖项")
        return False
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n基本功能测试失败")
        return False
    
    print("\n" + "="*50)
    print("所有测试通过！组件运行正常")
    print("="*50)
    
    print("\n使用说明:")
    print("python analyze_diffusion_policy.py --model_path /path/to/model")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)







