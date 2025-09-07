import numpy as np

data_dir = "/ML-vePFS/tangyinzhou/bingwen/ManiSkill/videos/datasets_mp/TabletopPickPlaceEnv-v1/20250903_122339/nonstop_plate_wooden/success/success_optimal_nonstop_plate_wooden_proc_0_num_0_trynum_0_epsid_5000.npz"

# 首先检查npz文件中所有的键
npz_data = np.load(data_dir, allow_pickle=True)
print("NPZ文件中的所有键:", list(npz_data.keys()))

# 加载arr_0数据
data = npz_data["arr_0"]

print("="*60)
print("数据详细分析")
print("="*60)

print(f"数据类型: {type(data)}")

# 递归函数来深入检查数据结构
def analyze_data(obj, indent=0, name="data"):
    prefix = "  " * indent
    print(f"{prefix}{name}: {type(obj)}")
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                print(f"{prefix}  {key}: numpy数组, 形状={value.shape}, dtype={value.dtype}")
                if value.size > 0 and value.dtype != 'object':
                    try:
                        print(f"{prefix}    范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
                    except:
                        print(f"{prefix}    (无法计算数值范围)")
            elif isinstance(value, dict):
                print(f"{prefix}  {key}: 字典 (包含{len(value)}个键)")
                analyze_data(value, indent+2, key)
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}  {key}: {type(value).__name__}, 长度={len(value)}")
                if len(value) > 0:
                    print(f"{prefix}    首个元素类型: {type(value[0])}")
            else:
                print(f"{prefix}  {key}: {type(value)}, 值={value}")
    
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}  形状: {obj.shape}")
        print(f"{prefix}  数据类型: {obj.dtype}")
        if obj.size > 0 and obj.dtype != 'object':
            try:
                print(f"{prefix}  数值范围: [{np.min(obj):.6f}, {np.max(obj):.6f}]")
            except:
                print(f"{prefix}  (无法计算数值范围)")
        elif obj.dtype == 'object':
            print(f"{prefix}  这是一个对象数组，尝试查看内容...")
            if obj.size == 1:  # 单个对象
                actual_data = obj.item()
                print(f"{prefix}  内部对象类型: {type(actual_data)}")
                if isinstance(actual_data, dict):
                    analyze_data(actual_data, indent+1, "内部字典")
    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}  长度: {len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}  首个元素类型: {type(obj[0])}")

# 检查数据结构
if data.dtype == 'object' and data.size == 1:
    # 这是一个包含单个对象的numpy数组
    actual_data = data.item()
    print(f"实际数据类型: {type(actual_data)}")
    analyze_data(actual_data)
else:
    analyze_data(data)

print("\n" + "="*60)
print("关键数据字段详情")
print("="*60)

# 获取实际的数据字典
if data.dtype == 'object' and data.size == 1:
    actual_data = data.item()
else:
    actual_data = data

# 显示一些关键字段的详细信息
if isinstance(actual_data, dict):
    important_keys = ['obs', 'action', 'reward', 'done', 'info', 'reserve', 'action_config', 'success']
    
    for key in important_keys:
        if key in actual_data:
            print(f"\n{key}:")
            if isinstance(actual_data[key], dict):
                for subkey in actual_data[key].keys():
                    if isinstance(actual_data[key][subkey], np.ndarray):
                        arr = actual_data[key][subkey]
                        print(f"  {subkey}: 形状={arr.shape}, dtype={arr.dtype}")
                    else:
                        print(f"  {subkey}: {type(actual_data[key][subkey])}")
            else:
                print(f"  类型: {type(actual_data[key])}")
                if hasattr(actual_data[key], 'shape'):
                    print(f"  形状: {actual_data[key].shape}")

print("\n" + "="*60)
print("数据内容示例")
print("="*60)

# 显示一些具体的数据内容示例
if isinstance(actual_data, dict):
    # 显示观测数据的第一帧
    if 'obs' in actual_data and isinstance(actual_data['obs'], dict):
        print("\n观测数据第一帧示例:")
        for key, value in actual_data['obs'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray) and subvalue.size > 0:
                        print(f"    {subkey}: 形状={subvalue.shape}, 第一个值={subvalue.flat[0]:.6f}")
    
    # 显示动作数据
    if 'action' in actual_data and isinstance(actual_data['action'], np.ndarray):
        print(f"\n动作数据:")
        print(f"  形状: {actual_data['action'].shape}")
        if actual_data['action'].size > 0:
            print(f"  第一个动作: {actual_data['action'][0]}")
    
    # 显示成功标志
    if 'success' in actual_data:
        print(f"\n任务成功: {actual_data['success']}")

print("\n" + "="*60)
print("保存wrist_rgb图像")
print("="*60)

import os
import cv2

# 创建debug_image文件夹
debug_dir = "debug_image"
os.makedirs(debug_dir, exist_ok=True)

# 读取wrist_rgb图像数据
if isinstance(actual_data, dict) and 'observation' in actual_data:
    obs = actual_data['observation']
    if 'wrist_rgb' in obs:
        wrist_rgb_list = obs['wrist_rgb']
        print(f"找到 {len(wrist_rgb_list)} 张wrist_rgb图像")
        
        # 保存前10张图像作为示例
        num_to_save = min(10, len(wrist_rgb_list))
        print(f"保存前 {num_to_save} 张图像到 {debug_dir} 文件夹...")
        
        for i in range(num_to_save):
            img = wrist_rgb_list[i]
            print(f"  图像 {i}: 形状={img.shape}, dtype={img.dtype}")
            
            # 如果图像是RGB格式，需要转换为BGR格式给OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                # RGB转BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            
            # 保存图像
            filename = os.path.join(debug_dir, f"wrist_rgb_frame_{i:03d}.jpg")
            success = cv2.imwrite(filename, img_bgr)
            
            if success:
                print(f"    已保存: {filename}")
            else:
                print(f"    保存失败: {filename}")
        
        print(f"\n总共保存了 {num_to_save} 张wrist_rgb图像到 {debug_dir} 文件夹")
    else:
        print("未找到wrist_rgb数据")
else:
    print("未找到observation数据")

print("\n" + "="*60)
