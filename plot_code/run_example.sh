#!/bin/bash
# Diffusion Policy 分析工具使用示例

echo "===================================================="
echo "Diffusion Policy 分析工具 - 使用示例"
echo "===================================================="

# 激活环境 (如果需要)
# conda activate dp_train

# 示例1: 分析最新的模型 (使用较少数据以便快速测试)
echo "示例1: 快速分析 (使用部分数据)"
python analyze_diffusion_policy.py \
    --model_path ../ckpts/20250906_005428/policy_best.ckpt \
    --data_root ../dataset/processed/nonstop_plate_wooden_new \
    --output_dir analysis_results_quick \
    --batch_size 16 \
    --max_train_batches 20 \
    --max_test_batches 10

echo "===================================================="
echo "示例2: 完整分析 (使用所有数据)"
echo "注意：这可能需要较长时间和较多GPU内存"
echo "python analyze_diffusion_policy.py \\"
echo "    --model_path ../ckpts/20250906_005428/policy_best.ckpt \\"
echo "    --data_root ../dataset/processed/nonstop_plate_wooden_new \\"
echo "    --output_dir analysis_results_full \\"
echo "    --batch_size 32"

echo "===================================================="
echo "可用的模型检查点:"
find ../ckpts -name "policy_best.ckpt" 2>/dev/null | head -3

echo "===================================================="
echo "分析完成后，查看结果:"
echo "ls analysis_results_quick/"
echo "===================================================="





