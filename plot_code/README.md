# Diffusion Policy 分析工具

基于 `analy_reference.py` 开发的 Diffusion Policy 模型性能分析工具，能够分析训练后的模型在训练集和测试集上的性能表现。

## 文件结构

```
plot_code/
├── analyze_diffusion_policy.py   # 主分析脚本
├── data_adapter.py               # 数据加载和预处理适配器
├── model_wrapper.py              # 模型加载和推理包装器
├── visualization_tools.py        # 可视化工具
└── README.md                     # 使用说明
```

## 功能特性

- **数据适配**: 复用训练代码的数据加载器，确保数据格式一致
- **模型推理**: 适配 DiffusionPolicy 模型，支持批量预测
- **性能指标**: 计算 MSE、MAE、相关系数等评估指标  
- **可视化**: 生成多种分析图表，包括：
  - MSE/MAE 对比图
  - 相关系数对比图
  - 预测vs真实值散点图
  - 时间序列对比图
  - 误差分布直方图
  - 总结图表

## 使用方法

### 基本用法

```bash
cd plot_code
python analyze_diffusion_policy.py --model_path /path/to/your/model/checkpoint
```

### 完整参数示例

```bash
python analyze_diffusion_policy.py \
    --data_root dataset/processed/nonstop_plate_wooden_new \
    --model_path ckpts/20250906_005428/policy_best.ckpt \
    --output_dir analysis_results_dp \
    --num_seeds 500 \
    --chunk_size 20 \
    --batch_size 32 \
    --max_train_batches 100 \
    --max_test_batches 50
```

### 参数说明

- `--data_root`: 数据根目录路径（默认: dataset/processed/nonstop_plate_wooden_new）
- `--model_path`: 模型检查点路径（必需）
- `--output_dir`: 输出目录路径（默认: analysis_results_dp）
- `--num_seeds`: 数据集种子数量（默认: 500）
- `--chunk_size`: 训练时使用的chunk大小（默认: 20）
- `--batch_size`: 分析时的批次大小（默认: 32）
- `--max_train_batches`: 最大训练批次数，None表示使用所有数据
- `--max_test_batches`: 最大测试批次数，None表示使用所有数据

## 输出结果

分析完成后，会在输出目录中生成以下文件：

- `metrics.json`: 详细的性能指标数据
- `error_metrics.png`: MSE/MAE 对比图
- `correlation_metrics.png`: 相关系数对比图
- `prediction_scatter.png`: 预测vs真实值散点图
- `time_series.png`: 时间序列对比图
- `error_distribution.png`: 误差分布直方图
- `summary.png`: 总结图表

## 注意事项

1. **数据路径**: 确保数据根目录存在且包含正确的数据结构
2. **模型路径**: 支持 `.ckpt` 文件或包含 `policy_best.ckpt` 的目录
3. **归一化参数**: 需要对应的 `norm_stats_1_epsnum_500.pkl` 文件
4. **GPU内存**: 根据 GPU 内存调整 `batch_size` 参数
5. **分析时间**: 对于大数据集，可以使用 `max_train_batches` 和 `max_test_batches` 限制数据量

## 与 analy_reference.py 的兼容性

本工具保持了与 `analy_reference.py` 相同的：
- 分析流程和方法
- 可视化图表类型和风格
- 性能指标计算方式
- 输出文件格式

主要差异：
- 适配 DiffusionPolicy 而非 MLPPolicy
- 支持 10 维动作空间而非 7 维
- 使用结构化数据集而非 NPZ 文件
- 支持多相机输入





