# Sudoku Diffusion LM - Quick Start Guide

## 🚀 一键运行完整流程

```bash
cd /data/szhang967/dlm_agents/sudoku

# 激活环境
source ~/.bashrc
conda activate dream

# 运行完整实验管道
bash run_experiment.sh
```

## 📋 分步执行

### 步骤 1: 测试模型实现

```bash
python test_model.py
```

**输出**: 所有测试应该通过，显示 29M 参数

### 步骤 2: 生成训练数据

```bash
# 生成 48k 训练 + 2k 测试样本 (~20-30分钟)
python generate_data.py \
    --train-size 48000 \
    --test-size 2000 \
    --output-dir ./data
```

**输出**: 
- `data/train_puzzles.npy` (48k 样本)
- `data/train_solutions.npy`
- `data/test_puzzles.npy` (2k 样本)
- `data/test_solutions.npy`

### 步骤 3: 训练模型

```bash
# 训练约 8-12 小时 (单卡 A100)
python train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --batch-size 128 \
    --num-steps 100000 \
    --lr 1e-4
```

**监控训练**:
```bash
# 在新终端中
tensorboard --logdir ./checkpoints/tensorboard
```

**输出**:
- `checkpoints/best_model.pt` (最佳模型)
- `checkpoints/final_model.pt` (最终模型)
- `checkpoints/checkpoint_*.pt` (定期保存)

### 步骤 4: 测试求解

```bash
# 在测试集上评估
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 10 \
    --steps 10 \
    --algorithm random-remask
```

**可选**: 求解自定义数独

```bash
# 创建数独文件 (0 表示空白)
cat > my_puzzle.txt << EOF
5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9
EOF

python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --puzzle-file my_puzzle.txt \
    --steps 10
```

## 🎯 核心设计

### 数据表示

- **输入格式**: 89 tokens 序列
  - 81 个单元格 (0-9, 0=空白)
  - 8 个行分隔符 (token 10)
- **词表**: 12 个 tokens
  - 0-9: 数独数字
  - 10: EOL (行分隔符)
  - 11: MASK (用于扩散训练)

### 模型架构

```
29M 参数 Diffusion Transformer:
├─ Token Embedding: 12 × 448
├─ Position Embedding: 89 × 448 (可学习)
├─ 12× Transformer Block:
│  ├─ Multi-Head Attention (8 heads, bidirectional)
│  └─ Feed-Forward Network (448 → 1792 → 448)
└─ Output Head: 448 × 12
```

### 训练策略

- **动态 Masking**: 每个样本随机 mask 20-90% 的 tokens
- **目标**: 预测被 mask 的 tokens
- **优化器**: AdamW (lr=1e-4, cosine decay)
- **批量大小**: 128
- **训练步数**: 100k (~530 epochs)

### 推理策略

兼容 `llada_sample.py` 的多种采样算法:

1. **random-remask**: 随机重新 mask (默认)
2. **self_conf-remask:vanilla**: 基于置信度的重新 mask
3. **self_conf-remask:entropy**: 基于熵的置信度
4. **self_conf-remask:topk_margin**: 基于 top-k margin 的置信度

## 📊 预期结果

| 指标 | 预期值 |
|------|--------|
| 训练准确率 (token) | >99% |
| 测试准确率 (token) | >95% |
| 解的有效性 | 80-90% |

*注: token 准确率衡量单个位置预测，有效性衡量完整的数独约束满足*

## 🔧 调试技巧

### GPU 内存不足

```bash
# 减小批量大小
python train.py --batch-size 64 ...
```

### 加速训练

```bash
# 减少训练步数（快速验证）
python train.py --num-steps 10000 ...
```

### 提高求解质量

```bash
# 增加推理步数
python inference.py --steps 20 ...

# 使用基于置信度的采样
python inference.py --algorithm self_conf-remask:vanilla ...
```

## 📚 文件结构

```
sudoku/
├── generate_data.py      # 数据生成
├── model.py              # DiT 模型定义
├── train.py              # 训练脚本
├── inference.py          # 推理/求解
├── test_model.py         # 模型测试
├── example_usage.py      # 使用示例
├── train_config.yaml     # 训练配置
├── DLM_README.md         # 详细文档
├── QUICKSTART.md         # 本文件
└── run_experiment.sh     # 一键运行脚本
```

## ❓ 常见问题

**Q: 训练需要多久？**
A: 在单张 A100 上约 8-12 小时完成 100k 步。可以用 `--num-steps 10000` 快速验证。

**Q: 可以在 CPU 上运行吗？**
A: 可以，但会很慢。推理可以在 CPU 上进行，训练建议使用 GPU。

**Q: 为什么解不总是有效？**
A: Diffusion 模型学习数据分布，但不保证约束满足。可以通过更多训练步数和推理步数提高成功率。

**Q: 如何与 llada_sample.py 集成？**
A: 模型已经兼容！见 `example_usage.py` 中的示例代码。

## 📧 联系

有问题或建议？欢迎提 issue 或联系作者。

---

**Happy Sudoku Solving with Diffusion! 🎲✨**

