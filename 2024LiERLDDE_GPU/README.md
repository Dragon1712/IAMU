# ERL-DDE 模型空候选修复（第 70 轮 Rollout 报错）

诊断：`model.py:104` 的 `seq3` 为 `tensor([], size=(0, 3))`。对空 attention 求 `argmax` 时抛异常，原来的 `except` 只打印状态，导致随后访问未赋值 `max_index`，触发 `UnboundLocalError`。

## 修复范围

仅修改 `model.py` 内的 `ES.forward`：

1. `seq1, seq2, seq3, id = self.cope_state(state)` 之后若 `seq3.shape[0] == 0`，返回原代码已使用的等待动作 `np.array([-1, -1, -1, -1, -1])`。
2. 把吞掉 argmax 异常的 `try/except` 改为检查空 attention；如果输入候选非空但 attention 仍为空，显式报告真实问题。
3. 不修改 KDE、损失/奖励、训练器、网络结构和权重大小。

调用 C++ `step` 时，非法船舶 id（如 -1）走原来的时间推进分支。该修复为既有哨兵动作补齐空候选状态处理，不是对“有效候选”选择规则的更改。

## 一键应用到你的服务器

将本目录的两个 `.py` 文件复制到包含 `model.py` 的项目目录；先备份/确保其余实验检查点不受影响：

```bash
cd ~/code/iamu/2024LiERLDDE_GPU
python apply_model_empty_seq3_fix.py model.py
python -m py_compile model.py
python test_model_empty_seq3.py
```

自动创建 `model.py.before_empty_seq3_fix`，便于回滚；如果找不到预期的源码模式，脚本会报错且不会覆盖原文件。

## 手动修改方式

在 `model.py` 中找到：

```python
seq1, seq2, seq3, id = self.cope_state(state)
```

在其下一行（保持缩进）增加：

```python
if seq3.shape[0] == 0:
    return np.array([-1, -1, -1, -1, -1])
```

找到原先的：

```python
try:
    max_index = torch.argmax(attention_weights)
except:
    print(seq3)
    print(state)
```

替换为：

```python
if attention_weights.numel() == 0:
    raise RuntimeError('Empty attention weights with nonempty seq3')
max_index = torch.argmax(attention_weights)
```

## 从第 70 轮恢复

你的日志已经确认第 69 轮结束后，保存了 `next_iter=70` 的完整状态。先备份完整检查点，再运行（按原来的运行设备和参数）：

```bash
cd ~/code/iamu/2024LiERLDDE_GPU
CKPT=BACAP_KDE_FIXED_checkpoints/1/checkpoint_full_latest.pt
cp -a "$CKPT" "${CKPT}.before_model_fix"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

python -u main_kde_fixed.py \
  --resume "$CKPT" \
  --device cpu --kde-device cpu \
  --cases 200 --loops 1 --populations 5 --lambda-size 80 \
  --kde-r1 0.9975 --kde-r2 0.9990 \
  --kde-max-attempts 24 --kde-restart-attempts 64 \
  --full-checkpoint-every 2 --seed 42 \
  2>&1 | tee -a fixed_resume_model70.log
```

如果此前使用其他 KDE 阈值/种群规模/设备，必须把命令替换为对应原配置；恢复函数会校验 KDE 参数。恢复到第 70 轮开始，而非第 70 轮已运行到 78% 的中间位置。

## 验证范围和注意事项

- 补丁脚本针对你之前使用的原始 `model.py`（含 `try: max_index = torch.argmax(attention_weights)` 和 `except: print(seq3); print(state)`），未在你服务器当前文件上执行。
- 本地验证源码替换语法和空候选逻辑；需要你在现有 PyTorch + C++ 环境下运行冒烟测试。
- 共享同一 `model.py` 的消融训练进程，已加载的 Python 模块通常不会因为磁盘修改而自动变更。新启动/重启的实验会使用修复代码；保留各实验原始日志，标记修复版本。
