# ERL-DDE-KDE-Fixed (modified experiment baseline)

**不是作者原算法的逐位等价优化版。** 此版本针对原始 ERL-DDE 在第 29 轮出现的 `accepted=0/80, rejected=118900/118900` 现象，设计了有界 KDE 筛选，确保候选采样不会无限循环。

代码以 `Dragon1712/IAMU` 的 `train.py`（SHA `221a99c0a82970830cc66e65fe2dfa8a7c6691ff`）、`main.py` 为基础。网络 `model.py`、Linux 版 `env.py` 和 C++ `libBACAP_env.so` 沿用用户现有的已通过冒烟测试的代码，无需替换。

## 文件

- `main_kde_fixed.py`：独立训练入口，保持原 `main.py` 不变；支持实验参数、热启动和完整状态恢复。
- `train_fixed.py`：保留原始 Rollout / ES 更新，修改 KDE 候选筛选及种群排斥；增加监控和完整状态检查点。
- `kde_fixed.py`：规范化 KDE、参考矩阵缓存、快速距离计算、阈值附近数值回退、有限采样及最低密度回退。
- `test_fixed.py`：不依赖港口数据的模拟单元测试和三轮训练/中途恢复测试。

## 1. 安装

**先备份原始实验和代码，且不要覆盖原 `train.py` 或 `main.py`。** 例如：

```bash
cd ~/code/iamu/2024LiERLDDE
mkdir -p ../ERL-DDE-KDE-Fixed-source
# 将本包解压至 ../ERL-DDE-KDE-Fixed-source 后：
cp ../ERL-DDE-KDE-Fixed-source/{main_kde_fixed.py,train_fixed.py,kde_fixed.py,test_fixed.py} .
python -m py_compile main_kde_fixed.py train_fixed.py kde_fixed.py
```

或把 ZIP 直接解压到 `2024LiERLDDE` 项目目录。**这里不含作者的 `model.py/env.py/case_code/` 和动态库，需保留现有项目原文件。**

## 2. 先完成小规模测试

```bash
conda activate erldde
cd ~/code/iamu/2024LiERLDDE
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python test_fixed.py
python -u main_kde_fixed.py \
  --cases 3 --populations 5 --lambda-size 8 \
  --kde-max-attempts 8 --kde-restart-attempts 16 \
  --full-checkpoint-every 1 2>&1 | tee fixed_smoke.log
```

注意：`test_fixed.py` 使用模拟网络及环境来测试控制流程，它不是实际港口环境测试。

## 3. 正式修正版训练（CPU）

```bash
python -u main_kde_fixed.py \
  --device cpu --kde-device cpu \
  --cases 200 --populations 5 --lambda-size 80 \
  --kde-r1 0.9975 --kde-r2 0.9990 \
  --kde-max-attempts 24 --kde-restart-attempts 64 \
  --full-checkpoint-every 10 --seed 42 \
  2>&1 | tee fixed_train.log
```

**0.9975 / 0.9990 是供起步验证的实验性阈值，不是原论文值，也未在你的 200 个案例上标定。** 建议先检查前 5–10 轮的 `accepted_by_threshold` 和 `forced_min_density`。若绝大多数是强制回退，建议研究阈值分位数、扰动尺度和多样性指标，再做阈值消融实验。

## 4. 修改了哪些算法内容？

- 原 KDE 公式是 `sum(exp(-.5*(distance/h)**2))/h`，原阈值 `r1=.5`、`r2=.2` 会随着参考模型数量增加而变得更容易触发。修正 KDE 为 **平均相似度** `mean(exp(-.5*(distance/h)**2))`，阈值重新指定在 `(0,1]` 区间。
- 每个候选最多尝试 `--kde-max-attempts` 次：第一次 KDE 达标立即接受；全部不达标时选择这次尝试中 KDE 最低的候选。这样始终生成指定个数候选，但可能接受高于新阈值的模型。
- 种群删除后的重新初始化不再使用无界 `while True`；最多重试 `--kde-restart-attempts` 次，仍未达标则使用最低 KDE 的初始化。
- 缓存每个种群的固定参考参数矩阵及平方范数，使用矩阵—向量乘法计算距离；接近阈值时用原 `pairwise_distance` 复核。
- 原代码每轮向已预分配长度为 `lambda_` 的 `all_models[i]` 追加未被 rollout 读取的模型引用，本版本取消这项追加以避免内存增长。
- 原始 ES 网络、适应度/奖励、演化梯度更新与环境交互流程保持；但 KDE 筛选、初始化、随机种子流和训练轨迹会改变，因此**结果必须作为新算法报告，而不是原论文完全复现结果**。

## 5. 诊断日志

```text
[KDE CANDIDATES] iter=29 pop=0 references=120 accepted=80 accepted_by_threshold=... forced_min_density=... attempts=... score_min=... score_max=...
[ITER DONE] iter=29 wall_sec=... cumulative_forced=... history=[...]
[CHECKPOINT] saved next_iter=30 path=...
```

`forced_min_density` 数量是最重要的算法偏差指标，需要在论文中报告；尽量不要把长期 100% 强制回退当成正常收敛。

## 6. 恢复与历史权重

完整检查点存放于新版本的 `BACAP_KDE_FIXED_checkpoints/<run_id>/checkpoint_full_latest.pt`。包括所有种群权重、历史模型、sigma、演化路径 p/s、统计值、下一迭代编号，以及随机状态；不依赖原作者只有权重的 `.pth` 文件。

```bash
python -u main_kde_fixed.py \
  --cases 200 --populations 5 --lambda-size 80 \
  --kde-r1 0.9975 --kde-r2 0.9990 \
  --kde-max-attempts 24 --kde-restart-attempts 64 \
  --full-checkpoint-every 10 --seed 42 \
  --resume BACAP_KDE_FIXED_checkpoints/0/checkpoint_full_latest.pt
```

**恢复时，所有检查点记录的训练与 KDE 参数必须相同。** 完整检查点是以训练轮结束为边界保存的；不会从中断的候选生成中间精确恢复。`--legacy-seeds` 模式因原程序每次从操作系统熵重新 `np.random.seed()`，仍然不能保证严格随机复现。

旧版训练保存的五个 `synced_model_i_55906.pth` 文件可以用来**热启动**新的修正版：

```bash
python -u main_kde_fixed.py \
  --warmstart-directory /path/to/BACAP_checkpoints/2 \
  --cases 200 --populations 5 --lambda-size 80
```

这只加载网络权重，历史 KDE、sigma、p/s、轮次均重新初始化，**不是**从原始第 29 轮继续训练。

## 7. 可选的 GPU KDE

考虑到小网络串行 Rollout 在 GPU 上可能反而更慢，默认 CPU Rollout + CPU KDE。若确认 CUDA 可用，可以仅将 KDE 放到第二张物理显卡：

```bash
CUDA_VISIBLE_DEVICES=1 python -u main_kde_fixed.py \
  --device cpu --kde-device cuda:0 \
  --cases 200 --populations 5 --lambda-size 80
```

CUDA KDE 的性能收益与数值误差必须单独实测。

## 验证范围

- 已在 Python 3 / PyTorch CPU 模拟网络与模拟环境中完成：KDE 归一化、最小密度选择一致性、有限种群重启、三轮完整训练、完整检查点、中途检查点恢复。
- 尚未在用户 Ubuntu + RTX 3090 + Linux C++ 港口环境和 200 个真实训练实例上执行端到端验证。
- 作者原实现的 KDE 阈值以及原训练奖励不能直接与此版本视为同一算法；建议保留独立版本标识和对照实验。
