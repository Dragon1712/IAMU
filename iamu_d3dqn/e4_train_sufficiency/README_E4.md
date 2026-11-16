# IAMU V1.0-E4 · 900 Episode 训练充分性消融

**状态：代码已交付并完成本地逻辑测试；并未连接服务器执行 GPU 训练。**

E4 用原 `iamu_d3dqn/train_budgeted_ddqn_parallel.py`，不改变 DDQN 模型、环境、奖励、候选动作或验证协议。所有实验都从**随机初始化**开始，不从 V0.9 的 `best.pt` 恢复。训练结果存入独立目录 `iamu_d3dqn/results/v10_e4_*`。

## 一、实验矩阵

每个条件均执行训练种子 **42、43、44**：

| 条件 | Episode | 学习率 | ε 从 1 到 0.05 的衰减周期 | 仅改变什么 |
|---|---:|---:|---:|---|
| `a300` | 300 | 3e-4 | 300 | 新跑 300 Episode 的公平对照 |
| `b900` | 900 | 3e-4 | 300 | 相对 a300，仅增加训练量 |
| `c900` | 900 | 1e-4 | 300 | 相对 b900，仅降低固定学习率 |
| `d900` | 900 | 3e-4 | 900 | 相对 b900，仅延长探索 |

**共 12 个独立训练任务（3 个种子 × 4 条件），其中 9 个 900 Episode。**

保持 V0.9 一致的参数包括：训练案例 Case 3–7、验证案例 Case 8/9、三种扰动、预算 B=2/5/10、ETA 训练 seeds 从 200000 起、验证 seeds 400001–400003、`set_ddqn`、`hidden_dim=96`、`keep_margin=.1`、`greedy_exploration_prob=.75`、`candidate_screen=positive`、`gamma=1`、`batch_size=64`、Replay 50k、target sync 250、`updates_per_step=.25`、`max_updates_per_episode=64`、`sync_every=1`。

**不把历史的 300-Episode best.pt 直接当成 a300 新控制实验**，因为历史实验每 25 Episode 验证一次，与新实验每 100 Episode 一次的选模机会不相同。它仍可以作为外部历史参照。

## 二、每 100 Episode 保存且正式验证模型

现有训练器自带同步冻结/验证逻辑： `--eval-every 100` 每经过 100 个 Episode 会生成

```text
results/v10_e4_b900_seed42/
  manifest.json
  train_curve.csv
  validation_curve.csv
  validation_ledger.json
  validation_snapshots/
    val_000100_grad_XXXXXXXXX.pt
    val_000200_grad_XXXXXXXXX.pt
    ...
    val_000900_grad_XXXXXXXXX.pt
  validation_details/val_000100/runs.csv
  ...
  best.pt
  latest.pt
  training_state.pt
  performance.json
```

**每个 900-Episode 任务应保留 9 个可评估 checkpoint；每个 300-Episode 任务应保留 3 个。**

每次验证由 Case 8/9 × 3 扰动 × 3 ETA seeds × 3 预算 × 4 策略 = **216 条闭环轨迹**组成。每个检查点含可加载网络权重及 `selected_episode/gradient_steps` 元信息，与 `validation_ledger.json` 和 `validation_curve.csv` 同步对应。`best.pt` 只是最低验证等待的单个模型，不能替代全部九个检查点。

训练器还保存 `training_state.pt`，包括 optimizer、replay 和 RNG，可用于原参数的真正中断恢复；`best.pt` 本身**不是**完整训练恢复文件。

## 三、安装

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_V10_E4_TrainingSufficiency.zip
cd iamu_d3dqn
conda activate erldde

python -m unittest discover -s e4_train_sufficiency -p 'test_e4.py' -v
```

检查原训练数据及 GPU：

```bash
ls ../2024LiERLDDE/case_code/{3,4,5,6,7,8,9}.txt
nvidia-smi
```

## 四、正式独立启动

**先核对命令（不启动训练）：**

```bash
python e4_train_sufficiency/launch_e4.py \
  --conditions a300,b900,c900,d900 \
  --seeds 42,43,44 \
  --devices cuda:0,cuda:1 \
  --actors 8 --validators 8 \
  --dry-run
```

**正式运行：**

```bash
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u e4_train_sufficiency/launch_e4.py \
  --conditions a300,b900,c900,d900 \
  --seeds 42,43,44 \
  --devices cuda:0,cuda:1 \
  --actors 8 --validators 8 \
  --validation-every 100 --recovery-every 25 \
  --prefix v10_e4 \
  2>&1 | tee logs/v10_e4_launcher.log
```

**CPU/GPU 分配：** 一块 GPU 同时最多启动一个 E4 训练进程；`--devices cuda:0,cuda:1` 同时执行两组不同的任务。每组训练分配 8 个 CPU actor / 8 个验证 worker，但这两个阶段按原训练器同步分时进行，**单个任务无需 16 个额外核心同时满载**。只有一张 GPU 则改为 `--devices cuda:0`；更多 GPU 可改 `cuda:0,cuda:1,cuda:2,...`。如果主机 CPU 不足，可统一调低 `--actors` 和 `--validators`，但建议同一轮所有实验保持一致。

建议在 tmux 或作业管理器中运行，以免终端断开结束进程：

```bash
tmux new -s iamu-e4
# 在 tmux 内运行上面的正式命令
```

**所有参数一致的中断恢复**：重试原命令并添加 `--resume`（包括原来的 `--devices` 映射；原始训练器将 device 纳入恢复配置摘要）。脚本会检查原计划与训练器源码 SHA；若修改了学习率、Episode、验证间隔或训练源代码，不允许复用旧训练目录。不能直接使用 `best.pt` 假装恢复 optimizer/replay。

## 五、核验每个100 Episode模型

```bash
python e4_train_sufficiency/inspect_e4.py \
  --prefix v10_e4 \
  --conditions a300,b900,c900,d900 \
  --seeds 42,43,44 \
  --load-test \
  --out results/v10_e4_inspection
```

生成：

- `results/v10_e4_inspection/checkpoint_inventory.csv`：**逐模型**检查 100/200/.../900 快照文件存在性、验证状态、验证轨迹数量、梯度步数、SHA256、网络加载状态及验证加权等待。
- `results/v10_e4_inspection/cross_seed_learning_curve.csv`：各条件、Episode 下跨训练种子的均值、标准差、最差/最好种子。
- `results/v10_e4_inspection/inspection_report.json`：不完整或失败的检查点及原因。

任何检查点不完整，脚本返回非零状态。不要混淆“写出 .pt 文件”和“验证全部 216 条轨迹成功”。

**查询单个网络：**

```python
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from pathlib import Path
folder = Path('results/v10_e4_b900_seed42/validation_snapshots')
checkpoint = next(folder.glob('val_000500_grad_*.pt'))
agent, meta = DoubleDQNAgent.from_checkpoint(str(checkpoint), device='cpu')
assert agent.q_network == 'set_ddqn'
print(checkpoint, meta.get('selected_episode'), meta.get('n_gradient_updates'))
```

上面的网络可以作为 `iamu_Interface` 的 E3-A 价值诊断的冻结 `--checkpoint` 输入，也可以作为正式评估系统的 agent，不需要进一步训练。

## 六、研究结论的判定

同时报告：

1. **同 Episode**：`a300` 对比 `b900` 的 Episode=100、200、300，检验长训练配置在共有阶段是否产生偏差；主要训练长度对照用 `b900` 的 Episode=300/600/900 与 `a300` 的 Episode=300。
2. **跨 Episode**：各 900 条件第 100、200、...、900 次评估的等待值、改约次数、最差场景表现及分预算 B=2/5/10；不要只看 `best.pt`。
3. **跨训练种子**：均值、离散度以及是否存在后期退化；三种子数量仍有限，不等于充分显著性证据。
4. **固定状态的 Q-gap 校准**：在同一批公开状态与后续策略下，把不同训练时刻的 Q-gap 与配对反事实收益比较，计算 MAE、符号一致率和 Keep/Edit 错判率。
5. **独立测试**：在使用 Case 8/9 选择训练超参数、快照规则后，对尚未被用于选择的 Case 10–14 做一次独立验证，避免多次在测试集上优化。

由于 `a300/b900` 在 Episode 300 以后拥有不同训练长度，新增的 600 个 Episode 也改变经验回放中的旧数据占比，所以实验检验的是**增加交互与梯度更新的总体效果**，不是完全隔离“训练步数”以外所有经验分布因素。

## 七、结果上传 GitHub

`results/` 目录可能被 `.gitignore` 忽略，建议仅上传用于分析的 CSV 和 manifest，不上传沉重的 `training_state.pt` 或所有模型权重：

```bash
cd ~/code/iamu
find iamu_d3dqn/results -maxdepth 3 -type f \
  \( -path '*v10_e4*validation_curve.csv' -o -path '*v10_e4*manifest.json' \
     -o -path '*v10_e4*performance.json' -o -path '*v10_e4*validation_ledger.json' \) \
  -print0 | xargs -0 -r git add -f --
git add iamu_d3dqn/e4_train_sufficiency
# 复查 git status --short 后再 commit / push
```

对于 E4 对比所需的 `validation_details` 分层 CSV 也可按需上传。不要无意提交模型权重、完整 optimizer 状态和大规模转移数据。
