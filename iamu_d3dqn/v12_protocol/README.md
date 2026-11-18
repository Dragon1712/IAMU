# IAMU V1.2 — 决策时刻价值学习（独立实验补丁）

**固定研究目标**：仅比较原始逐事件学习、预算耗尽尾部合并、全部强制 Keep 区间合并。**不修改** V1.1 物理环境、单船候选生成、状态特征、奖励计费、网络结构或动作空间。V1.2 只是新的训练协议，不承诺性能优于任何基线。

来源仓库：[Dragon1712/IAMU](https://github.com/Dragon1712/IAMU)，开发时固定基线 commit `2d96981a38434014f14e31be5641d422d28835f8`，已通过 GitHub connector 核对主分支指针。

**重要验收状态**：当前交付是 **可运行源码 + 通过纯逻辑测试的初版实现**。交付环境无法获取完整 Git checkout，也没有原训练服务器与全部物理模拟案例。已验证轨迹语义、目标公式和 Replay 顺序；真实环境集成测试、4-actor 8-episode smoke、GPU 性能、端到端恢复连续性仍需在你的服务器执行。**尚未完成 V1.2 科学实验，不能声称压缩有效。**

## 1. 代码位置

所有文件仅在 `iamu_d3dqn/v12_protocol/` 和 `iamu_d3dqn/tests_v12/` 下新增。V1.1 及以前的文件均不被修改。直接导入原 `iamu_dynamic.rl_env_v11` 和 `rl_ddqn_v11`。

| 文件 | 职责 |
| --- | --- |
| `transitions.py` | V1.1 Transition 外包装 `RawStep`、新 `MacroTransition`、prefix 总账 |
| `compress.py` | primitive / budget_tail / decision_epoch 语义、累计奖励、链一致性、复现审计 |
| `targets.py` | 独立目标 `reward+bootstrap_discount*Q_next` |
| `agent.py` | 原 Set-aware DDQN 网络 + V1.2 更新 + 专用模型检查点加载器 |
| `replay.py` | 最近 200 **完整 episode** 的窗口、Uniform/A1 分层、顺序和采样状态恢复 |
| `actors.py` | V1.1 原环境逐事件 actor，增加源/下一预算、事件 ID 和物理时间元数据 |
| `train.py` | 固定轮次多 CPU actor、有序消费、学习更新、验证、完整恢复、manifest |
| `validation.py` | 显式使用 V1.2 模型加载器，原 V1.1 完全相同物理测试协议和 Keep/Greedy/Periodic |
| `fixed_data.py` | 固定混合策略收集同一原始轨迹，A0–A3 等梯度预算离线机制筛查 |
| `analyze.py` | 配对强基线比较与尾部损失摘要 |
| `scripts/` | 8-episode smoke / 12组长训练的**手动**执行脚本 |
| `tests_v12/` | 数值、Replay、模型、恢复、真实环境单元测试 |

## 2. 四组实验，禁止更改混淆因素

| 组别 | Representation | Replay 采样 | 长期 Q 对照 |
| --- | --- | --- | --- |
| **A0** | primitive | uniform | 全部事件转移，gamma 单步 |
| **A1** | primitive | stratified（动作可选比例 25%） | 仅验证重要状态重采样是否足够 |
| **A2** | budget_tail | uniform | 保留预算>0的全部事件，把预算耗尽后的后续成本并入最后一个 Edit |
| **A3** | decision_epoch | uniform | 每个有合法 Edit 的状态形成锚点，主动 Keep 必须保留 |

A3 自然包含 A2 的预算耗尽尾部，不是互不相干的新方法。只有 `obs_mask[1:].any()` 为真才是 A3 锚点；不能用 Q-gap 或 margin 判断。压缩的是 Replay 转移，不跳过原始物理事件。没有任何 Edit 的 episode 可以生成零个 A3 宏转移，但仍正常计入真实等待成本。

训练主协议：`candidate_screen=positive`、`candidate_limit=4`、`union/12h`、`reward_scale=100`、`gamma=1`、`set_ddqn hidden_dim=96`、`execution_margin=bootstrap_margin=0`、训练案例 1–7、验证 8–9、每组 400 episode、4 actor、每轮 16 episode、每 100 episode 验证、学习率 `3e-4`、`batch_size=64`、target 每 250 **梯度步**同步、Replay 最新 200 个**完整 episode**、统一 `updates_per_raw_step=.25`，训练种子 42/43/44。零 margin 是控制实验协议，不预设为最优。

### 关键数学

从原始事件索引 `i` 到 `j`，定义：

\[
 R_{i:j}=\sum_{k=i}^{j-1}\gamma^{k-i}r_k,\quad
 \delta_{i:j}=\begin{cases}0,&\text{terminal}\\\gamma^{j-i},&\text{otherwise}\end{cases},\quad
 y=R_{i:j}+\delta_{i:j}Q_{\mathrm{target}}\bigl(s_j,\arg\max_a Q_{\mathrm{online}}(s_j,a)\bigr).
\]

无全局 gamma 的第二次乘法。未出现第一个合法 Edit 之前的前缀奖励**单独审计**，不错误加到第一个动作标签。`gamma` 基于原始事件步数，而非物理时间。

## 3. 安装（与历史实验隔离）

在原仓库已检出目标 commit 时：

```bash
cd ~/code/iamu
# 先确认：git rev-parse HEAD 应为 2d96981a38434014f14e31be5641d422d28835f8
git rev-parse HEAD
unzip -o /path/to/IAMU_V12_decision_epoch_patch.zip -d .
cd iamu_d3dqn
conda activate erldde
```

如你的主工作区已有改动，建议从指定提交创建单独 worktree（不覆盖已有实验）：

```bash
cd ~/code/iamu
git worktree add --detach ../iamu-v12-experiments 2d96981a38434014f14e31be5641d422d28835f8
unzip -o /path/to/IAMU_V12_decision_epoch_patch.zip -d ../iamu-v12-experiments
cd ../iamu-v12-experiments/iamu_d3dqn
conda activate erldde
```

**不要从 V1.1 `best.pt` 恢复完整 V1.2 训练**。训练从头初始化；模型检查点与完整训练恢复状态严格区分。

## 4. E0 正确性及 8-episode smoke（先做）

```bash
cd ~/code/iamu/iamu_d3dqn
python -m unittest discover -s tests_v12 -p 'test_*.py' -v
bash v12_protocol/scripts/run_v12_smoke.sh
```

如果使用 worktree，先在相应工作树的 `iamu_d3dqn` 下执行。smoke **分别**启动 A0/A1/A2/A3 的 8-episode、4-actor 小运行，训练案例1/2、验证案例8、减少 warmup 和 batch 以覆盖更新链路；这是工程 smoke，不能和 400-episode 的科学协议混比。

预期必须满足：

1. 对完整轨迹，`-100*sum(raw.reward)` 与 simulator 的 weighted_wait 一致；gamma=1 时 prefix+宏奖励等于原始总奖励。
2. A2 最后一次消耗预算的动作包含整个剩余成本；A3 包含主动 Keep，初始无决策前缀不进入首个动作回报。
3. `bootstrap_discount` 在终端为 0，非终端为 `gamma**原始步数`；测试 `gamma=.9` 得到 -7.94 / .729 / -11.4265。
4. Replay 保留完整 episode 窗口，恢复采样 RNG 和**准确索引顺序**。
5. 验证加载 V1.2 checkpoint，不使用 V1.1 默认加载器；硬预算不超支。
6. 至少进行一次 CPU 中断/恢复对照，并核对完整 round 后参数、采样和梯度的一致性。CPU 可追求强确定性；GPU 另报告非确定性来源。

若任何真实环境测试失败，**不要启动长训练**。

## 5. E1 固定数据实验：完全相同原始轨迹

在训练 Case 1–7 中采集一组 Keep/Greedy/均匀合法动作的预先固定混合策略轨迹：

```bash
python -m v12_protocol.fixed_data collect \
  --cases ../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt \
  --episodes 64 --scenarios shock,repeated,small --budgets 2,5,10 \
  --seed 42 --out results/v12_fixed_raw_seed42.pt
```

例如先只执行 A2，验证该表示可训练：

```bash
python -m v12_protocol.fixed_data train \
  --data results/v12_fixed_raw_seed42.pt --group A2 \
  --gradient-updates 500 --seed 42 \
  --val-cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --out results/v12_fixed_A2_seed42
```

之后用相同 `--data`、`--gradient-updates` 和 seed 分别运行 `A0/A1/A3`，输出 `conversion_audit.csv`、`train_curve.csv`、`final.pt`、`validation/runs.csv`、`manifest.json`。注意：固定数据表现只用于机制筛查，不等价于在线收益；64 episodes 是否足以支持全部方法、尤其 A3 的 Batch 必须以实际 Replay 数量为准，不足时**会明确失败而非静默采样替代**。

## 6. E2 400-episode 在线主实验（**不自动运行**）

只有 E0 验收完成后，才执行手动脚本：

```bash
DEVICE=cuda:0 bash v12_protocol/scripts/run_v12_main_matrix.sh
```

它按 `A0–A3 × 42,43,44` 串行运行 12 组，并使用全量案例/扰动/预算。建议先运行一组观察资源占用，再在不同 GPU 分配作业，不要让多个大规模验证任务挤占同一资源。

**手动单组 A2 完整命令**：

```bash
python -u -m v12_protocol.train \
  --representation budget_tail --sampling uniform \
  --train-cases ../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 400 --workers 4 --validation-workers 4 --round-size 16 \
  --eval-every 100 --seed 42 --device cuda:0 \
  --gamma 1 --execution-margin 0 --bootstrap-margin 0 \
  --batch-size 64 --replay-episodes 200 \
  --warmup-raw-steps 512 --warmup-actionable 64 \
  --updates-per-raw-step .25 --target-update 250 \
  --out results/v12_A2_seed42_400
```

恢复**完全相同配置**只添加 `--resume`：

```bash
# 将上一条命令及参数原样重用，只加 --resume。
# 不允许悄悄修改 episodes、gamma、模型、任务表等科学参数。
```

输出：`manifest.json`、`training_state.pt`（完整恢复）、`latest.pt`、`best.pt`（模型加载）、`train_curve.csv`、`validation_curve.csv`、`validation_details/val_*/runs.csv`、`complete.json`。Replay 统计包括原始环境步数、宏转移数、样本重用和 NumPy Buffer MiB。若超出 `--max-replay-numpy-mib` 将失败而不是默默调整窗口。

> **明确限制**：本版支持**相同任务表、相同 episodes 的断点恢复**。不实现“在原输出目录里直接扩增 episodes”的 fork 流程；需要扩展时必须创建新的配置与结果目录，不能把它伪装成严格 resume。

## 7. 分析

```bash
python -m v12_protocol.analyze --runs \
  A0_42=results/v12_A0_seed42_400/validation_details/val_000400/runs.csv \
  A1_42=results/v12_A1_seed42_400/validation_details/val_000400/runs.csv \
  A2_42=results/v12_A2_seed42_400/validation_details/val_000400/runs.csv \
  A3_42=results/v12_A3_seed42_400/validation_details/val_000400/runs.csv \
  --out results/v12_analysis_seed42
```

`analyze.py` 会根据相同 `case/scenario/ETA seed/budget` 配对 Keep、Greedy、Periodic 和 DDQN，计算 mean wait、mean spent、相对两条强基线的收益及 CVaR90 正后悔值。若要进行正式显著性分析，还需对训练种子、case和扰动轨迹相关性做分层重采样；不要把 216 条基线轨迹当作 216 个独立的 DDQN 配对样本。完整 54 个案例情景预算条件对应 54 条 DDQN 验证轨迹及 216 条四方法明细。

## 8. 已测试与尚未测试的界限

- **已测**：压缩语义、前缀/尾部、主动Keep、B0、零时距事件、gamma<1、Replay窗口/顺序/采样、数值TD目标（纯函数）。
- **隔离 mock 测试**：V1.2 Agent 更新与模型检查点往返使用了 V1.1 形状兼容的简化网络 stub；**不能替代真实 Set-aware DDQN。**
- **待服务器验证**：真实 V1.1 网络、真实港口 Case/环境、4actor smoke、模型恢复连续性、384+ 实验启动前工程验收。

如 A2/A3 表现有优势，再补普通 n-step 和 PER 对照；本轮不加入这些新因素。对于最终模型选择，Case 10–14 仅在确实未参与调参时使用，否则需另留独立测试集。
