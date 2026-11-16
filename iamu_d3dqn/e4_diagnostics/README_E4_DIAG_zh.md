# IAMU E4-D：Keep 策略塌缩定位与 Q-gap 反事实校准

**版本定位：** 2026-11-16 E4 训练充分性消融的独立诊断。补丁仅新增 `iamu_d3dqn/e4_diagnostics/`，不改已有 E4 训练器、E3 代码、网络权重或实验结果；默认严格复现原始 E4 的 `IAMUEventEnv`、验证情景与候选屏蔽设置。

## 0. 为什么需要这个实验？

GitHub 结果表明：

- `v10_e4_b900_seed43`：Episode 100 的验证等待为 1929.17；Episode 900 为 1971.95。在 B=10 时，Episode 900 平均只消耗 0.44 次改约，几乎退回完全 Keep（同条件 Keep 等待 1973.19）。
- `v10_e4_d900_seed44`：Episode 900、B=10 的平均改约为 0.78，等待 1971.89，也接近完全 Keep。
- `v10_e4_c900_seed43`：降低学习率后 Episode 900 的 B=10 平均改约仍只有 0.28 次，说明简单降低学习率不保证解决问题。
- E4 不同独立训练的同种子 Episode 100 也可能产生明显不同的验证结果，因此**尽量优先比较同一条训练运行轨迹中的早期/晚期检查点**，不能把不同独立训练当作“只延长训练”构成的严格因果配对。

需要诊断：后期 Keep 是**物理/预算约束和候选掩码**所迫，**固定 Keep margin** 所致，还是**Q 网络本身低估 Edit 的相对价值**？即使模型经常 Keep，这种行为是否真的遗漏了有利改约？

## 1. 科学依据与可检验假设

原 `DoubleDQNAgent.act` 对 Keep 的打分附加固定阈值 `keep_margin=0.1`，Keep 胜过 Edit 的条件为

\[
\max_{a\in\mathcal A_t^{\rm Edit}}[Q(s_t,a)-Q(s_t,\mathrm{Keep})]\le 0.1.
\]

**仅在存在合法 Edit 动作时该式才有意义。** `IAMUEventEnv` 还会在预算用尽、船舶已经开始服务、当前时刻不允许修改、或者 `candidate_screen=positive` 过滤时屏蔽 Edit。不能仅根据最终改约次数推断 Q 值塌缩。

因此每一个公开决策事件都按下面的判定分解：

| 现象 | 判定依据 | 主要可能原因 |
|---|---|---|
| 没有合法 Edit | `mask[1:].sum()==0` | 预算用尽、没有可移动候选、候选筛选等；需要看事件和筛选计数 |
| **仅 Keep margin 导致 Keep** | \(0<\max\Delta Q\le 0.1\) | 固定阈值抑制了本来略微正收益的候选 |
| **Q 本身偏好 Keep** | \(\max\Delta Q\le0\) | 可疑的 Q-gap 下移，须进一步用反事实收益核查 |
| 实际 Edit | \(\max\Delta Q>0.1\) | 网络愿意消耗预算 |

更进一步，必须在**相同公开状态**下比较 EP100/EP900，排除不同策略访问不同状态导致的混杂。

### 反事实标签的精确定义

固定某个公开状态 \(s\)、相同已生成的外生未来 \(\omega\)、相同的未来**续行策略** \(\pi_c\)，只改变当前第一个动作：

\[
G_{\mathrm{CF}}(s,a;\pi_c,\omega)=
W(\mathrm{Keep}\rightarrow\pi_c;\omega)
-W(\mathrm{Edit}_a\rightarrow\pi_c;\omega).
\]

\[
\Delta Q_\theta(s,a)=Q_\theta(s,a)-Q_\theta(s,\mathrm{Keep}),\qquad
 e_\theta(s,a)=\Delta Q_\theta(s,a)-\frac{G_{\rm CF}}{100}.
\]

`reward_scale=100` 是当前训练环境的原奖励缩放，必须从训练 manifest 读取；代码中不会把真实加权等待与归一化 Q-gap 混比。**正反事实收益**代表 Edit 较优。

该方法**不是对 Q 的无偏真值回归**：一个已实现的未来并不是条件期望；对未来 ETA 情景的条件抽样和真实数据校准仍需单独做。它适合用来检查值函数相对排序、漏改率、方向一致性及严重错误实例。使用 `anchor`（冻结 EP100 模型）和 `greedy` 两种固定续行策略检验结论的敏感性，不允许使用未来真实 ETA 选择后续动作。

### 四个具体假设

- **H1：掩码/预算解释**：晚期策略的低改约次数主要因为合法 Edit 比例低或预算提前用完。
- **H2：Keep margin 解释**：绝大部分被放弃的合法 Edit 的 Q-gap 落在 (0,0.1]，把阈值设为 0 会明显增加愿意 Edit 的事件数。
- **H3：Q-gap 塌缩解释**：晚期 EP900 在相同状态上对 Edit 的 Q-gap 系统性下移，从 >0.1 到 <=0.1；同时反事实正收益候选仍存在。
- **H4：Q-gap 校准不足**：预测的 \(\Delta Q\) 与配对反事实收益的缩放值方向不一致、MAE 较大，尤其在 B=2 和 Shock 下。

任何一个假设均可能被实验否定；当前不能单凭 Episode 900 接近 Keep 得出“Q-gap 塌缩已被证实”的结论。

## 2. 安装与依赖

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_E4_KeepCollapse_QGap_Diagnostic.zip
cd iamu_d3dqn
conda activate erldde
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s e4_diagnostics -p 'test_*.py' -v
```

**真实权重要求：** E4 的 `validation_snapshots/val_000100_grad_*.pt`、`val_000900_grad_*.pt` 等文件必须位于本地服务器的 `results/v10_e4_.../validation_snapshots/`。GitHub 上只有 ledger/CSV 而没有网络权重时，无法直接用 GitHub 结果文件替代本地 `.pt`。诊断器会严格核对 SHA256、ledger 状态、实际加载架构和 Keep margin。

### 真实环境冒烟（先执行）

```bash
python -u e4_diagnostics/diagnose_keep_qgap.py \
  --model results/v10_e4_b900_seed43@100 \
  --model results/v10_e4_b900_seed43@900 \
  --cases ../2024LiERLDDE/case_code/8.txt \
  --scenarios shock --eta-seeds 400001 --budgets 2 \
  --anchors early,keep --continuations anchor,greedy \
  --states-per-group 1 --max-edit-actions 4 \
  --out results/v10_e4_diag_smoke

python e4_diagnostics/summarize_diagnostic.py \
  --input results/v10_e4_diag_smoke
```

这个冒烟测试验证：与原 E4 Episode 100、900 的验证结果精确一致，早晚模型在同状态 Q 值可计算，Keep/Edit 反事实使用硬预算，输出字段与汇总报告正确。如果想重新运行，请换新的 `--out` 路径，程序不会覆盖旧结果。

## 3. 正式诊断：优先同一训练任务早晚检查点

建议至少比较：

| 模型 | 用途 |
|---|---|
| `results/v10_e4_b900_seed43@100` | 同一训练任务的早期参照 |
| `results/v10_e4_b900_seed43@900` | 严重 Keep 塌缩检查点 |
| `results/v10_e4_c900_seed43@900` | 低学习率训练的晚期表现 |
| `results/v10_e4_d900_seed44@900` | 另一训练种子的晚期近 Keep 检查点（跨种子比较仅作机制辅助） |

16 个 CPU 核可用时，建议**8 个独立 CPU 分片并行**，每个分片只允许一个 Torch/BLAS 线程。无需独占 4 块 GPU，因为全程是**冻结网络推理与物理仿真反事实回放**，不是重新训练：

```bash
python -u e4_diagnostics/run_parallel.py \
  --model results/v10_e4_b900_seed43@100 \
  --model results/v10_e4_b900_seed43@900 \
  --model results/v10_e4_c900_seed43@900 \
  --model results/v10_e4_d900_seed44@900 \
  --anchor-model results/v10_e4_b900_seed43@100 \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --anchors early,keep --continuations anchor,greedy \
  --states-per-group 2 --max-edit-actions 4 \
  --workers 8 --device cpu \
  --out results/v10_e4_diag_primary \
  2>&1 | tee v10_e4_diag_primary.log
```

先添加 `--dry-run` 可输出 54 个分片的命令和数量，而不创建结果目录。异常中断后使用**原配置**加 `--resume`，不能改科学参数后复用已有分片。只有全部分片通过后，才在 `results/v10_e4_diag_primary/merged/` 生成汇总数据。

如需强化**同一训练轨迹**的比较，另起一个独立实验，只使用 B900 Seed43 EP100、200、400、600、900，不与独立训练的模型混杂：

```bash
python -u e4_diagnostics/run_parallel.py \
  --model results/v10_e4_b900_seed43@100 \
  --model results/v10_e4_b900_seed43@200 \
  --model results/v10_e4_b900_seed43@400 \
  --model results/v10_e4_b900_seed43@600 \
  --model results/v10_e4_b900_seed43@900 \
  --scenarios shock,repeated,small --eta-seeds 400001:400003 \
  --budgets 2,5,10 --states-per-group 2 --workers 8 \
  --out results/v10_e4_diag_b900_seed43_epochs
```

## 4. 输出清单与如何判读

| 文件 | 用途 |
|---|---|
| `trajectory_episodes.csv` | 每个模型×验证轨迹的等待、预算、可改约状态数量、实际 Edit/Keep 数量 |
| `trajectory_events.csv` | 每次存在合法 Edit 或预算已耗尽的事件，Q_keep、Q_edit、最佳 Q-gap、阈值及 Keep 原因 |
| `trajectory_summary.csv` | 分模型、扰动、预算的 Keep 比例、平均改约和加权等待 |
| `state_bank.csv`、`coverage.csv` | 来自早期策略/Keep 的共同公开状态选择和合法状态覆盖率 |
| `q_values.csv` | **所有被比较模型在同一状态**的原始 Q_keep、Q_edit、Q-gap 及合法掩码 |
| `counterfactual.csv` | Keep/每个合法 Edit 在相同实现未来、相同续行策略下的完整加权等待、预算使用与增益 |
| `calibration_pairs.csv` | \(e=\Delta Q-G_{CF}/100\) 的逐 Edit 误差、方向和阈值判定 |
| `calibration_summary.csv` | MAE、RMSE、偏差、Pearson r、符号正确率、近 Keep margin 误差 |
| `decision_regret.csv`、`decision_summary.csv` | 首步后悔值、事后有利 Edit 的遗漏、事后有害 Edit、CVaR90 后悔 |
| `matched_q_shift.csv` | EP100 vs EP900 **同状态同候选**的 Q_keep 与 Q-gap 变化、过阈值反转与动作翻转 |
| `validation_parity.csv` | 每条轨迹与原 E4 `validation_details/val_XXXXXX/runs.csv` 的加权等待/预算逐条严格对齐 |
| `manifest.json`、`summary.json` | 实验配置、检查点路径和 SHA256、模型梯度步、案例哈希、可信度限制 |

生成自动可读诊断报告：

```bash
python -u e4_diagnostics/summarize_diagnostic.py \
  --input results/v10_e4_diag_primary/merged \
  --out results/v10_e4_diag_primary/diagnosis_report.md
```

### 应如何做出结论？

- 若**无合法候选**为主，或者 `final_screened_proposals` 明显偏高，应先看预算、动作掩码和 `candidate_screen=positive`，不得把它归咎于网络 Q。筛选机制改变后不能直接与原 V0.9 闭环成绩比较。
- 若大量合法事件属于**仅 Keep margin 导致 Keep**，可以后续单独做**不重新训练的 margin=0/0.05/0.1/0.15** 消融，但先不要改原始检查点设置。通过反事实确认 margin 是否避免有害 Edit。
- 若晚期的 \(\Delta Q\) 在**相同状态**显著下移，而 CF 有利 Edit 仍存在，支持 Q 值相对优势估计偏差或塌缩。再看 q_keep 和 q_edit 各自漂移，区分“Keep Q 被抬高”与“Edit Q 被压低”。
- 若两个后续策略给出的反事实增益符号大量冲突，则标签依赖续行策略，不能认定晚期 Q 一定错；应检验训练策略目标与真实执行策略是否匹配。
- 若两种续行下晚期 Q-gap MAE、漏改和首步 regret 都恶化，并且验证轨迹与原 E4 严格一致，就有较强依据指向 Q-gap 校准不足，而不是“模型还没多训练几轮”。

### 科研边界

1. **所用未来是模拟器已生成的外生未来，仅在事后反事实评价时可见；模型评分仍只使用当前公开观察**。事后选出“最佳首步动作”只用于衡量后悔值，不可部署为在线策略。
2. 一个固定状态只对应原实例的一条实现未来，因此不能直接计算 \(\mathbb E[G\mid s]\) 或严格置信区间；Case×ETA seed、预算和多状态之间高度相关。必须在不同 ETA seeds、独立测试 Case 10–14 和（必要时）独立未来情景校准下进一步确认。
3. **Q-gap 误差较大不必然等同于训练 bug**，部分误差来自未来单次随机实现、后续策略与被训练策略不同、gamma=1 的长时域回报估计、近似状态与函数泛化。
4. 固定状态检查是诊断机制；闭环加权等待和改约次数仍应使用原 E4 `runs.csv` 作为最终行为指标。不要通过本次反事实标签重新选择训练超参数，再声称 Case 8/9 是独立测试。

## 5. GitHub 上传结果

`iamu_d3dqn/results` 可能包含未跟踪或较大文件，请只强制添加分析用的 CSV/JSON/Markdown，不上传 `.pt` 大型权重：

```bash
cd ~/code/iamu
find iamu_d3dqn/results/v10_e4_diag_primary -type f \
  \( -name '*.csv' -o -name '*.json' -o -name '*.md' \) -print0 \
  | xargs -0 -r git add -f --
git add iamu_d3dqn/e4_diagnostics

git status --short
# 检查后，自行 commit/push
```

本补丁没有直接向 GitHub 写入任何文件，也没有在本地容器里运行真实 E4 服务器检查点。
