# IAMU-RL V0.7：预算状态、公开 ETA 候选价值、可消融动作选择

> 研究原型：不宣称 V0.7 已经优于 Greedy/Periodic。以 GitHub `Dragon1712/IAMU` 的 V0.6 文件 blob 为补丁基线。**本版会改变 `GLOBAL_DIM` (19→24) 和 `ACTION_DIM` (17→24)，V0.6 `best.pt` 无法继续加载，必须重新训练。** 不修改 ERL-DDE/KDE 权重和 V0.2–V0.6 结果目录。

## 1. 修改依据：先把 V0.6 实测异常和研究假设分开

原始验证记录 `results/v06_ddqn_100/validation_diagnostics/` 是 Case 8/9 × Shock/Repeated/Small × Seeds 400001–400003 × B=2/5/10，共 216 次运行、每预算 18 组场景。最佳模型第 50 回合选出。

- DDQN 在 **54/54** 次验证中用满预算；最后改约时间（B=2/5/10）均值分别为 **4.88 / 12.44 / 20.89**；Greedy 为 **12.37 / 58.95 / 111.60**；Periodic 为 **102.67 / 161.33 / 222.67**。
- DDQN 相对 Keep 的平均加权等待改善为 **-263.97 / -441.43 / -657.83**（负值代表恶化），Greedy 为 **+34.95 / +76.12 / +80.19**。
- B=5 中 DDQN 在 **16/18** 个场景比 Keep 更差；一个 Small/Seed 400002 的恶化为 **1257.90**。这说明过早、无选择地消耗预算已成为真实问题，但不能直接从汇总指标确定“观测、算法还是训练量”各贡献多少。

对应的外部研究：

1. **Carrara et al. (NeurIPS 2019)**, *Budgeted Reinforcement Learning in Continuous State Space*，https://research.google/pubs/budgeted-reinforcement-learning-in-continuous-state-space/ 。论文研究预算条件化控制；本版本只借鉴把预算量纳入状态，**不是**复现 Budgeted Fitted-Q 或论文的 Budgeted Bellman 算法。预算始终是每条轨迹的整数硬上限。
2. **van Hasselt, Guez & Silver (AAAI 2016)**, *Deep Reinforcement Learning with Double Q-Learning*，https://ojs.aaai.org/index.php/AAAI/article/view/10295 。保留原 Double DQN 的在线网络选择、目标网络评估结构；新增 Keep 选择边际同时用于执行和下一动作选择。
3. **Huang & Ontañón (FLAIRS 2022)**, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*，https://doi.org/10.32473/flairs.v35i.130584 。说明动作掩码的重要性；该论文研究策略梯度，不是对本实现 Double DQN “预测正收益”掩码的正确性证明。物理可行性/剩余预算掩码为严格约束；预测收益筛选**仅是可选的启发式限制**。
4. **Hester et al. (AAAI 2018)**, *Deep Q-learning from Demonstrations*，https://ojs.aaai.org/index.php/AAAI/article/view/11757 。从示范数据加速学习的研究动机；本版本实现的是 **Greedy 引导的 ε 探索**，不包含原论文的监督大间隔损失、优先回放或完整 DQfD 训练，不能称作 DQfD 复现。

## 2. 具体代码变更

### A. 绝对预算状态

保留原 19 维状态，同时添加 5 维（总计 24 维）：

- 索引 19：`B / 10`（总预算的绝对尺度）
- 索引 20：`b_remaining / 10`（绝对剩余预算）
- 索引 21：候选动作数量 / `candidate_limit`
- 索引 22：距离上次修改的时间 / 公开实例时间尺度
- 索引 23：最佳候选公开 ETA 预测改善 / `reward_scale`

V0.6 的 `remaining/B` 与 `used/B` 仍保留在前 19 维中；额外的第 19、20 维解决了初始状态 B=2 与 B=10 比例相同而不可区分的问题。数值除以**跨回合固定**的 10，而不是只除以当前 B。改变 B=10 之外的尺度可以做后续归一化消融。

### B. 候选全局价值特征

保留原有 17 维候选特征，追加 7 维（合计 24 维），均由 **`RealizedRuntime` 当前已发生的公开状态 + `published_eta`** 预测：

1. 公开 ETA 下该候选全局等待改善 `score(current)-score(candidate)`；
2. 修改前的预测剩余总等待；
3. 修改后的预测剩余总等待；
4. 其他船舶的最大预测额外开始延迟；
5. 其他船舶预测开始被推迟的比例；
6. 其他船舶预测等待损失的正增量；
7. 其他船舶预测等待改善的正增量。

注意所有预测依然将公开点 ETA 当作未来抵港假设，**没有使用未来真实到港真值**，也不能保证其最差情形可靠。为了控制计算开销，每个事件先计算一次当前计划的公开 ETA 预测，再对每个候选分别计算一次，并在 Greedy 决策时复用缓存。

### C. 动作选择机制，均可单独关闭

- `--candidate-screen none|positive`：默认 `none`，`positive` 仅保留 `estimated_gain > --min-gain` 的物理可行候选。始终允许 Keep；仅屏蔽即时公开 ETA 预测不合理的动作。**这不能作为真实风险或 CVaR 安全保证。** 可能排除点预测为负但长期实际有益的改约，因此应与 `none` 做严格消融。
- `--keep-margin 0.1`：仅在 Q 选择时把 Keep 的分数提高 0.1（已按 `reward_scale=100` 归一的 Q 单位）。该 margin 同时用于 DDQN 的下一动作 argmax，以保持训练与推断的选择规则一致。不要把它作为真正的等待损失约束。
- `--greedy-exploration-prob 0.75`：只在 ε 探索分支中，以 75% 的概率采用**公开 ETA 贪心**示范动作（包括 Keep），其他探索仍均匀采样合法动作；测试/推断时 ε=0，不使用教师。它不是行为克隆损失，更不是原始 DQfD。
- `--budget-features legacy|absolute` 和 `--value-features legacy|global`：可将额外预算特征和额外价值特征分别置零，同时**保持网络输入维度和架构相同**，用于清晰消融。`value-features=legacy` 同时清零新增加的候选数、修改间隔及候选最佳收益 3 维全局特征。训练必须新建检查点；不能把这两个开关用于旧 V0.6 模型。

### D. 诊断输出

- 训练记录增加 `screened_proposals`、`last_revision_time`、`mean_estimated_edit_gain`。
- 评估 `runs.csv`、`summary.csv` 额外记录点预测收益与筛选次数。
- `evaluate_budgeted_ddqn.py --save-actions` 输出 `action_traces.csv`，含每个事件的时刻、触发类型、动作索引、修改船舶、预计收益和剩余预算。不会在轨迹特征中写入未来真实 ETA。

## 3. 安装并验证

先**备份现有进程和检查点**；本补丁修改 RL 相关 Python 源码，如果其他进程正在加载同一代码路径，不要覆盖这些进程依赖的工作目录。建议在独立 Git 分支工作：

```bash
cd ~/code/iamu
git switch -c experiment/iamu-rl-v07
git apply --check /你的上传路径/IAMU_budgeted_ddqn_v0.7.patch
git apply /你的上传路径/IAMU_budgeted_ddqn_v0.7.patch
cd IAMU_dynamic
python -m unittest discover -s tests -v
```

**已在本地执行** 61 项测试通过，包括原有 48 项和新增 13 项。V0.6 原有 `simulate()` 与新环境 Greedy 的等待一致性测试继续通过。新增的单元测试验证预算可区分、公开点预测评分与 `RealizedRuntime.score` 一致、未来真实 ETA 不渗入观察、筛选及 Keep 合法性、梯度/检查点恢复等。

## 4. 推荐机制消融（全部使用同一训练/验证种子和案例）

同样的神经网络架构，按以下四组逐层增加机制：

| ID | budget_features | value_features | screen | teacher_prob | keep_margin | 目标 |
|---|---|---|---|---|---|---|
| A | legacy | legacy | none | 0 | 0 | 旧信息输入的 DDQN 基线（**新版本从零训练**） |
| B | absolute | legacy | none | 0 | 0 | 仅检验绝对预算价值 |
| C | absolute | global | none | 0 | 0 | 再检验全局候选特征 |
| D | absolute | global | positive | 0.75 | 0.1 | 再检验筛选 + 教师 + Keep margin 的组合 |

要区分三个动作机制的独立贡献，还应在 D 附近分别关闭 `candidate-screen`、`teacher-prob` 和 `keep-margin`，不能声称 D-A 的全部增益都来自 RL 或某一个组件。`--epsilon-decay-episodes` 请与训练回合匹配，例如先使用 300 回合、衰减 300 回合，之后再研究探索曲线。

训练实例仍可使用 Case 3–7，验证 Case 8–9（这些只是开发验证集，已反复观察，**不得当独立测试集**）。选择好模型后，再用独立案例和种子做测试。

运行脚本 `bash run_v07_ablation.sh` 默认进行 4 组 **300 回合**训练，可能有较高 CPU 耗时，不会在后台自动运行；如先检查功能，可用 `EPISODES=30 EVAL_EVERY=15 bash run_v07_ablation.sh`。

```bash
cd ~/code/iamu/IAMU_dynamic
EPISODES=300 EVAL_EVERY=50 bash run_v07_ablation.sh
```

最佳权重分别保存在新目录 `results/v07_ablation_A` 到 `_D`。以下仅是**验证诊断**示例：

```bash
python -u diagnose_v07_validation.py \
  --checkpoint results/v07_ablation_D/best.pt \
  --methods keep,greedy,periodic,expected,ddqn \
  --risk-samples 16 --forecast-std 8 \
  --save-actions \
  --out results/v07_ablation_D/validation_diagnostics
```

`diagnose_v07_validation.py` 专门评估 checkpoint 记录的验证案例/随机种子，避免误用 `evaluate_budgeted_ddqn.py`（后者会主动拒绝测试集与验证集重叠）。

**锁定超参数和最佳检查点之后**，才在未用于开发的 Case 10、11 和新种子 700001:700020 上做一次独立测试（不要按该测试结果继续调参）：

```bash
python -u evaluate_budgeted_ddqn.py \
  --checkpoint results/v07_ablation_D/best.pt \
  --test-cases ../2024LiERLDDE/case_code/{10,11}.txt \
  --seeds 700001:700020 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --methods keep,greedy,periodic,expected,ddqn \
  --risk-samples 16 --forecast-std 8 \
  --out results/v07_heldout_D
```

## 5. 初步功能试验：不是性能证据

本地 6 船演示案例，12 回合，positive 筛选 + teacher_prob 0.75 + keep_margin 0.1：训练梯度累计 137 步；载入保存的权重后完成 40 条独立评估。短训 DDQN 在 8 个预算-场景评估中均 Keep（预算使用 0），结果等于 No-Replan，未产生退化，**但也没有学会有益改约**。该结果提醒：过强的 Keep margin 或过滤器可能让策略陷入保守，必须将 mean_spent、改约次数、平均收益和 CVaR 一起分析。

在另一个真实 50 船 Case 0 单例的程序检查中，ETA-only、B=2、候选 2 的 Greedy 不受 `candidate-screen=positive` 影响（等待均约为 4141.63），确认同一筛选器不会改变原 Greedy 正收益动作选择；这个例子本身也没有体现优越性。

## 6. 局限与注意

- 不使用新预测分布模型，也不引入 Shore-Power/减排目标；本阶段仅研究等待与有限预算改约。
- `positive` 是点预测的保守启发式，仍可能拒绝真正有用的长期动作、接受实际有害动作；**不提供真实风险安全保证**。
- 模型输出的 Q 值与公开点预测收益可能偏离，依靠更多训练回合并不能自动修复；应用 V0.7 前须在验证集检查逐事件动作轨迹。
- 由于现在要为候选计算公开 ETA 整体等待和连锁外部性，决策时可能增加计算时间；虽然缓存避免了 Greedy 的重复评分，仍须在完整 50 船环境报告从候选生成到动作执行的耗时。
- 扰动合成分布与现实港口不同；当前仍未引入真实轨迹或偏尾、状态条件 ETA 预测模型。V0.5 的合成预测校准发现不能直接转移为真实世界保证。
- 保存的 V0.6 `best.pt` 不可与 V0.7 输入维度兼容；请从零训练，不要覆盖旧检查点。
