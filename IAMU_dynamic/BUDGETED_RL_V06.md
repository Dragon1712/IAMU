# IAMU-RL V0.6 — 事件驱动、硬预算、候选条件 Double DQN

> **研究原型，并非已证明优于 V0.4/Periodic 的最终算法。** 原有 `iamu_dynamic/core.py`、`risk_selection.py`、`validate_eta_forecast.py` 和 ERL-DDE 权重均未修改。

## 1. 已发表论文与本实现之间的关系

1. Carrara et al., **Budgeted Reinforcement Learning in Continuous State Space**, NeurIPS 2019, https://papers.neurips.cc/paper_files/paper/2019/hash/4fe5149039b52765bde64beb9f674940-Abstract.html 。借鉴 **预算条件化的状态/价值评估思想**；本实现使用明确的整数预算、每条轨迹硬掩码，并**不是**原论文的 Budgeted Fitted-Q 算法复现。
2. van Hasselt, Guez & Silver, **Deep Reinforcement Learning with Double Q-learning**, AAAI 2016, https://ojs.aaai.org/index.php/AAAI/article/view/10295 。训练目标由在线 Q 网络选择下一步动作、冻结目标网络计算该动作的目标价值，以缓解 Q 值过估计。
3. Bacon, Harb & Precup, **The Option-Critic Architecture**, AAAI 2017, https://ojs.aaai.org/index.php/AAAI/article/view/10916 。将**决策时机、持续动作**作为后续长期研究的候选方向；本版**未实现** Option-Critic。

模型不是 PPO、Constrained Decision Transformer，也没有使用任何已发表算法的训练权重。使用 PyTorch 从零训练一个候选条件 Q 网络，作为可重复的第一阶段消融基线。

## 2. 结构与科学约束

```text
Scenario (public ETA notices + simulator-only hidden true arrivals)
    -> IAMUEventEnv.reset()/step(action)
    -> observations (公开 ETA、历史 ETA 更新、已到港/已开工状态、剩余预算)
    -> candidates (原始 SlotPlanner.single_move，可行性已验)
    -> action mask: Keep or accept ONE Single candidate
    -> CandidateQNet Q(observation, candidate)
    -> Double DQN (experience replay + separate target network)
    -> real physical dispatcher; event-wise waiting cost
```

**动作索引**：`0=Keep`，`1..K=按当前规则排序的候选单船修改`；预算不足、已经开工船舶或不可行安排不会出现在动作掩码中。第一阶段故意将每事件最大改约数固定为 1，避免把 Joint 修复的贡献与何时改约混在一起。

**回报守恒**：每艘船实际开始作业时支付一次 `-weight * max(0, actual_start - actual_arrival) / reward_scale`。在完整事件轨迹上：

`sum(reward) * reward_scale == -metrics.weighted_wait`

请使用 `gamma=1.0` 作为未折扣的首轮基准，以免学习目标偏离原有全周期加权等待。其他 `gamma` 可作消融，但不再严格等价于最终等待目标。

**因果可观测**：网络**不接收** `true_etas`、未来 ETA 事件、合成 `scenario.label` 和随机种子。已到港状态只在真实时钟已经达到该船的实际到港时间时公开，属于可见的历史事实而非未来真值。训练用真实到港信息仅参与物理执行与回报。未来事件列表保留在仿真环境内部，不能作为网络输入。第一次 `reset()` 暴露的是 t=0 的只可 Keep 状态，不允许 RL 利用额外的 t=0 改约机会。

**事件时钟**：`--clock union` 在 ETA 通知和公开周期时钟上都允许决策，同时为 Greedy、Periodic、RL 提供相同的观察事件；Periodic 只在周期时钟上改约。`--clock eta` 仅允许 ETA 更新后决策，能严格复现 `simulate(policy='budget_aware', max_edits_per_event=1)` 的确定性贪心。**这两种时钟模式所得结果不能不加说明直接合并。**

**动作空间**：参数共享的 Q(s,a)，候选数量可变，非法索引打掩码。首次版本不包括双船 Joint，不承诺岸电/碳排放优化，也没有接入训练好的 ERL-DDE/ES 提示网络；`SlotPlanner` 仍是 V0.4 的贪心初始与单船修复优化器。接入冻结 ERL-DDE 是独立后续工作。

## 3. 环境安装和测试

无需额外 Gymnasium；依赖原项目及 `numpy`, `torch`。

```bash
cd ~/code/iamu/IAMU_dynamic
python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'
python -m unittest discover -s tests -v
```

在原项目上运行应保留所有旧实验，新增文件之间独立。部署时不要覆盖 `results/v02*`～`v05*`。

## 4. 极小的端到端试验（只检查执行链，不得用于论文性能声明）

随本源码有三个仅 6 船的独立演示 TSV，便于验证代码能结束完整训练和推断：

```bash
python -u train_budgeted_ddqn.py \
  --train-cases examples/rl_smoke_train.tsv \
  --val-cases examples/rl_smoke_val.tsv \
  --episodes 8 \
  --scenarios shock,repeated --budgets 2,5 --uncertainties 8 \
  --val-seeds 400001:400002 \
  --clock union --candidate-limit 3 \
  --batch-size 16 --warmup 32 --target-update 20 --eval-every 4 \
  --out results/v06_smoke_train

python -u evaluate_budgeted_ddqn.py \
  --checkpoint results/v06_smoke_train/best.pt \
  --test-cases examples/rl_smoke_test.tsv \
  --seeds 700001:700002 \
  --scenarios shock,repeated --budgets 2,5 \
  --methods keep,greedy,periodic,expected,ddqn \
  --risk-samples 4 \
  --out results/v06_smoke_test
```

本地同配置已完成 8 个训练回合和 40 次小样本配对评估。**DDQN 在测试上仍有退化，不是成功的性能验证**。检查点选择只使用验证集；测试结果仅用于显示闭环功能，不应据此调参。

## 5. 正式研究原型（需要服务器 CPU/GPU 时间）

**新实例划分建议**：从未用于 V0.4/V0.5 研究的 case_code 中锁定互不重复的训练、验证和独立测试集。例如下面选择 3～7 做训练、8～9 做验证、10～11 做测试；如果你已使用过这些案例调参，请更换尚未看过的编号。随机种子也保持不交叉。必须保存 manifest 与最终结果。

**先进行 50～100 回合的真实尺寸功能试跑：**

```bash
python -u train_budgeted_ddqn.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 100 --scenarios shock,repeated,small \
  --budgets 2,5,10 --uncertainties 4,8,12 \
  --clock union --candidate-limit 4 --period 12 \
  --val-seeds 400001:400003 \
  --batch-size 64 --warmup 512 --eval-every 25 \
  --out results/v06_ddqn_100
```

**再用相同实验协议开展较长训练（只要学习曲线与验证性能支持）：**

```bash
python -u train_budgeted_ddqn.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 2000 --scenarios shock,repeated,small \
  --budgets 2,5,10 --uncertainties 4,8,12 \
  --clock union --candidate-limit 4 --period 12 \
  --val-seeds 400001:400005 \
  --batch-size 64 --warmup 512 --eval-every 100 \
  --out results/v06_ddqn_long
```

**最佳 checkpoint 锁定后才进行独立测试：**

```bash
python -u evaluate_budgeted_ddqn.py \
  --checkpoint results/v06_ddqn_long/best.pt \
  --test-cases ../2024LiERLDDE/case_code/{10,11}.txt \
  --seeds 700001:700020 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --methods keep,greedy,periodic,expected,ddqn \
  --risk-samples 16 --forecast-std 8 \
  --out results/v06_ddqn_heldout
```

评价脚本在相同原始场景、时钟、预算、候选上执行配对试验，记录 `weighted_wait`、No-Replan 改善、Period 参照改善、最差损失、经验 CVaR90、预算和完整在线耗时。

如果比较 Risk Lookahead，可以在 `--methods` 加上 `risk_lookahead`，但可能大幅增加运行时间。**只用验证集挑选模型权重/超参数，最终测试集必须封存；不可根据测试结果重新改参数。**

## 6. 初步开发验证

- 原版 40 项测试 + 新增 8 项测试：**48 项全部通过**。
- 对 demo-case 上 Shock/Repeated、预算 0/2/5：`greedy` 在 `eta` 时钟和 `periodic` 在 `union` 时钟均与原 `simulate(max_edits_per_event=1, candidate_limit=4)` 加权等待、修改次数一致。
- RL 动作掩码禁止预算越界、禁止已开工船舶再改约；真实总等待与逐步回报之和相等；同一可观测 ETA/物理前缀下改变未来真实到港不会改变网络输入；检查点采用 `weights_only=True` 加载。
- 6 船 demo 8 回合训练可以产生 `best.pt`，独立测试路径成功输出 `runs.csv` 和 `summary.csv`。DDQN 未表现出稳定性能优势，请勿将此短训当作收敛或论文结论。

## 7. 必须优先解决的现实瓶颈与研究局限

1. **50 船训练速度**：当前 `SlotPlanner.single_move` 的全计划验证与大量位置组合很贵。在本地 CPU 上，12 回合的 50 船含验证试跑未能在限时窗口内完成。大规模训练前应先记录每回合耗时、减少候选数或优化可行性检查；不能承诺 2,000 回合的完成时间。
2. **状态是人工汇总的部分可观测特征**，未引入 GRU，也没有预测误差分布模型。RL 可能因观测缺失学不到真正的长期策略；这是后续历史状态编码消融的问题。
3. **两层调度不同**：外层 RL 学改约，内层仍是贪心 SlotPlanner；当前没有将已训练 ERL-DDE checkpoint 变成实时排序/位置提示器。不要称为端到端 ERL-DDE/RL 联合模型。
4. **风险目标**：当前训练直接最大化负的真实加权等待。CVaR 只用于评估，未写入 TD 损失；之后可研究分位数/风险敏感 RL，不能据此自称已实现 Risk-CVaR RL。
5. **数据分布**：合成 Shock/Repeated/Small，不等于真实港口 ETA，V0.5 已证实 Shock/强扰动下固定高斯预报失校准。首版 Model-Free RL 不把这个概率模型当作必需输入。
6. **公平比较**：原 Periodic 单独周期时钟，V0.6 在所有策略共用 `union` 观察事件时做比较，避免 RL 获得额外未知未来信息。所有基线每次也只准改 1 船，和 V0.4 默认 2 船配置**不是同一个实验条件**。
7. **终端完整回报**：所有等待以每艘船第一次开始作业计费，已经开始的船无法通过后续改约被追溯地提前。默认不折扣对应真实累计成本。

**下一步推荐消融**：`Keep+Single DDQN` vs Greedy/Expected/Periodic → 预算 B=0/2/5/10 → ETA-only vs Union → 当前摘要状态 vs GRU 历史状态 → 候选数/计算成本 → Joint 双船动作 → 冻结 ERL-DDE/岸电/碳排放模型。

### 真实 50 船案例的入口检查

使用 `examples/github_original_case0.txt` 训练 1 个 Shock 回合，`fixtures/case1.txt` 进行 1 个不同种子的验证回合，`--candidate-limit 2 --clock eta --budget 2`，训练与验证顺利完成，检查点已生成。由于只有 101 个 transition，未达到 `warmup=128`，**本次梯度更新次数为 0**；这一检查只能证明真实尺寸环境和检查点流程可执行，不能作为强化学习训练成绩。将回放 warmup 降到合适设置、增加回合后才产生实际梯度更新。
