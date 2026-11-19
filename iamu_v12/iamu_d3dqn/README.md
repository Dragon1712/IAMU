# IAMU 三模型离线 Q 值诊断（DDQN-96 / DDQN-130 / CC-D3QN-96）

**与 GitHub `Dragon1712/IAMU` 中 `iamu_d3dqn/` 代码兼容。** 请将 `offline_q_diagnostics.py` 和 `test_offline_q_diagnostics.py` 复制到 `iamu_d3dqn/` 根目录，与 `train_budgeted_ddqn_parallel.py` 同级。

本工具不会训练模型、更新梯度或修改任何检查点。它只读取三个已经训练完成的 **冻结模型文件** `best.pt`，从原 IAMU 物理仿真器重建验证场景，计算各模型的 Q 值，并产生 CSV 诊断文件。

## 先检查权重是否存在

GitHub 仓库中目前有三个实验的 CSV、JSON，但未查到 `.pt` 权重文件。运行前请在训练服务器上执行：

```bash
cd ~/code/iamu/iamu_d3dqn
ls -lh results/v082_8workers_sync1_300/best.pt \
       results/v09a_ddqn130_seed42_300/best.pt \
       results/v09a_d3qn_seed42_300/best.pt
```

- 如果原 DDQN-96 只在旧目录，使用 `--ddqn96 ../IAMU_dynamic/results/v082_8workers_sync1_300/best.pt`。
- 如果 `best.pt` 丢失，但服务器上保留了 **对应最佳验证检查点**，也可以用 `--ddqn96 /path/to/validation_snapshots/val_000150_grad_000005123.pt`；DDQN-130 的最佳点为 Episode 75 / grad 2506；D3QN-96 为 Episode 100 / grad 3387。注意必须核对 `validation_curve.csv` 和快照内容，不能任意选择不同检查点。
- 不要使用 GitHub 的 `train_curve.csv` 或 `validation_curve.csv` 假装网络权重；它们不包含参数，无法计算 Q 值。

## 运行：完整验证集

```bash
cd ~/code/iamu/iamu_d3dqn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u offline_q_diagnostics.py \
  --ddqn96 results/v082_8workers_sync1_300/best.pt \
  --ddqn130 results/v09a_ddqn130_seed42_300/best.pt \
  --d3qn96 results/v09a_d3qn_seed42_300/best.pt \
  --manifest results/v09a_d3qn_seed42_300/manifest.json \
  --reference-policies keep,greedy,periodic \
  --device cpu \
  --out results/q_diag_3models_best_seed42 \
  2>&1 | tee q_diag_3models_best_seed42.log
```

默认直接复用 `manifest.json` 中的测试网格：Case 8/9 × `shock,repeated,small` × seed 400001–400003 × 预算 2/5/10，共 54 个配对实例。

每个实例使用三种 **共同参考轨迹**（Keep、Greedy、Periodic），并让三个网络各自额外运行一条**不探索、无训练更新**的策略轨迹。因此默认执行 54 × 6 = **324 条事件驱动仿真轨迹**。三个模型在同一个参考状态上的 Q 输出可直接比较动作差值与排序；不同策略自己走到的状态不能直接拿绝对 Q 值作公平比较。

可以先进行小规模自检：

```bash
python offline_q_diagnostics.py --max-groups 1 \
  --out results/q_diag_smoke
```

如果三个 `best.pt` 都不在默认位置，应按前述命令传入路径。

如果只想调查严重冲击和预算 2，可以使用：

```bash
python offline_q_diagnostics.py \
  --cases ../2024LiERLDDE/case_code/9.txt \
  --scenarios shock --seeds 400001:400003 --budgets 2 \
  --out results/q_diag_case9_shock_b2
```

不需要各模型实际执行自身轨迹时可增加 `--no-onpolicy`，此时仅计算共同参考状态的 Q 值和动作分歧，不生成 Monte Carlo 误差统计。

## 输出文件说明

| 输出文件 | 内容及用途 |
|---|---|
| `q_states.csv` | **每次决策×每个模型**：原始 Keep Q、最佳 Edit Q、Q-gap、网络会选的动作、实际执行动作、D3QN Value 和原始 Advantage |
| `action_q_values.csv` | **每个合法动作**的 Q、相对 Keep 的 Q 差、加了 Keep margin 的选择分数、公开 ETA 规划器点预测收益；D3QN 额外输出 raw/centered Advantage |
| `common_state_comparison.csv` | 参考轨迹中**同一个状态**对应三个模型的动作、Q-gap、是否一致 |
| `choice_agreement_summary.csv` | 总体动作一致率及**至少存在一个合法 Edit 时**的动作一致率；建议后者作为首要诊断 |
| `q_summary.csv` | 按网络、参考策略、情景、预算汇总可改约状态的 Q-gap 与 Edit 选择率，以及仅 Keep 合法的比例 |
| `episode_outcomes.csv` | 参考策略及三个网络自身策略的真实等待、改约次数及相对 Keep/Periodic 的收益和后悔值 |
| `performance_summary.csv` | 分预算/扰动及总计的平均等待、预算消耗和相对 Periodic 的 CVaR90 正向后悔值 |
| `onpolicy_mc_returns.csv` | 每个模型自身运行状态的 `Q(所执行动作)`、实际后续**累计奖励**和两者误差 |
| `q_calibration_summary.csv` | `Q - 实际回报` 的 Bias、MAE、RMSE、90 分位绝对误差；数值单位是训练用的 **缩放奖励** |
| `diagnostic_manifest.json` | 参数、案例网格、真实检查点路径、SHA256、架构、参数数量、所选择 Episode 等，便于论文复现 |

## 如何解释结果

### 1. 共同状态动作差异是最直接的架构诊断

同一组 `global_features/action_features/mask` 输入三个冻结网络。对每个网络计算：

`q_gap = max_legal_edit Q(s, Edit) - Q(s, Keep)`

当 `q_gap > keep_margin = 0.1` 时，模型通常会优先 Edit；等于阈值时还需考虑候选分数及 argmax 次序。若没有合法 Edit，则 `q_gap` 为空，不应误当作 0。

`choice_agreement_summary.csv` 分别提供含 Keep-only 状态的总体一致率，以及只在可执行 Edit 的状态上的一致率。**比较时优先看后者**，否则许多没有选择余地的状态会人为提高一致率。

### 2. D3QN Value/Advantage 不能被误读

D3QN 使用合法动作的 Masked Mean，所以 `Value` 代表**本网络预测的合法动作平均 Q 值**，不是理论最优状态价值。`adv_raw` 含有任意公共平移自由度，不宜跨状态或模型比较；分析时优先看 `adv_centered` 或 Q-gap。

### 3. `Q - Monte Carlo 回报` 只能做离线校准诊断

在网络自己的冻结策略轨迹上，计算从当前决策到终局的实际折算累计奖励 `G_t`（本实验 `gamma=1`）。定义：

`error_q_minus_mc = Q(s_t, a_t) - G_t`

均值为正表示相对这**一次真实到港扰动结果**的高估，负值表示低估；不等于已准确测得真实期望 Q，也不是动作的因果效应。随机 ETA、未观测未来扰动、对 Keep 的启发式偏置都应在解释时考虑。不要直接用模型内部 Q 值量级比较性能，应对比实际等待、后悔值和跨种子稳定性。

### 4. 没有“历史训练梯度”

本工具不重新训练。因此它不能恢复当时 Value/Advantage 梯度随训练变化的历史曲线；历史梯度需要在训练期间额外记录。这份工具只实现你要求的三个**已完成实验**的离线 Q 值与动作价值诊断。

## 测试

```bash
cd ~/code/iamu/iamu_d3dqn
python -m unittest test_offline_q_diagnostics -v
```

测试会创建三份**随机初始化的测试检查点**和两个小规模 TSV，验证 Q Mask、三个模型加载、参考策略、自己的运行、CSV 输出与有限回报计算。它不依赖已有 `best.pt`，也不构成真实港口模型性能证明。

本次交付在受限本地容器中通过基于相同 DDQN/D3QN 模型类、模拟环境接口的 2 项端到端冒烟测试；正式服务器上的真实 IAMU 动态仿真仍需执行上述 unittest 与全量诊断。
