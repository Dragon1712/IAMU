# IAMU V1.0-E3：预算前瞻价值校准 + 置信度自适应融合

> **交付范围：** 仅增量补丁 `iamu_Interface/`，不覆盖 V0.9、V1.0 E1/E2 的代码与 `results/`。**本交付是可执行实验代码，不是已经在用户服务器上跑完的新结果。**
>
> **实验定位：** 冻结 Set-aware DDQN 的 42、43、44 三个训练种子，不再训练模型；统一使用原来的港口仿真、动作掩码、SingleMove 候选和每船一次预算扣减；不新增不可观测未来特征。Case 8/9 仅用于机制探索，**Case 10–14（若尚未被使用）保留给最终独立验证，不能用于挑选超参数**。

## 1. 背景与待证伪假设

E2 三种子结果表明：固定混合 `q_joint` 的平均改善约 +3.59 加权等待，但 B=2 的平均退化约 9.70，个别 Shock 负收益超过 275；固定 H=2 推理计算量比冻结 DDQN 大很多。MPC 样本仅 8，需核查这些损失究竟源自**预测不准**还是**融合不准**。

- **H1（预测质量）**：H=2 的多阶段 MPC 预测的 Keep–Edit 收益比 H=0 更接近独立模拟的后续行动收益。注意 MPC 使用后续贪心单船修复，反事实评估的后续策略使用冻结 Set-aware DDQN，存在**续行策略差异**，须独立报告。
- **H2（融合质量）**：在完全相同的 Q、MPC 样本和 CVaR 风险估计下，置信度自适应融合优于固定权重融合（需要实测，不能先验认定）。
- **H3（机制）**：收益对 Q/MPC 方向一致、预算紧张程度、MPC 估计区间宽度具有可解释的异质性；尤其关注 B=2、Shock。

## 2. 数学方法（仅作为待检验方法）

对每个当前合法 Edit `a`，用相同 `M` 个抽样 ETA 场景分别模拟 Keep 与 Edit 后的多阶段修复；Keep 后剩余预算 `b`，Edit 后剩余预算 `b-1`。每个样本上的收益（在 DDQN 回报尺度）为：

\[
D_a^{(m)}=\frac{C_H(\operatorname{Keep},b;\xi_m)-C_H(a,b-1;\xi_m)}{R}, \quad R=\texttt{reward_scale}=100.
\]

\[
\widehat A_a=\frac1M\sum_m D_a^{(m)}, \qquad
SE_a=\frac{\operatorname{sd}(D_a)}{\sqrt M}.
\]

固定权重 E2 对照为

\[
S_{\rm fixed}(a)=(1-\beta_B)\Delta Q_a+\beta_B\widehat A_a
-\beta_R\frac{\Delta\operatorname{CVaR}_{0.9}(a)}{R}.
\]

E3 **预注册的启发式置信度门控**为：

\[
\begin{aligned}
 h_a &= t_{0.975,M-1}SE_a,\\
 c_a &= \max\Big(0,\min\big(1,1-\frac{h_a}{\max(|\widehat A_a|,10^{-12})}\big)\Big),\\
 d_a &= \mathbf 1[\Delta Q_a\cdot\widehat A_a\ge0],\\
 w_a &= \beta_B c_a d_a,\\
 S_{\rm adaptive}(a)&=(1-w_a)\Delta Q_a+w_a\widehat A_a
 -\beta_R\frac{\Delta\operatorname{CVaR}_{0.9}(a)}{R}.
\end{aligned}
\]

`c_a` 在置信区间覆盖 0 时变为 0；若 Q 与 MPC 明显反向，则 `d_a=0`，退回冻结 Q 的该动作价值。这只是**不确定性门控的可消融假设**。小样本、偏斜、重尾和不充分校准会使 Student-t 近似区间不具备严格覆盖保证。`--neutral-q` 默认 0，可设置允许接近零的信号不算冲突；该参数必须在验证集选定后冻结。

**行动规则：** 默认保留原冻结 DDQN 的 Keep margin；纯 MPC 使用 `min_gain / reward_scale` 作为物理改约阈值。所有方法受原动作掩码和硬预算约束。

### 控制组（同一环境/估计器）

| 模式 | 评分 | 用途 |
|---|---|---|
| `q_only` | 原始 Set-aware Q + Keep margin | V0.9 基线（不得变更） |
| `q_risk_only` | Q + 同一 CVaR 项 | 将风险修正与预算融合分开 |
| `pure_mpc` | 仅 MPC 期望增益、无 Q、无 CVaR | 严格纯 MPC 对照 |
| `pure_mpc_risk` | MPC 增益 + 同一 CVaR 项 | 与固定、自适应融合的风险匹配控制 |
| `q_fixed` | E2 固定 Q/MPC 混合 + 风险 | 主要对照，应与 E2 `q_joint` 逐轨迹一致 |
| `q_confidence` | 仅 `c_a` 门控 Q/MPC | 分离方差影响 |
| `q_agreement` | 仅 `d_a` 门控 Q/MPC | 分离 Q/MPC 冲突影响 |
| `q_adaptive` | `c_a d_a` 双门控 + 风险 | 被检验的完整新方法 |

**关键公平性：** 上述策略在**相同状态**调用相同的 `RiskSpecification`、相同事件种子生成的 ETA 场景、相同的 MPC 执行逻辑、相同的 CVaR 估计。运行闭环后各策略状态轨迹可能分叉，后续状态无法强求相同，但外生原始 ETA 实现配对一致。

## 3. 两条严格分离的实验链

### E3-A：预测收益本身是否可靠？

入口 `experiments/eval_e3_value_calibration_v10.py`：

1. 沿**冻结 V0.9 策略**推进，选取公开状态；`--selection near_margin` 优先挑选 Q-gap 接近 Keep margin 的状态（只使用当前 Q 值），也可 `first`。每组记录全部符合筛选条件的数量，不会因为某个分片无候选而误报模型崩溃。
2. 在该状态下，对 Keep / 每个 Edit 用在线 M=8 场景估计 `H=2` MPC 增益与配对标准误；同一 M 个样本做 H=0 消融。**统一固定该估计**，不根据未来结果改变动作评分。
3. 用**独立抽样 M=64** 的场景、相同后续 MPC 贪心修复取得 `mpc_reference_advantage`：其与在线估计的差用于检验 **Monte Carlo 数值误差**，不是实际港口的真值。
4. 用**额外 8 个独立模拟未来世界**做固定状态反事实：当前公开历史保持不变，未来 ETA 通知在预先固定的 12h、24h 时刻逐步释放；未来物理到港由独立隐变量决定；Keep 与 Edit 分支使用**相同世界与相同冻结 DDQN 后续策略**。这里得到 `synthetic_frozen_advantage`，用于检验**模拟预测价值与一种已定义后续策略的可用收益差距**。通知生成模型与真实港口分布不完全相同，因此仅称 *model-based future benchmark*。
5. 再在原始生成案例真正实现的未来通知下，用**同样的冻结 DDQN 续行策略**完成 Keep/Edit 的配对反事实，得到 `realized_frozen_advantage`。**这只是一条真实实现的事后观察，不是未来期望的无偏标签**。
6. 最后在**完全相同**的 MPC 估计与以上两类反事实上，离线比较 8 种融合规则的选择、相对可选动作最佳收益的 regret。这样预测误差和融合规则选择错误可以分开度量。

输出：

| 文件 | 内容 |
|---|---|
| `details.csv` | 每个状态-Edit 的 Q-gap、MPC 均值、配对 SE、近似区间、独立大 M 误差、H=0/H=2 对照、模拟未来与原始未来反事实收益 |
| `fixed_state_policy.csv` | **同一状态、同一 MPC 估计、同一反事实世界**下的 8 种融合动作和 oracle regret |
| `summary.csv` | 分扰动误差、区间覆盖、H=0 相比 H=2 的价值预测误差 |
| `fixed_state_summary.csv` | 同一状态的离线融合规则对照 |
| `coverage.csv` | 合法状态数、抽中的状态数，避免“仅一个候选时排序一致率 100%”误解 |
| `empty_group.json` | 无可比较动作的分片被记录为正常空样本，不伪造数据 |

**重要：** 在线预测器的场景数据**不使用真实未来到港值、真实未来 ETA 通知或事后反事实结果**。只有计算参考标签的独立评估流程能访问未来仿真隐变量。

### E3-B：相同估计下闭环融合是否有效？

入口 `experiments/eval_e3_fusion_v10.py`：

使用统一的 Set-aware 冻结检查点和 `IAMUEventEnv` 比较 8 种策略；`q_fixed` 应与 E2 原 `q_joint` 精确复现，`q_only` 应与 V0.9 `set_ddqn` 精确复现。

输出 `runs.csv / paired_summary.csv / manifest.json`（逐组写 `runs.partial.csv`），额外记录实际改约数、MPC 时间、风险时间、Q/MPC 冲突率、不可信区间占比、有效融合权重及相对纯 Q 动作变化次数。`--audit` 可输出逐状态与逐候选明细，主要用于重点诊断 Case 8 Shock B=2 的严重退化。

**请同时比较** `q_adaptive` vs `q_fixed`、`q_risk_only` vs `q_fixed`、`pure_mpc_risk` vs `q_fixed`。如果置信度规则仅减少 MPC 权重而没有改善损失尾部，也可能只是返回 Q 基线，应统计门控频率。

## 4. 安装（不覆盖已完成实验）

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_Interface_V10_E3.zip
cd iamu_Interface
conda activate erldde

# 原来已完成 setup_interface.py 即可跳过；不要用 --force 覆盖冻结快照。
test -f iamu_dynamic/core.py || python setup_interface.py --source ../iamu_d3dqn --copy-assets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 5. 冒烟与回归验证（必须先完成）

```bash
python -m unittest discover -s tests -p 'test_v10_e3*.py' -v
bash scripts/run_e3_smoke.sh
```

本机已验证 11 项与环境无关的评分测试、Python 语法及报告脚本；由于交付环境没有服务器上的 `best.pt`、底层 V0.9 物理仿真快照，**不能声称真实环境测试已经通过**。服务器的集成测试与基线一致性检查才是关键验收条件。

## 6. E3-A 与 E3-B 运行命令

### E3-A：固定状态前瞻价值校准

```bash
bash scripts/run_e3_calibration_validation.sh
```

54 组、每组最多 2 个近边界状态，8 个独立模拟未来世界，64 样本独立 MPC 参考。如果计算压力过大，可先运行仅一组：

```bash
python -u experiments/eval_e3_value_calibration_v10.py \
  --smoke --states-per-group 1 --reference-samples 32 \
  --future-worlds 4 --risk-samples 16 --mpc-samples 8 \
  --out results/v10_e3_calibration_precheck
```

**多候选诊断（独立、不可与原训练环境的闭环成绩混作同一基线）：**

```bash
python -u experiments/eval_e3_value_calibration_v10.py \
  --states-per-group 2 --min-candidates 2 \
  --candidate-screen none --reference-samples 64 --future-worlds 8 \
  --out results/v10_e3_calibration_multicandidate
```

### E3-B：全量闭环融合

```bash
bash scripts/run_e3_fusion_validation.sh
```

可与 V1.0-E2 逐轨迹比较：

```bash
python scripts/check_e3_e2_parity.py \
  --e2 results/v10_e2_multiseed_42/runs.csv \
  --e3 results/v10_e3_fusion_seed42/runs.csv
```

仅在 `H=2`、`risk-samples=16`、`mpc-samples=8`、`beta-budget=0.25`、`beta-risk=0.25`、相同模型/场景/候选协议时才应期待 `q_fixed` 对应 E2 的 `q_joint` 完全一致；冒烟测试使用更少场景，不可混比。

### 三训练种子及并行执行（建议优先使用）

```bash
python -u scripts/run_e3_parallel.py \
  --workflow fusion --training-seeds 42,43,44 \
  --workers 8 --mpc-samples 8 --risk-samples 16 \
  --out results/v10_e3_parallel_fusion

python -u scripts/run_e3_parallel.py \
  --workflow calibration --training-seeds 42 \
  --workers 8 --states-per-group 1 \
  --reference-samples 64 --future-worlds 4 \
  --out results/v10_e3_parallel_calibration
```

遇到意外中断，用原参数添加 `--resume`，每组分片独立保留日志；不能更改科学参数后复用旧分片。**部分失败时不会生成伪装完成的合并结果**。

并行 E3-B 产出在：`results/v10_e3_parallel_fusion/merged/fusion/train42/runs.csv` 等目录。

## 7. 结果分析与选择标准

顺序运行时：

```bash
python -u experiments/analyze_e3_v10.py \
  --fusion results/v10_e3_fusion_seed42 \
           results/v10_e3_fusion_seed43 \
           results/v10_e3_fusion_seed44 \
  --value-calibration results/v10_e3_calibration_seed42 \
  --out results/v10_e3_analysis
```

并行合并时将 `--fusion` 路径替换为 `.../merged/fusion/train42`、`train43`、`train44`。统计脚本使用训练种子和 Case×ETA Seed 分层的探索性配对 Bootstrap，区分等待、改约次数和 CVaR90 正后悔值。只要区间跨零，就不能宣称稳定改善；即使不跨零，三训练种子和验证集重复调参也不足以证明独立泛化。

**必要验收标准：**

1. `q_only` 逐轨迹等于 V0.9 原始模型；`q_fixed` 等于 E2 同配置 `q_joint`；所有改约次数不超过硬预算。
2. E3-A 报告 MPC 相对独立高样本参考的误差与 H0/H2 差异，另报后续冻结策略下的模拟/真实反事实误差，**不能把它们混成一个 MAE**。
3. E3-B 所有融合模式使用相同的抽样和评价规则，比较 `q_adaptive` 相对 `q_fixed` 的整体、B=2、Shock 和 CVaR90 regret，不仅看均值。
4. `--min-candidates 2` 的多候选诊断至少应有足够的合法状态；如果无，报告缺失覆盖，不得给出虚假的候选排序稳定性结论。
5. 记录总耗时、MPC 耗时与有效权重；若自适应只回退原 Q 而没有改善严重损失，则不应把门控方法作为已验证优于固定权重的模型。

## 8. 后续机制分析依据

- **公共随机数/配对估计：** 使用同一个未来 ETA 场景对 Keep 与 Edit 求成本差，减少共同外生扰动造成的估计方差；SE 从**配对差值**计算，而非单独对 Keep/Edit 成本估 SE 后相加。
- **多阶段预算：** 复用 E2 的 `evaluate_multistage`，每个备选动作有独立的剩余预算与分支状态，未来 ETA 信息只在模拟未来阶段公布。
- **不确定性收缩：** 采用近似 Student-t 区间是否覆盖零作为可解释门控；是用于实验的启发式而非真正置信校准或严格安全保证。
- **反事实同场景：** 同一固定公开状态、同一潜在未来场景、同一冻结后续策略，仅修改当前 Keep/Edit；通过同场景配对控制外生差异。
- **双层比较：** E3-A 固定状态和 MPC 估计，让融合规则成为唯一改变维度；E3-B 则完成实际闭环在线收益对照。两组数据共同判断是 MPC 预测错误还是 Q/MPC 融合错误。

如果实验得到真实改善，下一步再考虑风险特征与预算价值加入网络训练，或者将候选接口从 SingleMove 延伸至 ERL-DDE Hints。**本 E3 不改变候选优化器。**
