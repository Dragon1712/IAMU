# IAMU V1.0-E2：多阶段预算前瞻 + ETA 风险采样稳定性

**研究范围不变：** 冻结 V0.9 Set-aware DDQN，保留原 `iamu_Interface` 的单船候选、硬预算与物理仿真；仅将 `q_budget/q_joint` 中原启发式预算紧张度替换为多阶段、分支独立、有限预算的场景滚动修复估计。

## 1. 本版改变了什么

| 项目 | 旧 V1.0-E1 | 新 V1.0-E2 |
|---|---|---|
| `q_budget` | 固定 `beta_budget * scarcity` 扣分 | Keep/各 Edit 独立预算的多阶段 MPC 成本差，与冻结 Q-gap 凸组合 |
| `q_joint` | 冻结 Q + CVaR 修正 + 启发式预算扣分 | 冻结 Q + CVaR 修正 + 多阶段 MPC 成本差 |
| `q_budget_heuristic` | 不存在 | 完整保留 E1 `q_budget` 的旧公式 |
| `q_joint_heuristic` | 不存在 | 完整保留 E1 `q_joint` 的旧公式 |
| `lookahead` | 最多 1 个未来阶段（此前 smoke） | 支持 H=0,1,2,3... 个顺序阶段；未来每个阶段预算独立扣减 |
| ETA 采样检查 | 单一 M=16 | 独立参考 N=256；M=16/64/128；重复种子和嵌套公共随机数 |

**本次不修改 `iamu_d3dqn/`、底层 `rl_env.py`、DDQN 网络权重、`candidate_screen` 或真实未来事件。**

## 2. 数学原理与执行过程

当前状态为 \(s_t\)，预算 \(b_t\)，合法候选 \(a\)。

- Keep：\(P^0=P_t\)，未来可用预算 \(b^0=b_t\)。
- Edit：\(P^a=P_a\)，未来可用预算 \(b^a=b_t-1\)。
- 对同一组 \(\xi^{(m)}\) 进行 H 次未来阶段推进。仿真器以采样 ETA 计算物理实现，但**未来决策仅能观察在未来时刻已公开或通过信号模型生成的 ETA**。
- 每个未来阶段至多执行一次合法单船修复，只有执行时才扣除 1 单位预算；H 阶段结束后用该样本完成全部未开始船舶的模拟并计入总等待。

\[
\widehat J_H(s,b,a)=\frac1M\sum_{m=1}^{M}C_H(P^a,b^a;\xi^{(m)})
\]

\[
A_{\rm MPC}(a)=\frac{\widehat J_H(s,b,\mathrm{Keep})-\widehat J_H(s,b-1,a)}{\text{reward_scale}}
\]

**预算融合评分**（0≤\(w_B\)≤1）：

\[
S_{\rm budget}(a)=(1-w_B)\,[Q(s,a)-Q(s,\mathrm{Keep})]+w_B A_{\rm MPC}(a)
\]

**风险-预算融合评分：**

\[
S_{\rm joint}(a)=S_{\rm budget}(a)-w_R\,\frac{\mathrm{CVaR}_{0.9}(W_a)-\mathrm{CVaR}_{0.9}(W_{Keep})}{\text{reward_scale}}
\]

只有评分超过**原始 DDQN keep margin** 才执行。不会再次叠加硬编码的预算紧张度罚项。`--lookahead-risk-weight` 只作用于纯 `lookahead` 评分；`q_joint` 的 MPC 分量使用预期成本，风险修正单独使用当前候选的 CVaR 差，避免同一尾部指标在两处重复扣减。`--beta-budget` 的意义从 E1 的“扣分系数”变成了 E2 的“Q/MPC 凸组合权重”，**前后实验不可把这个数值直接当成相同超参数。** 原 E1 保持在 `q_*_heuristic` 中，其系数由 `--legacy-beta-budget 0.1` 控制。

**研究限制：** H 阶段内的后续修复目前使用基于阶段公开 ETA 的贪心单船优化；这是有限阶段滚动后瞻的近似回报评估，**不是**精确动态规划或最优多阶段随机规划。也并未重新训练 Q 网络，Q 与 MPC 融合仍属于冻结推理消融。

## 3. 安装

将新的压缩包在 IAMU 根目录解压；它覆盖 `iamu_Interface/` 中本次更新的实验文件，不改 `iamu_d3dqn/`：

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_Interface_V10_E2.zip
cd iamu_Interface
conda activate erldde

# 如果此前已经执行 setup_interface.py 并复制 V0.9 引擎快照，无需重复安装。
# 如果缺少 iamu_dynamic/core.py 等文件，执行：
python setup_interface.py --source ../iamu_d3dqn --copy-assets

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_v10_e2*.py' -v
python -m unittest discover -s tests -p 'test_v10_integration.py' -v
```

## 4. 实验执行顺序

### E2-0：一组冒烟验证、旧启发式对照

```bash
bash scripts/run_e2_multistage_smoke.sh
```

该脚本在 Case 8 / Shock / ETA Seed 400001 / B=2 比较：`set_ddqn`、新 `q_budget`、新 `q_joint`、旧两个 `_heuristic` 以及纯 `lookahead`。开启基线一致性验证。

### E2-1：多阶段预算模型验证

```bash
bash scripts/run_e2_multistage_validation.sh
```

全量 54 个条件；每次候选使用风险 M=16，MPC 采用相同随机场景前 8 个，共 H=2 个未来阶段，单船候选数上限4。**计算开销可能远高于 E1，先做冒烟再展开。**

可增加 `--out` 使用全新的输出路径；输出目录若非空会报错，避免覆盖。

#### H 深度消融（建议分开运行）

```bash
for H in 0 1 2 3; do
  python -u experiments/eval_risk_budget_v10.py \
    --policies set_ddqn,q_joint,lookahead \
    --scenarios shock,repeated,small \
    --seeds 400001:400003 --budgets 2,5,10 \
    --risk-samples 16 --mpc-samples 8 \
    --lookahead-stages "$H" \
    --beta-budget 0.25 --beta-risk 0.25 \
    --verify-baseline --device cpu --audit \
    --out "results/v10_e2_H${H}" \
    2>&1 | tee "v10_e2_H${H}.log"
done
```

**注意**：H=0 为仅现时计划的模型预测比较；H≥1 才能估计未来保存预算的价值。

### E2-2：ETA 风险采样稳定性（独立实验）

```bash
bash scripts/run_e2_risk_stability_smoke.sh
bash scripts/run_e2_risk_stability_validation.sh
```

冻结原始 Set-aware DDQN 的公开决策状态，不改变真实轨迹；对每个状态计算：

- ETA 抽样规模 M∈{16,64,128}；
- 每个 M 使用 3 个独立批次，批次内 M=16 是 M=64/128 的前缀，减少混杂；
- 用**独立**的 256 样本参考估计作比较；
- 估计误差：候选预期收益 MAE、尾部 CVaR 差 MAE、有害改约概率 MAE；
- 候选排序：预期收益最高和 CVaR 改善最好候选的一致率；
- 决策稳定性：Q+Risk 的 Keep/Edit 动作与参考的匹配率；
- 记录风险计算时间。

再执行：

```bash
bash scripts/run_e2_risk_stability_validation.sh \
  --joint --out results/v10_e2_risk_stability_joint_seed42
```

此时对比 **Q+Joint 固定 MPC 校正条件下** 的风险采样噪声影响。MPC 用独立、固定的场景批次，防止“风险估计变化”和“MPC 预算估计变化”混在一起。

**MPC 自身也有场景采样误差**，可先在同一原型轨迹上检验 M_mpc=4/8/16：

```bash
bash scripts/run_e2_mpc_sampling_smoke.sh
```

该脚本改变的是预算前瞻采样量，而前面的风险稳定性脚本改变的是 CVaR 风险采样量。必须分别评价，避免将两者混淆。

可以提高 `--reference-samples 512`；256/512 仅为蒙特卡洛参考，不代表真实港口到港分布。**稳定性与预测分布是否校准是两个问题**，未来仍需用真实到港数据或原 `validate_eta_forecast.py` 检验预测覆盖率。

### E2-3：三个训练种子

当前默认 Seed42。Seed43/44 的 `best.pt` 在训练服务器原目录，例如：

```bash
for SEED in 42 43 44; do
  python -u experiments/eval_risk_budget_v10.py \
    --checkpoint "../iamu_d3dqn/results/v09b_set_ddqn_seed${SEED}_300/best.pt" \
    --manifest "../iamu_d3dqn/results/v09b_set_ddqn_seed${SEED}_300/manifest.json" \
    --policies set_ddqn,q_joint,q_joint_heuristic \
    --scenarios shock,repeated,small --seeds 400001:400003 \
    --budgets 2,5,10 --risk-samples 16 --mpc-samples 8 \
    --lookahead-stages 2 --beta-budget 0.25 --beta-risk 0.25 \
    --verify-baseline --device cpu \
    --out "results/v10_e2_multiseed_${SEED}" \
    2>&1 | tee "v10_e2_multiseed_${SEED}.log"
done
```

Case 8/9 仅用于开发与参数选择，**Case 10–14 留作独立测试，不能根据其结果反复调参。**

## 5. 输出解释

`eval_risk_budget_v10.py`：`runs.csv`、`paired_summary.csv`、`decision_audit.csv`、`manifest.json`。

新增决策审计字段：`mpc_gap_scaled`、`mpc_mean_keep`、`mpc_mean_edit`、`mpc_scenarios`、`mpc_stages`、`mpc_seconds`、`risk_seconds`。比较同一公开状态中是否由于预算前瞻改变动作。

`eval_risk_sampling_stability_v10.py`：`stability_details.csv`、`stability_summary.csv`、`manifest.json`。

先检查 `mean_action_agreement`、`mean_best_tail_agreement`、`mean_tail_mae` 随 M 上升是否改善，同时检查 `mean_within_state_action_flip_rate` 是否降低；如果 M=128 仍然波动大，优先校准信念分布与检查极端场景，而不是盲目提高风险权重。

## 6. 上传 GitHub 结果

因 `iamu_Interface/.gitignore` 默认忽略 `results` 和 `*.log`，需要明确选择需要上传的 CSV 与 manifest：

```bash
cd ~/code/iamu
find iamu_Interface/results -type f \
  \( -name '*.csv' -o -name 'manifest.json' \) \
  -print0 | xargs -0 -r git add -f --
git add iamu_Interface/iamu_dynamic iamu_Interface/experiments \
        iamu_Interface/scripts iamu_Interface/tests \
        iamu_Interface/README_E2_多阶段预算前瞻与风险采样稳定性.md
git status --short
```

审查之后再自行提交和推送。**没有在这里写入 GitHub。**
