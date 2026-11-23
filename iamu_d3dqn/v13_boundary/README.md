# IAMU V13-E4: 预算=1 的 Keep 未来成本低估、A2 固定数据与 Rollout 基线

**基线：** `Dragon1712/IAMU` commit `c49e8d09517da632433afe270d9a2d01649f5cb9`。

仅新增 `v13_boundary/` 和 `tests_v13_boundary/`，不覆盖既有物理环境、V12/V13 代码或实验结果。**默认 DRY RUN**，没有自动在线 400-Episode 训练。

## 安装及测试

```bash
cd ~/code/iamu
unzip -o /path/to/IAMU_V13_Budget_Boundary_E4.zip
cd iamu_d3dqn
conda activate erldde
python -m unittest discover -s tests_v13_boundary -p 'test_*.py' -v
bash v13_boundary/scripts/run_v13_boundary.sh
```

## E4-1：最关键的预算一次 Keep 后续价值误差

输入为**服务器本地** `results/v13_p0_p3_seed43/probe/probe_bank.pt`，不是 GitHub 上的 `probe_states.csv`。用以下命令确认 EP100 实际梯度文件名：

```bash
for m in A0 A2 A3; do
 find "results/v12_${m}_seed43_400/validation_snapshots" -maxdepth 1 -name 'val_000100_grad_*.pt' -print
done
```

每组只存在一个结果后，设置准确路径：

```bash
export A0_EP100=$(find results/v12_A0_seed43_400/validation_snapshots -name 'val_000100_grad_*.pt' -print -quit)
export A2_EP100=$(find results/v12_A2_seed43_400/validation_snapshots -name 'val_000100_grad_*.pt' -print -quit)
export A3_EP100=$(find results/v12_A3_seed43_400/validation_snapshots -name 'val_000100_grad_*.pt' -print -quit)
RUN=1 STAGE=boundary bash v13_boundary/scripts/run_v13_boundary.sh
RUN=1 STAGE=boundary_summary bash v13_boundary/scripts/run_v13_boundary.sh
```

`branch_values.csv` 每个状态×未来外生扰动×模型×续行策略×动作保存：`interval_reward`、`bootstrap_discount`、`next_bootstrap_value`、`true_suffix_return`、`bootstrap_error`、`estimated_wait_cost_understatement`。注意 `estimated_wait_cost_understatement=100*bootstrap_error`：正数表示 Keep 预测的剩余等待成本偏低。

`paired_gap.csv` 每条 tape 逐行验收精确恒等式：

\[
(y_{Edit}-y_{Keep})-(G_{Edit}-G_{Keep})=e_{Edit}-e_{Keep}.
\]

**Self continuation** 是被冻结审计模型的零探索贪心；**Greedy** 是物理贪心策略敏感性对照。结论不能把两套续行混为一谈。`zero_gap_trace.csv` 记录八条 tape 均为零差的状态 Keep 后首次 Edit 的实际时间、事件与修改语义。

只加载 `best.pt`/`latest.pt` 时，`target` 由在线权重初始化，记为 `target_equal_online`，并非训练时历史 target。如 EP400 本地 `training_state.pt` 与 `latest.pt` 权重完全一致，可另行执行：

```bash
python -m v13_boundary.audit_boundary \
 --probes results/v13_p0_p3_seed43/probe/probe_bank.pt \
 --checkpoint A2_EP400=results/v12_A2_seed43_400/latest.pt \
 --full-state A2_EP400=results/v12_A2_seed43_400/training_state.pt \
 --continuations self,greedy --tapes 8 --out results/v13_boundary_seed43/target_real_a2
```

分片支持 `--start-index` / `--stop-index`，索引针对**过滤后预算=1 的 40 状态**。`summarize_boundary.py` 可接受多个不重叠分片，先在状态内平均八条 tape，再按 episode 聚类 Bootstrap。不能把同一状态的八个外生扰动当独立状态。

## E4-2：用两条固定数据曲线区分更新次数和样本曝光

```bash
GPU=0 RUN=1 STAGE=fixed bash v13_boundary/scripts/run_v13_boundary.sh
```

输入为 V13 同一原始数据派生的 `compression/primitive.pt` 与 `compression/budget_tail.pt`，严格检查相同源 SHA256、物理 episode 与任务表。设 `G=4000`，`Gshort=round(G*N2/N0)`，每条 A0/A2 曲线均保存 `init/short/full.pt` 与多个探针报告，输出 `fixed_data_curve.csv`、`fixed_gap_summary.csv`。缩短更新也改变 target 同步次数和 optimizer 时间，结论必须注明。

## E4-3：冻结策略 Rollout 的一次策略改进（仅在 E4-1/E4-2 后）

```bash
RUN=1 STAGE=label_bank bash v13_boundary/scripts/run_v13_boundary.sh
RUN=1 STAGE=labels bash v13_boundary/scripts/run_v13_boundary.sh
# 诊断参考必须先有 V13 frozen A0 continuation:
RUN=1 STAGE=a0cf bash v13_diagnostics/scripts/run_v13_p0_p3.sh
GPU=1 RUN=1 STAGE=rollout bash v13_boundary/scripts/run_v13_boundary.sh
```

R1：只训练冻结 A0 rollout 动作差评分器；R2：A2 TD+均匀预算 gap；R3：A2 TD+预算=1 gap。三组使用相同网络初始化与训练配置。训练标签使用不同外生种子的训练-only 探针银行，绝不能从诊断用 120 探针生成标签。

生成标签会模拟训练池所有合法 Edit，Keep 在同状态×tape 下缓存复用；`label_cost.csv` 保存**标签池实际物理步数**，R2/R3 在标签子集的 Keep 成本摊销意义下匹配预算。摊销成本不能宣称为每个方法独立生成数据所需总成本。

这只是 **one-shot rollout policy improvement**，不等同多轮 approximate policy iteration。本轮没有独立案例 10–14 或在线闭环证明。

## 交付验收状态

隔离环境可运行单元测试、模块命令帮助及 Shell 语法检查。真实 IAMU 环境、历史 `.pt` 文件、40 状态分支和离线模型训练必须在原服务器执行。每个实验记录代码与输入 SHA256，失败不能覆盖旧结果。完整物理环境快照是本地可信 pickle；不要加载不可信 `.pt` 文件。
