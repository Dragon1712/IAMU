# IAMU V1.1：候选与 Replay 语义修正 + 多 Actor 确定性入库的新实验协议

> 基于 GitHub `Dragon1712/IAMU` 2026-11-17 提交 `52dec41ac8ed` 中的 `iamu_d3dqn` 接口。**本补丁只新增文件，不直接覆盖旧 `rl_env.py`、`rl_ddqn.py`、旧训练器和既有结果。** 在服务器上运行 `build_variants.py` 将生成带 `_v11` 后缀的新模块；源码锚点不符则立即失败。新环境增加动作后与旧模型的对比只能作为迁移诊断，严格算法对照须全部在新协议内重训。

## 1. 已实现与暂不混合的范围

| 部分 | 实现状态 | 解释限制 |
|---|---|---|
| 候选扫描前 K → 逐个检查至 K 个有效候选 | 已实现（`rl_env_v11`）| 不是全局最优 K；排序沿用原实现 |
| 基准预测执行成本只计算一次／事件 | 已实现 | 不是跨事件的永久缓存 |
| 候选审计、重叠 forced Keep 原因、互斥主类 | 已实现 | `decision_flags` 附着每条新 Replay 转移；旧 Replay 没有该字段 |
| physical mask 与 point heuristic mask | 已实现 | 只对最终 K 个动作槽位编码，不表示全量未收录船舶 |
| `positive` / `none` / `mixed` 候选 | 已实现 | 三者改变了可见动作集合，须分别新训练 |
| execution margin 与 bootstrap margin | 已实现 | 旧 checkpoint 若无新字段，两者均回退为旧 keep_margin |
| 带元数据的增量 Replay 分层索引 | 已实现 | 样本不足时从另层补足，记录实际比例与 shortfall；新数据结构不改变物理环境 |
| 未加权重采样 vs importance weighting vs uniform | 已实现 | importance 仅支持两个 stratum 都有样本，rho=0/1 禁止 |
| 多 Actor 并行、同轮冻结权重、按 Episode ID 学习 | 已实现（新训练器）| 不保证旧异步协议的相同训练轨迹 |
| 多 CPU worker 验证、状态及权重快照、EP200 全状态归档 | 已实现 | CPU 验证独立进程；断点仅以完整 round 为基准 |
| 固定 EP200 TD 自举标签 | 已实现（`fixed_label_probe.py`）| 是固定 bootstrap 参考，不是真实未来收益 |
| 阈值冻结模型扫描、筛除候选的事后反事实抽检 | 已实现 | 后者是单实现未来的诊断，不是部署 oracle |
| 合批模型推理 | 提供独立函数 `batch_inference.action_batch` | 未默认接入采集；浮点误差边界需测试 |
| MPC 动态预算前瞻部署 | **未改变 E3 MPC 决策策略** | 仅提供公用事件候选适配层；缺预测未来 ETA 消息生成器，部署模式明确拒绝读取隐藏未来 |
| 跨 checkpoint 的验证缓存和环境深复制优化 | **暂不启用** | 必须先固定环境/预测器版本与缓存键，才能宣称缓存等价 |

## 2. 物理候选 vs 即时收益筛选

新环境仍按原候选排序：当前事件船优先、ETA 是否晚于原计划、计划开始时刻与船舶 ID。唯一关键改动是不会**先截取 K 艘船**；而是扫描后才截断已保留的 K 个候选。

- `positive`：只返回当前公开 ETA 点预测增益超过 `min_gain` 的可行提案。
- `none`：返回排序后前 K 个物理可行的不同方案，即使其点预测增益为负。
- `mixed`：默认约一半正收益与一半其余物理可行方案；某组数量不够时填剩余槽位。组内仍沿原排序，不是全局选 K 最优。

每个时刻记录 `pending_ships, ships_examined, unchanged_proposals, infeasible_proposals, screened_proposals, returned_candidates, search_exhausted`，以及 `initial_event, decision_allowed, budget_exhausted, all_started, has_raw_proposal, has_retained_edit` 这些**非互斥**布尔量。所有标记都只依赖当下公开状态、可行性与点预测；真实物理未来不会进入 agent 的观察。

`rl_env_v11` 对返回动作保留两层掩码：`physical_mask`（允许模型执行）与 `heuristic_mask`（即时点预测为正）。不要把 `heuristic_mask` 直接解释为物理合法性，更不要把它的值等同于反事实长期收益。

## 3. 分层 Replay 目标及合法性

`IndexedReplay` 不在每次梯度更新时遍历全 Replay。它维护固定容量环形存储和两个 O(1) 插入、删除的密集索引池。每个 minibatch 随机选择预计比例 `rho` 的可行动作状态，其余来自 forced Keep；**同一层中不改变 Keep/Edit 原自然比例**。批内无放回，更新间样本仍可重用。当某层不足，会从另一层补齐，并在 `train_curve.csv` 报告 `actual_rho` 和补齐次数。

- `--replay-weighting unweighted`：模型损失更重视可行动作状态，**改变近似训练重点**。
- `--replay-weighting importance`：按实际采样比例使用 `p/rho_actual` 与 `(1-p)/(1-rho_actual)` 乘在 per-item Huber loss 上，近似恢复原经验分布的期望目标；端点不允许。
- `--replay-weighting uniform`：原 Replay 等概率抽样，不改变抽样权重，方便做重采样效果对照。

训练网络的 `bootstrap_margin` 仅影响 Double-DQN 下一动作 `argmax`，执行时只使用 `execution_margin`；不把 margin 冒充真实改约成本。

## 4. 首先执行：安装与单元测试

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_V11_ParallelControlledProtocol.zip
cd iamu_d3dqn
conda activate erldde
python v11_protocol/build_variants.py --check-only
python v11_protocol/build_variants.py
python -m unittest discover -s tests_v11 -p 'test_*.py' -v
```

`build_variants.py` 会新增以下模块：

```text
iamu_dynamic/rl_env_v11.py
iamu_dynamic/rl_ddqn_v11.py
iamu_dynamic/rl_experiment_v11.py
parallel_validation_v11.py
v11_protocol/v11_actors.py
```

这些模块由旧版本源码在服务器**按唯一锚点**生成；未匹配会报错，不会静默打补丁。老代码没有变。测试应包含 9 个纯逻辑测试和 5 个真实 IAMU 环境测试。若有生成源不匹配、真实仿真器的 Keep 成本对照失败、提案与原 `SlotPlanner.single_move` 不一致，**禁止继续训练**。

## 5. P0：先诊断候选覆盖，不必新训练

```bash
python -u v11_protocol/audit_candidates.py \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --scenarios shock,repeated,small --seeds 400001:400002 \
  --budgets 2,5,10 --anchors keep,greedy \
  --states-per-group 3 --counterfactual-per-state 2 \
  --candidate-limit 4 --out results/v11_candidate_audit
```

产生 `reason_flags.csv`、`candidate_coverage.csv`、`screened_counterfactual.csv`。比较同一状态的 `old_first_k_count` 与 `returned`，计算旧环境“前 K 艘船检查不到”导致的候选遗漏。对即刻预测非正的候选用同一事后物理未来、固定全 Keep 后续策略测试收益；**它是研究抽检，不是部署时的预知未来**。固定另一个后续策略和更多未来的检验应单独追加。

## 6. P0：统一新多 Actor 协议训练

为节省时间，先以单卡、4 CPU actor、4 validation worker、8 Episode 冒烟：

```bash
python -u v11_protocol/train_v11.py \
  --train-cases ../2024LiERLDDE/case_code/1.txt ../2024LiERLDDE/case_code/2.txt \
  --val-cases ../2024LiERLDDE/case_code/8.txt \
  --scenarios shock --budgets 2 --val-seeds 400001 \
  --episodes 8 --round-size 4 --eval-every 4 \
  --workers 4 --validation-workers 4 --device cuda:0 \
  --candidate-screen positive \
  --execution-margin 0.1 --bootstrap-margin 0 \
  --replay-rho 0.25 --replay-weighting unweighted \
  --out results/v11_smoke
```

**不要将 `v11_smoke` 的结果作为性能结论**。观察每个 round 的 `policy_version` 一致、实际 Episode ID 是 1...8、Replay 样本比例、梯度步和验证条数。全部通过后再执行四卡的标准协议。

四卡 16 Actor/16 Validator 的预设参数扫描：

```bash
python v11_protocol/launch_four_gpu.py \
  --matrix v11_protocol/bootstrap_margin_3way.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --actors-per-gpu 4 --validators-per-gpu 4 --dry-run

python -u v11_protocol/launch_four_gpu.py \
  --matrix v11_protocol/bootstrap_margin_3way.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --actors-per-gpu 4 --validators-per-gpu 4 \
  2>&1 | tee v11_margin_3way.log
```

默认矩阵包含 Seed43 的 **3 个** margin 组（A:0/0，B:0/0.1，C:0.1/0.1；表示 bootstrap/execution）以及 Seed42 的一个 B 组复现实验，因而可以同时使用 **4 块 GPU、总计 16 个 actor 与 16 个 validator**。只有 Seed43 内的三组构成同 seed 对照，Seed42 的结果只能用于跨种子复现。后续应补齐其余 seed 的 A/C 组，不得把不同 seed 视为直接配对的 margin 组。

每个 GPU 仅一个 learner，每组 4 actor + 4 validator。它采用新的轮次采集协议：同一轮所有 worker 载入相同冻结模型；worker 完成后可以立刻接收本轮下一个预定任务；训练数据入库**严格按照 Episode ID**，learner 可在其他 actor 采集时执行有序更新。轮次末尾再冻结新的 actor 权重；到验证边界暂停采集，调用独立 spawn CPU 验证池。

`--round-size` 调大可减少权重同步但增加策略滞后，**会改变训练数据与梯度轨迹**。先在短程运行中比较 8/16/32 的吞吐和 staleness，再为所有新对照统一固定一个值（默认 16）。Job-keyed actor RNG 消除了 worker 分配时的探索随机数差异，但 GPU 算子、硬件等仍可能不保证位级相同。

## 7. EP200 冻结基线与独立诊断

在 `--fork-episode 200` 的完整运行中，验证边界完成后自动另存 `full_state_ep0200.pt` 及 SHA metadata。不要把只有网络权重的历史 E4 `val_000200.pt` 当作完整 Replay 和 Adam state。

读取**本地可信** EP200 Replay，拆分 overlap flags 和互斥主类别：

```bash
python v11_protocol/replay_reasons.py \
  --fork results/v11_m1_seed43/full_state_ep0200.pt \
  --trust-local-training-state \
  --out results/v11_replay_reasons_seed43
```

输出 `primary_reasons.csv`、`overlap_flags.csv`、`reason_combinations.csv`，重点分析 `budget_exhausted` 与 `all_started` 的交集，以及各 episode 最后 25% 事件中的 forced Keep。**该分析与候选修复之后的新 Replay 数据对应，不可反向拿来声称旧 94.8% 的各类比例已经测得。**

冻结执行阈值扫描不需要重训：

```bash
python v11_protocol/margin_scan.py \
  --checkpoint results/v11_m1_seed43/best.pt \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --margins 0,0.05,0.1,0.15 \
  --out results/v11_m1_margin_scan
```

固定 TD 自举标签诊断：

```bash
python v11_protocol/fixed_label_probe.py \
  --fork results/v11_m1_seed43/full_state_ep0200.pt \
  --reference-checkpoint results/v11_m1_seed43/validation_snapshots/val_000200_grad_XXXXXXXXX.pt \
  --compare results/v11_m1_seed43/validation_snapshots/val_000200_grad_XXXXXXXXX.pt \
  --compare results/v11_m1_seed43/validation_snapshots/val_000300_grad_YYYYYYYYY.pt \
  --trust-local-training-state \
  --out results/v11_fixed_label_seed43
```

实际快照名以 `validation_ledger.json` 为准；命令中的 `XXXXXXXXX/YYYYYYYYY` 是路径占位符，不要原样执行。这一参考的 selector/evaluator 都取自 EP200 的 frozen Online/Target，固定标签不是事后收益，也不是 Q*。

## 8. 对照顺序和解释边界

建议不要立即对所有因素做全因子重训，先依次完成：

1. **不重训**：候选覆盖和 screened proposal 事后正收益抽检；若几乎没有真实未来获益，则暂缓 `none/mixed` 的昂贵新训练。
2. **不重训**：同一 frozen checkpoint 的 execution margin 离线评价；不改变网络的 bootstrap 学习。
3. **新协议**：相同 candidate mode、rho 和 seed 上训练 A(bootstrap0/execution0)、B(0/.1)、C(.1/.1) 三组，区分 bootstrap/execution 影响。
4. **新协议**：选定双 margin 后，在同一 candidate mode 下改变 `rho` 和 Target 同步；`unweighted/importance` 作为明确分离的训练目标方案。
5. **新协议**：选少数配置，再测试 `none`、`mixed` 候选模式。必须重训，不以旧检查点严格比较新的候选策略。
6. **最后**：固定模型和参数，再在独立 Case 10–14 评估；MPC 与其他算法共用**公开**的候选、事件、动作规则，部署时预测未来必须由独立模型生成，不使用 `Scenario.true_etas` 或未公开的消息。

### 关于计算语义与加速

已经实际采用：事件内基准 forecast 只做一次、候选不足时继续扫描直至找到 K 个或候选耗尽；增量 Replay 索引避免每次 gradient 全量扫描；验证 worker 各自加载一次模型并完成多条任务；固定状态/固定标签只做一次昂贵的评估计算。`action_batch` 可以对固定模型的多个状态一次推理，但暂不默认接入 actor（可能产生浮点差异）。

未声称等价：改变 `round-size`、worker job 分配规则、Replay 采样权重、margin、候选生成、更新数量、Target 频率都会影响训练轨迹，需要在本新协议内统一进行严格对照。

### 必须做的验收

- 对 `rl_env_v11` 与原环境，在完全 Keep 策略下的每条相同 Case+Scenario+Seed 加权等待应一致；`SlotPlanner.single_move` 方案与新 `_proposal_with_reason` 返回的方案应一致。
- 排除生成器锚点匹配失败、候选物理 mask 不合法、已开始作业船仍被修改、预算超支、训练/验证与原配置不一致等错误。
- `train_curve.csv` 中 Episode ID 递增且无缺失，round 内 policy_version 完全相同，Replay 索引的两层总数始终等于 Replay 总量；实际 rho 和样本不足比例必须报告。
- 原 V1.1 checkpoint 的 `execution_margin/bootstrap_margin` 必须在新加载器中往返保持一致，旧版本 checkpoint 回退兼容行为需单独测试。
- `manifest.json` 必须带代码 SHA；修改环境代码或配置后使用**新输出目录重新训练**，不能悄悄恢复旧实验。

**状态说明：** 本交付已经本地通过纯逻辑/源码安全检查；真实 IAMU 环境和多 actor GPU 冒烟需要在服务器安装、生成变体后验证，不能凭这些本地测试直接宣布新算法性能已提升。
