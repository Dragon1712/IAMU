# IAMU V1.2 研究增量补丁（不包含训练结果）

基于远程 GitHub `Dragon1712/IAMU` `main` 的已核查提交 `2d96981a38434014f14e31be5641d422d28835f8`。

**研究约束**：所有船等权；原硬预算、Keep/单船 Edit、`gamma=1`、`reward_scale=100`、多 actor 冻结轮次、episode ID 有序 Replay 消费不变。**不启动训练，不改写历史结果，不把真实未来用于在线决策。** 新实验使用 `results/v12_*`。

## 0. 为什么以补丁包交付

GitHub 连接允许读取仓库，却在创建研究分支时返回 403 `Resource not accessible by integration`。因此当前版本没有写入 GitHub，补丁应用及提交由拥有本地仓库权限的研究人员完成。本包不会下载、删除或覆盖服务器上的 `.pt` 文件。

## 1. 安全检查与应用

在 **干净的 IAMU Git 工作树**中操作，先将本压缩包解压到仓库之外：

```bash
# 工作目录不要求与包所在目录一致
python /path/to/iamu_v12_patch/apply_v12.py --repo /home/moco/code/iamu --check
python /path/to/iamu_v12_patch/apply_v12.py --repo /home/moco/code/iamu --apply
```

Windows：

```powershell
python C:\path\to\iamu_v12_patch\apply_v12.py --repo D:\Coding\code\iamu --check
python C:\path\to\iamu_v12_patch\apply_v12.py --repo D:\Coding\code\iamu --apply
```

脚本读取当前 HEAD、检查未提交工作，并**在写入前完成全部旧代码锚点检查和 Python 语法检查**。它创建分支 `research/limited-budget-replay-v12`；若同名分支或目标新文件存在则拒绝覆盖。不支持无提示重复应用。

如果服务器有未跟踪但仍需保留的 `iamu_d3dqn/results/` 数据，可在两个命令末尾均加入 `--allow-untracked-results`；**不允许**以这个选项忽略已修改的源码。若 HEAD 已非指定提交，先比较新旧代码；确认后才可手动追加 `--allow-other-head`，但所有锚点仍需匹配。

如要撤销未提交的补丁，应先检查 `git status` 和 `git diff`，不要执行无检查的清理命令，以免影响结果文件。

## 2. 已修改与新增的模块

| 模块 | 内容 |
|---|---|
| `rl_env_v11.py` | 区分未搜索与搜索穷尽；累计候选生成和投影计时 |
| `rl_ddqn_v11.py` | Transition 显式 `budget_before/after`、`event_index`、`raw_steps`、`transition_version` |
| `v11_actors.py` | 显式 episode-keyed exploration seed；预算元数据与耗时报告 |
| `indexed_replay.py` | 保留 ring slots、head/size、两层索引顺序和随机状态 |
| `train_v11.py` | 严格 manifest、3 类 Replay 指标、可选派发协调器、可选耗尽尾段压缩、原始事件计数 |
| `fixed_label_probe.py` | 从 fork 恢复 gamma/bootstrap margin，分开标记 Online 与 Target 来源 |
| `recovery_v12.py` | 独立的 V1.2 训练状态精确恢复；历史 V0.8 恢复代码不动 |
| `experiment_fingerprint_v12.py` | 科学配置、任务列表、数据文件和代码哈希 |
| `dispatch_v12.py` | 可选有界异步派发，仅处理任务/消息，不修改 learner/Replay |
| `research_v12.py` | 三类标记及预算耗尽尾段压缩算法与回报校验 |
| `build_decision_bank.py` | 多锚点轨迹分层固定 actionable 状态库 |
| `evaluate_qgap_bank.py` | 全候选 Q、Q-gap、margin 动作、EP200/300/400 状态转换 |
| `counterfactual_decision_probe.py` | 独立 Keep/Edit 克隆、同未来下 Greedy/Keep 续演 |
| `margin_scan_v12.py` | 冻结模型扫描；基线只算一次；配对 CSV、热图、等待-修改图 |
| `replay_tail_audit_v12.py` | 历史 EP200 完整 Replay 的离线压缩账目诊断 |
| `uniform_replay_v12.json` | 4 个新训练任务的显式矩阵；**不会自动运行** |

### 训练保护

- `--exhaustion-tail-mode raw` 为默认。`collapse` 是**新待验证干预**，不允许作为历史训练的“无损替换”。
- `--dispatch-coordinator` 为可选，默认**关闭**；先验证短程同任务、同轮权重、相同 Replay 采样与参数结果，再启用。
- V1.2 训练状态只接受 V1.2 保存格式。历史完整 fork 只用于诊断/对照；如果要严格复用旧 M0/43，先核查科学语义。不要直接用新训练器恢复 V1.1 旧 `training_state.pt`。
- GPU 和 worker 数变化可以在科学配置一致时恢复，但**不承诺跨硬件位级一致**。

## 3. 第一阶段：短程验收（必须先运行）

```bash
cd /home/moco/code/iamu/iamu_d3dqn
python -m unittest discover -s tests_v11 -p 'test_*.py' -v
python v11_protocol/launch_four_gpu.py \
  --matrix v11_protocol/uniform_replay_v12.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --dry-run
```

建议同一配置做 `raw` 8 episode 冒烟和 `collapse` 8 episode 冒烟，分别使用**全新**目录：

```bash
python -u -m v11_protocol.train_v11 \
  --train-cases ../2024LiERLDDE/case_code/1.txt ../2024LiERLDDE/case_code/2.txt \
  --val-cases ../2024LiERLDDE/case_code/8.txt \
  --scenarios shock --budgets 2 --val-seeds 400001 \
  --episodes 8 --round-size 4 --eval-every 4 --state-every-rounds 1 \
  --workers 4 --validation-workers 4 --seed 43 --device cuda:0 \
  --q-network set_ddqn --execution-margin 0 --bootstrap-margin 0 \
  --replay-weighting unweighted --replay-rho 0.25 \
  --exhaustion-tail-mode raw --out results/v12_smoke_raw_8
```

将最后一行的 `raw` 改为 `collapse`、输出目录改成 `results/v12_smoke_collapse_8` 即为第二个冒烟。测试时保留原始训练状态，对连续运行和中断后从**完整 round 边界**恢复分别比对任务顺序、Replay 内容/索引、采样序列、梯度数、权重和优化器。单机同环境下争取位级一致，否则报告偏差。切勿用跨硬件的微小浮点差异解释为训练机制效应。

## 4. Q-gap 固定状态库（无需新训练）

下面模型路径为**占位符**，请从服务器每组 `validation_ledger.json` 的 `snapshot` 字段选择已有的 EP200、EP300、EP400 文件。

```bash
python v11_protocol/build_decision_bank.py \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --anchors greedy,periodic,m0,m1 \
  --rl-checkpoint m0=/absolute/path/to/M0_EP400.pt \
  --rl-checkpoint m1=/absolute/path/to/M1_EP400.pt \
  --max-states 200 --per-stratum 6 \
  --out results/v12_decision_bank

python v11_protocol/evaluate_qgap_bank.py \
  --bank results/v12_decision_bank \
  --checkpoint m0_ep200=/absolute/path/to/M0_EP200.pt \
  --checkpoint m0_ep300=/absolute/path/to/M0_EP300.pt \
  --checkpoint m0_ep400=/absolute/path/to/M0_EP400.pt \
  --checkpoint m1_ep200=/absolute/path/to/M1_EP200.pt \
  --checkpoint m1_ep300=/absolute/path/to/M1_EP300.pt \
  --checkpoint m1_ep400=/absolute/path/to/M1_EP400.pt \
  --margins 0,0.05,0.10,0.15 --out results/v12_qgap
```

状态库的 `public_observations.npz` 只包含网络可见特征与动作 mask。`realized_future_snapshots.pt` 含完整仿真器和已实现未来，**仅用于离线诊断**，未经信任不得反序列化。`states.csv/json` 包含来源策略、预算、阶段、候选身份和状态 ID。`checkpoint_changes.csv` 使用 `*_ep200/300/400` 名称识别跨 checkpoint 的 Edit–Keep–Edit 等变化。

## 5. 实现未来下的成对反事实（只做诊断）

```bash
python v11_protocol/counterfactual_decision_probe.py \
  --bank results/v12_decision_bank \
  --continuation greedy --trust-local-snapshots \
  --out results/v12_cf_greedy

# 仅选小子集测试续演敏感性
python v11_protocol/counterfactual_decision_probe.py \
  --bank results/v12_decision_bank \
  --continuation keep --max-states 40 --trust-local-snapshots \
  --out results/v12_cf_keep_40
```

每个保留候选相对 Keep 计算一个同实现未来的 gain，包含同一后续策略、独立预算核算、独立深复制。收益仅为 `realized_future_diagnostic`，不是条件期望，也不是全局最优。

## 6. 已有模型的 execution margin 扫描

```bash
python v11_protocol/margin_scan_v12.py \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --checkpoint m0_ep200=/absolute/path/to/M0_EP200.pt \
  --checkpoint m0_ep300=/absolute/path/to/M0_EP300.pt \
  --checkpoint m0_ep400=/absolute/path/to/M0_EP400.pt \
  --checkpoint m1_ep400=/absolute/path/to/M1_EP400.pt \
  --margins 0,0.05,0.1,0.15 \
  --out results/v12_margin_scan
```

主输出 `runs.csv`, `summary.csv`, `paired_comparisons.csv`，可选 `figures/`。在这个脚本内，Keep/Greedy/Periodic 各场景仅评价一次。若需完整实验 A，提供全部 4 个模型 × 3 个 checkpoint。

## 7. 历史 EP200 Replay 耗尽尾段离线核查

```bash
python v11_protocol/replay_tail_audit_v12.py \
  --fork results/v11_m1_seed43/full_state_ep0200.pt \
  --trust-local-training-state --out results/v12_tail_audit
```

旧 Transition 没有预算元数据，离线工具仅在发现**完整** episode 时，使用 `train_rows` 的初始预算和动作序列重建预算；若无法检查完整性则标记 skipped，绝不从 `obs_global` 浮点特征猜测。仅核查剩余回报及存储量，不做在线训练。

## 8. Uniform 对照矩阵（暂不正式启动）

`v11_protocol/uniform_replay_v12.json` 准备了以下四个新任务：

| 任务 | seed | Replay | tail |
|---|---:|---|---|
| v12_uniform_seed43 | 43 | Uniform | raw |
| v12_rho050_seed43 | 43 | ρ=0.50 unweighted | raw |
| v12_uniform_seed42 | 42 | Uniform | raw |
| v12_rho025_seed42 | 42 | ρ=0.25 unweighted | raw |

历史 M0/43/ρ=0.25 是否严格可复用，必须比对新旧科学协议。**矩阵的 dry-run 仅打印命令，不会训练**。通过前述测试并取得长程训练许可后，才使用 `launch_four_gpu.py` 去掉 `--dry-run`。

实验 E 在线 `collapse` 应使用**独立新目录**、单 seed 和相同任务列表；先对照原始事件数、样本数量和更新数，再看 Q-gap 与闭环等待。不得把原始与压缩尾段的数据在同一 Replay 中混用。

## 9. 本地预检、限制与后续核查

本补丁制作端已完成 Python 源码语法检查、纯逻辑压缩/Replay 测试、合成短程连续与恢复参数一致性测试及乱序派发测试。**没有**直接取得服务器 GPU、真实 IAMU 仿真环境或历史 `.pt`，因此不能声称完成服务器端环境积分测试、全链路恢复一致性、Q-gap 输出一致性或性能提升。

落地验收应优先核对：新的 `search_performed` 语义，`candidate_limit` 保持 4、已有 `single_move` 行为不变；原始轨迹加权等待与 `reward_scale` 严格核对；回报压缩前后相同；新 manifest 数据/代码 SHA 变化使恢复和缓存安全失效。若任一验收失败，停止长时间训练。
