# IAMU V1.3-P0–P3：压缩价值学习的机制诊断（独立增量补丁）

基线：`Dragon1712/IAMU` commit **`53b8c2e37bf73da46b637b41ccd739274cf604ab`**。

本次**只实现第一批 P0–P3**。不修改 `iamu_dynamic/core.py`、`rl_env_v11.py`、`v12_protocol`、网络结构、奖励、候选生成和训练器；**不执行 400-episode 在线训练**。暂不实现 P4 固定更新/等复用训练、势函数变换及修复后在线验证。其原因是需要先获得 P0–P3 诊断数据并排除实现/建模问题。

## 文件及职责

- `common.py`：输入/代码/案例 SHA256、稳定 `state_id`、动作语义、平衡 episode 配置、可信本地序列化。
- `collect_raw.py`：从训练 Cases 1–7 采集完整逐事件轨迹：100 条 frozen A0+epsilon，100 条 0.5-keep/其余均匀选择 Edit。4 CPU worker 可并行，返回结果始终按 episode ID 有序保存。保存 `raw_episodes.pt` 和 `episodes.csv`；记录每条外生场景 SHA256。
- `audit_compression.py`：对同一原始轨迹执行 `primitive`、`budget_tail`、`decision_epoch`；检查累计奖励、gamma=0.9、终止、合法性、公共锚点及动作语义。生成 `compression_audit.json`、`shared_anchors.csv` 以及 **同源的三个 `*.pt` 表示数据集**。加入可精确动态规划的小型有限预算 MDP Q-gap 等价检查。
- `build_probe_bank.py`：使用与训练轨迹不重叠的外生种子建立独立 snapshot bank；目标 120 个合法 Edit 状态：预算=1、2、≥3 各40，Shock 20+Repeated 10+Small 10。按分层 reservoir 抽样；不足时明确记录缺口，不复制状态。保存环境快照、船舶 ID、修改前/后泊位安排、原场景哈希；逐条重放原始动作与奖励。
- `future_tapes.py`：**先生成外生未来，再分支执行**。使用 `state_id + tape_id + seed` 生成每条未来磁带；时间/事件排序与执行的动作无关，保留已经处理的事件、未处理的同一时刻事件以及已开始船舶的真实到港。注意：这是公开 ETA 条件下的**合成近似**，不是已校准的真实条件后验。
- `counterfactual.py`：同状态、同未来轨迹、Keep/Edit 配对，默认8条未来磁带；Greedy continuation，另可切换为冻结 A0 continuation。将 A0/A2/A3 的 Q-gap 与指定 continuation 的行动价值比较。输出 `counterfactual_pairs.csv`、`tape_pairs.csv`、`future_tapes.csv` 和**单条原始仿真未来**的 `original_future_pairs.csv`。记录近似置信区间并将跨零状态标记为 `uncertain`，不强行判错。
- `audit_targets.py`：对固定轨迹的宏转移只读计算 `reward`、逐条 bootstrap discount、next selected Q、bootstrap、TD target、Q(s,a)、TD error。`--full-state` 须与模型快照梯度版本一致，否则拒绝；只用 `best.pt` 时明确标记 target=online 为分析近似，**不是历史目标网络**。
- `audit_paired_targets.py`：**H1 强化诊断**。同一 probe 的 Keep/Edit 在同一未来磁带中构造各自**第一个决策时刻宏转移**，严格只穿过强制 Keep；并列输出两支的区间奖励、自举、终止、长度和 TD-gap。
- `summarize.py`：`gap_metrics.csv`、`target_groups.csv`、`paired_target_groups.csv`、`h1_joined_states.csv`、`mechanism_report.json`。记录 H1 观察、H2 尚未证伪、H3 暂缓。
- `scripts/run_v13_p0_p3.sh`：只打印命令；**只有 RUN=1 才执行**。不包含任何在线训练。

所有加载 `raw_episodes.pt`、`probe_bank.pt`、`training_state.pt` 的代码依赖本地可信 pickle。**不要加载不可信来源的训练状态**。模型 `best.pt` 通过 V1.2 的 `weights_only=True` 方式读取。

## 1. 安装及测试

从 IAMU 根目录解压新增包：

```bash
cd ~/code/iamu
unzip -o /path/to/IAMU_V13_P0_P3.zip
cd iamu_d3dqn
conda activate erldde
python -m unittest discover -s tests_v13 -p 'test_*.py' -v
bash v13_diagnostics/scripts/run_v13_p0_p3.sh # DRY RUN; 只打印
```

`tests_v13/test_real_env_v13.py` 应在你的完整 IAMU 仓库运行、不可 skip，验证真实环境 deepcopy snapshot 重放及保留公共未来。若此项无法通过，不应进入正式 P2 反事实。

## 2. P0/P1：同一原始训练轨迹 + 压缩审计

```bash
RUN=1 STAGE=collect bash v13_diagnostics/scripts/run_v13_p0_p3.sh
RUN=1 STAGE=audit bash v13_diagnostics/scripts/run_v13_p0_p3.sh
```

检查：

```bash
cat results/v13_p0_p3_seed43/compression/compression_audit.json
```

`max_ledger_error` 应在浮点容差内；检查三种表示的 `*.pt` 具有同一 `source_sha256`。这是**回报和决策语义保持**测试，不等于有限训练 Q 值等价证明。

## 3. P2：独立固定状态 + 共同未来配对

```bash
RUN=1 STAGE=probes bash v13_diagnostics/scripts/run_v13_p0_p3.sh
cat results/v13_p0_p3_seed43/probe/probe_audit.json
RUN=1 STAGE=counterfactual bash v13_diagnostics/scripts/run_v13_p0_p3.sh
```

检查 `probe_audit.json` 是否达到 120，以及各 `source_budget × scenario` 的 shortfall。若缺样，不能复制既有状态凑数；可使用**不同探针外生种子区间**增加独立数据，并在独立输出目录重跑。

本实现从原始 snapshot 保留已经发生的公共状态，未来外生磁带先独立生成，再在相同磁带中执行 Keep 与物理 Edit。每个磁带下的潜在到港与公告不随未来动作/随机数调用数量变化。未来更新使用公开 ETA 构造**合成近似条件模型**：可用于诊断对特定模拟模型的价值失准，但若需真实条件期望 Q，必须额外验证场景模型校准；不能将它写成最优 Q。

如果要测试冻结 A0 continuation（与 Greedy continuation 分开报告）：

```bash
RUN=1 STAGE=a0cf bash v13_diagnostics/scripts/run_v13_p0_p3.sh
```

可将并行反事实按固定 probe 索引切分，确保每个 `state_id/tape_id` 的外生随机数不因分片而改变。例如对已确认有 120 个状态的 bank，在 4 个终端分别执行命令，并使用 `--start-index/--stop-index` 为 `(0,30)、(30,60)、(60,90)、(90,120)`，每组 `--out` 用不同目录。`summarize.py --pairs` 可以同时接收多个分片的 `counterfactual_pairs.csv`，但各分片不得重叠或漏掉状态。

## 4. P3：TD target 的组成与共同状态双分支

```bash
RUN=1 STAGE=targets bash v13_diagnostics/scripts/run_v13_p0_p3.sh
RUN=1 STAGE=paired bash v13_diagnostics/scripts/run_v13_p0_p3.sh
RUN=1 STAGE=summary bash v13_diagnostics/scripts/run_v13_p0_p3.sh
```

输出主要是：

```
results/v13_p0_p3_seed43/
  collect/raw_episodes.pt
  collect/episodes.csv
  compression/compression_audit.json
  compression/primitive.pt
  compression/budget_tail.pt
  compression/decision_epoch.pt
  probe/probe_bank.pt
  probe/probe_states.csv
  counterfactual_greedy/counterfactual_pairs.csv
  counterfactual_greedy/future_tapes.csv
  counterfactual_greedy/original_future_pairs.csv
  targets_a0/target_audit.csv
  targets_a3/target_audit.csv
  paired_targets_a3/paired_target_components.csv
  paired_targets_a3/paired_target_gap.csv
  summary/gap_metrics.csv
  summary/target_groups.csv
  summary/paired_target_groups.csv
  summary/h1_joined_states.csv
  summary/mechanism_report.json
```

`paired_target_gap.csv` 测的是给定模型的**一条宏转移 TD 目标差**，不是优化控制的真实动作价值。`counterfactual_pairs.csv` 测的是指定 Greedy/A0 continuation 下的模拟动作价值，亦不是 Q*。这两个量必须保留不同列、不能互称为“真值”。

`target_audit.py` 默认使用 `best.pt` 的 target=online 近似。如要看真实的轮次末 target：只在 `training_state.pt` 的 online 权重与传入 V1.2 模型 checkpoint 完全相同时提供 `--full-state`；否则拒绝以避免混用最佳和最终权重。

## 5. 如何检验 H1/H2/H3

- **H1（Keep/Edit 价值传播不同步）**：先看 `h1_joined_states.csv` 中预算为 1、反事实正收益明确的 Edit 是否被 A3 漏选；再把同状态 `keep_bootstrap`、`edit_bootstrap`、宏长度与 `paired_td_gap` 关联。单看 Edit target 更负**不能**证明 H1。
- **H2（压缩经验复用过强）**：本批只完成 Replay 输入/TD 组成诊断；需要下一批 P4 的 `F0/F2/F3/F2-R/F3-R` **同原始数据、同初始参数**消融才能检验，不能提前宣布成立。
- **H3（观测不足）**：延期，第一批不增加状态特征/网络。

**本次尚未运行 200-episode 数据采集、120-state 真实反事实或任何 V1.3 在线训练。** 在当前隔离开发环境已完成数值/纯逻辑测试；真实 V1.1 环境集成测试需服务器执行。测试结束前不要发布效果结论。
