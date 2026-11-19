# IAMU V1.2 机制诊断与固定轨迹实验：P0—P4 离线代码增量包

**基准**：`Dragon1712/IAMU`，提交 `70606c7b7ac4c6fc9063e25c92f56f58e1061c76`。**本包不包含历史 `.pt`**，无需服务器访问 GitHub。GitHub 连接对本会话仍拒绝写分支（403），因此这是需要在服务器应用的补丁，而不是已经写到远端的提交。

## 重要：不能在需要恢复旧实验的工作目录中直接修改源码

现有 `v12_protocol/train.py` 会对所有 `v12_protocol/*.py` 求哈希；本次新增文件或替换 `agent.py` 等会改变严格恢复指纹。**即使不改变默认训练行为，旧的 V1.2 `training_state.pt` 也不能使用本包的代码继续严格 resume**。历史实验的 `results` 和原研究代码均须只读。

### 1. 先创建独立的代码副本（只复制源码，不复制 `.pt`）

```bash
export SRC=/home/moco/code/iamu/iamu_v12
export DST=/home/moco/code/iamu/iamu_v12_next
mkdir -p "$DST/iamu_d3dqn"
rsync -a --exclude='/results/' --exclude='*.pt' --exclude='*.log' \
  "$SRC/iamu_d3dqn/" "$DST/iamu_d3dqn/"
ln -s "$SRC/2024LiERLDDE" "$DST/2024LiERLDDE"
```

如 `DST/2024LiERLDDE` 已存在，勿重复创建链接；检查案例内容再继续。假设 ZIP 解压到 `/home/moco/code/iamu/iamu_v12_next_patch/`：

```bash
python /home/moco/code/iamu/iamu_v12_next_patch/apply_next.py \
  --repo "$DST/iamu_d3dqn" --check
python /home/moco/code/iamu/iamu_v12_next_patch/apply_next.py \
  --repo "$DST/iamu_d3dqn" --apply
```

安装器使用**Git blob SHA-1**核验 `replay.py`、`agent.py`、`fixed_data.py`、`validation.py` 四个源码文件恰好等于该基准提交；不调用 Git 命令，不访问网络。任一文件已变动、新模块已存在、必需文件缺失时拒绝覆盖。被替换文件会在代码目录外生成备份 `p0p4_backup_...`。不会改动 `iamu_dynamic/`、`compress.py`、`transitions.py`、原始 checkpoint 或历史 CSV。若校验失败，不要 `--force` 或手工覆盖；应比较当前源码后重新适配补丁。

### 2. 运行验收

```bash
cd "$DST/iamu_d3dqn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest discover -s tests_v12 -p 'test_*.py' -v
python -m v12_protocol.probe_bank --help
python -m v12_protocol.probe_eval --help
python -m v12_protocol.counterfactual_probe --help
python -m v12_protocol.analyze_diagnostics --help
python -m v12_protocol.fixed_data train --help
```

本包另有一份选择性命令脚本 `run_next_linux.sh`；不会自动启动任何 400 episode 在线实验：

```bash
bash /home/moco/code/iamu/iamu_v12_next_patch/run_next_linux.sh help
bash /home/moco/code/iamu/iamu_v12_next_patch/run_next_linux.sh unit
```

### 3. 收集独立 train_raw/probe_raw（只训练案例1–7）

这会运行**真实 CPU 仿真**，不会训练 400-episode 在线主实验：

```bash
RUN=/home/moco/code/iamu/iamu_v12_next_patch/run_next_linux.sh
bash "$RUN" collect-train         # 64 episodes, scenario seed 起点 200000
bash "$RUN" collect-probe         # 32 episodes, 独立 scenario seed 起点 300000
```

两套数据均沿用现有每事件 `Keep/Greedy/Uniform legal` 三等分公开行为混合，不改变物理环境。输出：

```text
$DST/iamu_d3dqn/results/v12_pnext_data/train_raw_seed42.pt
$DST/iamu_d3dqn/results/v12_pnext_data/probe_raw_seed42.pt
```

注意这是**本地可信生成的 `.pt`**，`torch.load(weights_only=False)` 不应用于不可信文件。`fixed_data.collect` 的原入口与原数据格式不变。

### 4. P2 公共状态库（与 A0—A3 的压缩无关）

```bash
bash "$RUN" bank-smoke      # 上限8个probe，完整复现候选指纹
bash "$RUN" restore-smoke   # 独立再次恢复8个probe
bash "$RUN" bank-full       # 最多144个probe，seed=1729, 每层最多8个
bash "$RUN" restore-full    # 完整审计，可能消耗较多 CPU 时间
```

共同状态来自原始 `probe_raw`：按 `scenario × initial_budget × remaining_budget{1,>1}` 分层；空层记0，不重复填满。主键为 `(source_file_sha256, episode_id, raw_step_index)`，包含公开观察数组、候选船ID顺序、**整个候选方案/当前计划/公开ETA哈希**、完整行为前缀与场景生成信息。恢复时重新读取案例、生成相同场景并重放前缀；候选排序或隐藏状态路径不匹配即失败。状态ID及隐信息**不会输入学习网络**。

### 5. P3 旧 checkpoint Q-gap 诊断和成对反事实

先清点**原始**研究目录保存的历史模型；模型 `.pt` 仍在原位置：

```bash
bash "$RUN" checkpoints
# 示例：使用 find 输出的真实路径，不猜测梯度编号：
CP_A0=$(find "$SRC/iamu_d3dqn/results/v12_A0_seed42_400/validation_snapshots" \
  -name 'val_000100_grad_*.pt' -print -quit)
test -n "$CP_A0" && test -f "$CP_A0"
cd "$DST/iamu_d3dqn"
python -m v12_protocol.probe_eval \
  --bank results/v12_pnext_data/probe_bank.pt \
  --data results/v12_pnext_data/probe_raw_seed42.pt \
  --checkpoint "A0_42_EP100=$CP_A0" --device cpu \
  --out results/v12_pnext_qgap_example
```

可在 `--checkpoint` 后继续添加 `A0/A1/A2/A3 × EP100/200/300/400` 现有 `.pt` 对照；缺失路径会写入 `missing_checkpoints.csv`，不会伪造模型。`V12Agent.from_checkpoint` 只恢复 online 模型，不可用于还原历史训练时的 Target/optimizer/TD-error。

反事实首先仅运行8状态 smoke，随后再启动最多144状态：

```bash
bash "$RUN" cf-smoke
bash "$RUN" cf-full
```

输出 `counterfactual_labels.csv` 标记 `label_kind=realized_path_fixed_continuation`，默认 Greedy 后续策略。所有 Keep/Edit 分支从**相同真实场景与完整动作历史重建的仿真器**深拷贝分叉；对比起点之后累计奖励。它是单条已实现未来下的路径条件收益，不是期望 Q 或 Q*。

若已经生成全量 `probe_qgap.csv` 和 `counterfactual_labels.csv`：

```bash
python -m v12_protocol.analyze_diagnostics \
  --qgap results/v12_pnext_qgap_example/probe_qgap.csv \
  --labels results/v12_pnext_data/counterfactual_full/counterfactual_labels.csv \
  --out results/v12_pnext_pathwise_analysis
```

对应字段有 `pathwise_wrong_keep`、`pathwise_harmful_edit`、`missed_gain`、`decision_regret` 与 `gap_path_residual`；按 scenario / 初始预算 / 剩余预算分别报告真实分母。不会把状态视为独立训练种子。

### 6. P4 固定数据 F1：四组、同一原始轨迹、从零初始化、各1000步

先确认公共数据审计通过。**F1 不继续训练旧 V1.2 模型**。可以先单独运行一组：

```bash
bash "$RUN" f1 A0
bash "$RUN" f1 A1
bash "$RUN" f1 A2
bash "$RUN" f1 A3
```

默认使用 CPU，有助于重复性；`F1_DEVICE=cuda:0 bash "$RUN" f1 A0` 可指定 GPU。当前机型 4×3090、24核、512GB，四个 F1 任务无需为每组复制源码。`F1_SKIP_VALIDATION=1` 可做纯训练机制验收，此时 `complete.json` 会明确写明**未做闭环验证**。完整 F1 应在最后对每组各做一次同协议闭环验证，开启带指纹的静态基线缓存。输出位于新的 `results/v12_pnext_F1_A{0..3}_seed42/`；同路径非空即拒绝覆盖。

每组固定记录：

- `manifest.json`：数据源、独立probe、代码快照SHA、初始化权重SHA、训练种子、设备、总更新、验证执行状态。
- `conversion_audit.csv`：每episode 前缀/宏/原始成本守恒。
- `train_curve.csv`：梯度步、sample_draws、normalized_reuse。
- `td_diagnostics.csv`：每25步抽样、每250步汇总的真实TD尺度，含互斥动作类别和宏长度分层；空格 `n=0` 且统计空白，不伪装零误差。
- `probe_qgap.csv` 与 `probe_metrics.csv`：共用固定状态；初始步及每250梯度步。
- `checkpoints/grad_*.pt` 与 `final.pt`：**只供推理的模型快照，不是完整训练恢复状态**。
- `validation/runs.csv`、`validation/summary.csv`：只在未指定 `--skip-validation` 且验证确实完成时存在。
- `complete.json`：完成所有实际要求的步骤后写入。

A1 保持 primitive/stratified rho=.25，A2 budget_tail/uniform，A3 decision_epoch/uniform。`gamma=1`、`reward_scale=100`、`target_update=250`、`batch=64`、`lr=3e-4`、`set_ddqn hidden96`、无margin，与要求一致。

### 7. P5 之后的 F2 选择门槛（本包不自动运行）

观察 F1：若压缩组仍退化或 TD/样本复用异常，再对固定 Replay 大小为 `N_g` 的组给出 `U_g=max(1,round(1000*N_g/N_A0))` 的匹配复用预算，并报告 `target_update=250` 造成的同步次数变化。**本包不直接实现新的在线方法、不运行 F2、不启动 12组400 episode 训练，也不使用 Case10–14 调参。**

### 8. 实测与限制

在本对话容器中，实际源码缺少完整 GitHub checkout（只能通过只读 GitHub 连接审查源码），本包完成了**11 项基于兼容接口替身的离线确定性测试**，结果见 `TEST_REPORT.txt`。这些测试**不是**实际 V1.1 港口仿真、真实 Set-aware DDQN、CPU端到端恢复或 GPU 基准；这些验收需在服务器补做。GitHub 分支创建依旧因集成权限403失败。

测试入口与基础 CLI 已完成 Python/Bash 语法检查。`apply_next.py` 未在你的实际服务器上执行，`train_raw/probe_raw/probe_bank/F1` 数据文件也尚未生成。请不要把脚本能力当成已取得的实验证据。
