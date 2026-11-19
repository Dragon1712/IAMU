# IAMU-RL V0.8.1 — 完整数量验证并行化、异步快照选择与训练状态恢复

## 本版严格保留原有验证数量与指标

以 GitHub `Dragon1712/IAMU` 的 V0.8 `train_budgeted_ddqn_parallel.py` 为基础，仅修改并行训练入口，新增 CPU 验证模块、恢复状态模块与测试。**未修改** `rl_env.py`、`rl_ddqn.py`、`rl_experiment.py`、任何调度规则、奖励函数和风险模型。

原验证方法：`['keep','greedy','ddqn','periodic']`（`--clock union` 时；ETA-only 仍按原始代码保留三个方法）。默认案例 Case 8、9；3 种场景、3 个 seed、3 个预算，保持：

```
2 cases × 3 scenarios × 3 validation seeds × 3 budgets × 4 policies
= 54 independent groups × 4 original methods = 216 simulations per checkpoint
```

每 `--eval-every 25` 回合验证一次、训练 100 回合仍然 **4 轮、合计 864 次完整验证仿真**。没有删除 Greedy/Periodic，也没有缩小验证集。结果保存每轮完整 216 条 `runs.csv` 和 12 组 `summary.csv`，算法指标按原 `evaluate_paired()` 逐条重新聚合，包括均值、中位数、胜负次数、经验 CVaR90、平均改约次数等。

## 新架构

```
CPU training actors (4): 仍采用 V0.8 环境与动作机制
       ↓ 完整训练轨迹
GPU learner (cuda:2): 经验回放、Double DQN 梯度更新
       ├─ 每个验证节点：冻结模型参数为 validation_snapshots/val_XXXXXX_grad_YYYYYYYYY.pt
       │     保存训练状态 training_state.pt 后立即异步提交验证
       │          ↓
       │     独立 CPU validation coordinator → spawn CPU validation Pool (2～4)
       │          54 组 × 4 策略；按固定顺序聚合结果；输出实时进度
       │          ↓
       │     按“被验证的冻结模型”决定是否更新 best.pt
       └─ 验证期间继续更新网络，不等待 evaluate_paired() 返回
```

验证最多有一份快照正在执行 CPU 池计算，其余验证请求进入 **FIFO 队列，不丢弃、不跳过**。训练全部完成后仍需等所有排队验证结束才输出 `[complete]`；如果计算很多，异步验证的总耗时不会自动消失，只是可以与训练重叠。

**重要：** `best.pt` 从胜出的冻结 `validation_snapshots/*.pt` 加载保存，不从验证返回时的最新 Learner 参数保存。元数据包括：`validation_id`、`selected_episode`、`n_gradient_updates`、`selection_score`、`validated_snapshot_sha256`，可逐项审计。

## 文件

新增/修改：

```
IAMU_dynamic/
├── train_budgeted_ddqn_parallel.py       # 修改：异步提交/收集、即时进度/写盘、resume
├── parallel_validation_v081.py            # 新增：CPU 多进程验证池与 216 条仿真汇总
├── parallel_recovery_v081.py              # 新增：原子保存完整训练状态并恢复
├── PARALLEL_DDQN_V081.md                  # 新增：本说明
└── tests/
    └── test_parallel_validation_v081.py  # 新增：并行数目/数值/快照/恢复测试
```

输出（新目录，绝不覆盖旧 V0.8）：

```
results/v081_parallel_100/
├── manifest.json
├── train_curve.csv                # 每完成一回合即写盘
├── validation_curve.csv           # 每次完整验证后写盘
├── events.jsonl                   # 进度事件与 Worker PID
├── training_state.pt              # 原子训练恢复快照（optimizer/replay/RNG/待验证）
├── latest.pt                      # 最新网络权重（不一定通过验证，不能单独续训）
├── best.pt                        # 仅从“已完成验证的冻结快照”产生
├── validation_ledger.json         # 每轮快照、状态、记录数、SHA256
├── validation_snapshots/
│   ├── val_000025_grad_XXXXXXXXX.pt
│   ├── val_000050_grad_XXXXXXXXX.pt
│   ├── val_000075_grad_XXXXXXXXX.pt
│   └── val_000100_grad_XXXXXXXXX.pt
├── validation_details/
│   ├── val_000025/{runs.csv,summary.csv}
│   └── ...
└── performance.json
```

注意：以上为 100 回合/每 25 回合验证的示例结构。程序开始前只有 manifest/event，第一次训练回合完成后就有 `train_curve.csv`；到验证节点时先保存 snapshot 和 `training_state.pt`，不再出现原版“验证数小时无任何 checkpoint”的情形。

## 服务器部署：不要修改正在运行的 Python 进程

当前旧 V0.8 PID `3510088` 已加载旧代码。修改磁盘文件 **不能**使该进程变为 V0.8.1；而且旧 V0.8 只有 manifest 时，V0.8.1 的 `--resume` **不能**恢复原进程未保存的内存状态。建议先让旧任务结束（或评估丢失进度的代价后停止），在独立分支与独立结果目录启动 V0.8.1。

```bash
cd ~/code/iamu
git status
git switch -c experiment/v081-parallel-validation
# 将本交付补丁上传服务器后：
git apply --check /上传路径/IAMU_parallel_validation_v0.8.1.patch
git apply /上传路径/IAMU_parallel_validation_v0.8.1.patch
cd IAMU_dynamic
python -m unittest tests.test_parallel_validation_v081 -v
```

第一次建议先真实 50 船运行 25 回合、2 个验证 Worker 验证 216 条能顺利结束；正式 100 回合推荐：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 100 --workers 4 --validation-workers 2 \
  --prefetch-per-worker 2 --sync-every 4 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --val-seeds 400001:400003 \
  --clock union --candidate-limit 4 \
  --budget-features absolute --value-features global \
  --candidate-screen positive --keep-margin 0.1 \
  --greedy-exploration-prob 0.75 \
  --batch-size 64 --warmup 512 \
  --updates-per-step 0.25 --max-updates-per-episode 64 \
  --eval-every 25 --validation-progress-every 6 \
  --state-every 5 --log-every 1 --device cuda:0 \
  --out results/v081_parallel_100
```

**CUDA 提醒：** 上述设置 `CUDA_VISIBLE_DEVICES=2` 意味着进程只可见物理 GPU 2，PyTorch 内编号为 `cuda:0`，因此 `--device cuda:0` 正确。若未设置可见设备过滤器，则使用 `--device cuda:2` 指向物理 GPU 2。避免 `CUDA_VISIBLE_DEVICES=2 --device cuda:2` 组合，否则该进程中 GPU 2 的逻辑索引不存在。

## 查看实时日志

训练日志每条完整回合打印 `[train]`，验证日志输出：

```
[validation-queued] val_000025 frozen_grad=... runs=216 groups=54; training continues
[validation-start] val_000025 54 groups / 216 simulations ...
[validation-worker-online] ... pid=...
[validation-task-start] ... group=... scenario=... B=...
[validation-progress] val_000025 24/216 elapsed=... pids=[...]
[validation-finished] val_000025 wait=... validated_grad=... current_grad=... 216 runs
```

默认验证进度每 6 组（24 次仿真）汇报；长任务即使尚未完成一组，也有 15 秒进度心跳。`events.jsonl` 保存 Worker PID、任务开始、验证过程、最佳快照选择、梯度次数和恢复记录。

```bash
cd ~/code/iamu/IAMU_dynamic
# 仅用于检查流程，不要解析为独立测试结果
tail -f results/v081_parallel_100/events.jsonl
cat results/v081_parallel_100/validation_ledger.json
tail -5 results/v081_parallel_100/train_curve.csv
```

可按 `ps -o pid,ppid,stat,psr,%cpu,etime,wchan,args -p <PID> --ppid <PID>` 核对进程数量。验证 Worker 为另一层子进程（验证协调器的子进程），主进程 `pstree -ap <PID>` 可以看到。CPU 负载是否达到线性加速取决于实际环境耗时和同时运行的其他任务，不能仅凭架构假设。

## 完整训练状态恢复

`training_state.pt` **不是**简单的网络权重，保存：
- 在线网络、目标网络与 Adam 优化器全部状态；
- 梯度更新次数、ReplayBuffer 内容和采样随机数状态；
- Python/NumPy/PyTorch 主进程 RNG 状态；
- 已完成的 episode 编号和训练记录；
- 还未得到结果的完整验证请求、其模型快照路径、记录数量与 SHA256；
- 最佳验证分数和已有验证曲线；
- 尚未消费的梯度更新信用余额。

每 `--state-every 5` 个回合及**每轮验证开始前**、**每轮验证完成后**保存完整状态，并采用临时文件 + 原子重命名；每条训练记录立即写 CSV。`latest.pt` 只用于推理，**不能替代** `training_state.pt` 做完整续训。

若 V0.8.1 因异常退出：用原来的命令、**相同**案例/训练超参数/随机种子/输出目录，额外增加 `--resume`，例如：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  ...与首次运行完全相同的其他参数... \
  --device cuda:0 --out results/v081_parallel_100 --resume
```

若完整状态只保存至第 40 个回合、随后有第 41～43 回合未落盘，恢复会从第 40 个回合重新采样未被持久化的回合；可能重复计算，但不会把它们误计为已完成。未完成的异步验证会使用原冻结快照重新排队并按同一指标计算。

**重要限制：** 进程启动、任务完成顺序、Actor 私有随机数及在途任务无法逐字节恢复，属于**统计意义上的续训**，并非可位级重演的精确断点续训。完整状态包括 Python/NumPy 数据结构，`torch.load(weights_only=False)` 只应读取自己服务器上生成、可信任的训练状态文件；不可加载不明来源的 `.pt`。模型推断检查点仍使用 `weights_only=True`。

保存 replay buffer（例如 50,000 个转换）可能产生较大的文件和磁盘写入开销；可通过 `--state-every 10` 降低保存频率，但也增大意外退出时的潜在重算量。正式训练前请检查磁盘剩余空间。

## 正确性检查与局限

1. 原 `--clock union` 的验证次数**严格不变**，每一验证请求在 `validation_details/<id>/runs.csv` 包含原有的完整组合。
2. 同一冻结策略、同一物理随机种子、相同动作下，CPU 并行 vs 串行应有一致的最终等待/改约次数（见新增集成测试）；计算耗时可能不同。
3. `best.pt` 按 `validated_snapshot_sha256` 与具体快照绑定；即使 Learner 已执行更多梯度，也只保存验证过的版本。
4. 验证采用专用 CPU 池、训练继续执行；验证池可能和训练 Actor 竞争 CPU。对于 12 核 Xeon，建议先试 4 训练 Actor + 2 验证 Worker，再实测 4+4。
5. 验证中包含 Keep/Greedy/Periodic，对其使用同一冻结状态与调度器；不能因为训练只用 DDQN 选择 `best.pt` 就删除这些基线。
6. 没有更改 RL 状态、候选收益特征、奖励或 ERL-DDE/岸电模型；本版只是计算架构与容错能力的工程改进，并不保证训练收敛。

## 本地开发验证与未验证事项

本地使用与 IAMU 同一函数接口的轻量替身仿真完成：216/216 串行与并行等待一致、12/12 汇总一致；验证快照的 gradient_steps 与训练继续更新后的梯度不同，`best.pt` 与快照一致；待完成验证的状态恢复可重新排队并验证 216/216。新增真实 IAMU 集成测试需要在你的现有 Linux 环境运行，尚未在你的真实 50 船任务上测得加速比；请勿把轻量替身测试当作完整物理实例的性能结论。
