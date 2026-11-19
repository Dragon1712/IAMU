# IAMU-RL V0.8.2 — 四 CPU 训练 → 冻结 → 四 CPU 验证 → 更新最佳检查点

## 目的与基线

在 V0.8.1 的多 CPU Actor + 独立并行验证实现基础上，将**训练和验证从异步重叠执行改为严格的同步阶段**。保持 V0.7 的泊位规划器、动态环境、状态/动作、候选评估、DDQN、奖励函数、训练回放和配对验证指标不变。当前 GitHub `main` 原版是 V0.8（`3b737b3`）；可使用附带的 **V0.8→V0.8.2** 补丁，或者在本地已有 V0.8.1 的分支上使用 **V0.8.1→V0.8.2** 补丁。

每个阶段采用以下流程：

```text
Phase 1: 4 个训练 Actor 并行采集当前阶段的 Episode；GPU Learner 更新 DDQN
        ↓ 等待所有预取/在途 Episode 完成；不得派发下一阶段 Episode
Phase 2: 冻结当前参数为 validation_snapshots/val_XXXXXX_grad_XXXXXXXXX.pt
        ↓ 原子保存 training_state.pt、latest.pt、train_curve.csv
Phase 3: 4 个验证 CPU 进程并行运行原来全部 Case × Scenario × Seed × Budget × Policy
        ↓ 训练 Actor 不采集新数据，Learner 不执行梯度更新
Phase 4: 汇总完整验证，按照冻结快照成绩更新 best.pt，并保存验证记录及恢复状态
        ↓ 完成后才派发下一阶段 Episode
```

例如 `--episodes 100 --eval-every 25 --workers 4 --validation-workers 4`：

- Episode 1–25 → 冻结 `val_000025` → 全量验证 → 更新 `best.pt`
- Episode 26–50 → 冻结 `val_000050` → 全量验证 → 更新 `best.pt`
- Episode 51–75 → 冻结 `val_000075` → 全量验证 → 更新 `best.pt`
- Episode 76–100 → 冻结 `val_000100` → 全量验证 → 更新 `best.pt`

**验证次数不变：** 两个验证 Case、三种 Scenario、三个 Seed、三个 Budget、四种策略 (`keep,greedy,ddqn,periodic`)，每次共 **2×3×3×3×4 = 216 条仿真**，100 回合、25 回合一次共 **864 条**。`--clock` 非 `union` 时依旧继承原来不含 Periodic 的验证配置；本版本不改变其原有数量。

## 为什么需要同时修改任务预取边界？

若只在 `completed==25` 处等待验证，原 V0.8.1 可能已经预派发 Episode 26～32。它们会在验证时继续计算，违反严格同步要求。V0.8.2 在 `assign()` 限制 `pending[0].episode <= stage_limit`，且冻结前要求 `queued==completed`，这保证所有在途采样结果已回收到 Learner、GPU 更新已完成。阶段恢复时同样先完成未返回的验证，再派发新的训练任务。

## 在 12 核 24 线程 Xeon Silver 4310 上部署

推荐保持每个进程内部 BLAS/torch 线程数为 1，避免超额订阅。即使训练进程和验证进程都存在，**两个计算池不会在同一时间运行实际仿真**。训练 Actor 在验证阶段保持空闲，验证进程池在训练阶段尚未开始验证。

如果物理 GPU2 需要在程序中使用：`CUDA_VISIBLE_DEVICES=2` 之后进程内填写 `--device cuda:0`，因为屏蔽其他 GPU 后序号重新从 0 开始。

```bash
cd ~/code/iamu/IAMU_dynamic
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 100 --workers 4 --validation-workers 4 \
  --prefetch-per-worker 2 --sync-every 4 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --val-seeds 400001:400003 \
  --clock union --candidate-limit 4 \
  --budget-features absolute --value-features global \
  --candidate-screen positive --keep-margin 0.1 \
  --greedy-exploration-prob 0.75 --batch-size 64 --warmup 512 \
  --updates-per-step 0.25 --max-updates-per-episode 64 \
  --eval-every 25 --state-every 5 --validation-progress-every 6 \
  --device cuda:0 --out results/v082_phased_100
```

**请使用新输出目录。** 不建议中断已经在运行的 V0.8/V0.8.1 进程来切换 Python 源码；旧进程不会动态重载新实现。V0.8.2 `--resume` 只接受由 V0.8.2 生成的 `training_state.pt`，有意拒绝异步版 V0.8.1 的状态，以免错误解释在验证期间已经完成的训练回合。

## 日志、快照与恢复

- `events.jsonl`：按事件记录 `phase_training_start`、`phase_training_end`、`validation_queued`、`phase_validation_start`、`validation_progress`、`validation_finished`、`phase_validation_end` 及 Actor 状态。
- `train_curve.csv`：每完成一条轨迹即时写入，包括 Worker PID、Learner 梯度步数、GPU 更新耗时及策略滞后。
- `validation_curve.csv`：包括 `gradient_steps`、`learner_gradient_steps_at_receive`、`evaluated_runs`、`snapshot_sha256` 和验证耗时。同步版本要求前两个值相等。
- `validation_ledger.json`：验证快照路径、SHA256、是否已完成及分数。
- `validation_snapshots/`：每次验证的**冻结且不可变**网络快照；`best.pt` 从已验证快照构建，不使用验证之后的当前权重。
- `training_state.pt`：持久化网络、目标网络、优化器、回放池、随机数及已完成回合/验证状态。重新运行原命令附加 `--resume` 即可恢复 V0.8.2 的保存进度。如果在验证中断，先重新评价**原冻结快照**，再执行后续 Episode。已采集但未写入持久状态的在途轨迹需要重跑。

`training_state.pt` 可能包含复杂 Python 对象，仅加载本机可信实验文件；请勿下载或使用不可信训练状态。

## 一致性与边界

1. 给定同一冻结模型及固定验证参数，216 次并行验证计算的等待/预算应与串行结果一致（先前 V0.8.1 已实现该检查，V0.8.2 未改动 `evaluate_paired` 的计算）。
2. 阶段结束时 `queued==completed`，验证开始/结束间不会出现 `actor_start` / `train_completed` 事件，且 `gradient_steps` 保持不变。
3. 该改动**不会让训练与原始 V0.7 串行版逐位一致**：4 个 Actor 仍然可能以不同完成顺序写入 Replay Buffer。
4. 由于训练/验证不再重叠，V0.8.2 的墙钟时间未必比 V0.8.1 异步版本短；它优先保证各阶段互不影响、模型检查点可审计。
5. 实测 CPU 加速仍需在真实 Case 3–9 环境验证；当前自动测试包含轻量接口替身和小尺寸 IAMU 案例，不能作为 50 船真实性能证据。

## 验证命令

```bash
python -m unittest tests.test_sync_phased_v082 -v
python -m unittest tests.test_parallel_validation_v081 -v
```

另可在服务器运行后检查每轮事件顺序：

```bash
grep -E '"event": "(phase_training_end|phase_validation_start|validation_finished|phase_validation_end|phase_training_start)"' \
  results/v082_phased_100/events.jsonl
```

期望 `phase_training_end` → `phase_validation_start` → `validation_finished` → `phase_validation_end` → 下一个 `phase_training_start`。
