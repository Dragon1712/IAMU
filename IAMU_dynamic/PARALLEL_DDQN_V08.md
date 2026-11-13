# IAMU-RL V0.8 — 多 CPU 环境采样 + 单 GPU Double DQN Learner

## 定位与兼容性

本交付只新增 `train_budgeted_ddqn_parallel.py`、`benchmark_parallel_workers_v08.sh` 和一个集成测试文件；**不覆盖**原来的 `train_budgeted_ddqn.py`、V0.7 的 `rl_env.py`、`rl_ddqn.py`、ERL-DDE 权重或已有结果。兼容 V0.7 的 `candidate-screen`、`keep-margin`、`greedy-exploration-prob`、`budget-features`、`value-features` 参数。若你的仓库仍是 V0.6，可传入 `--budget-features legacy --value-features legacy` 并关闭 V0.7 特有开关。

目标机器：Intel Xeon Silver 4310，**12 物理核 / 24 逻辑线程**，4× RTX 3090。建议 **4 个 worker 起步**，再试 6 或 8；无需用满 24 个逻辑线程。每个 worker 限制为单线程 BLAS/PyTorch 计算，减少严重的线程超卖。Python `multiprocessing` 使用 `spawn`：CUDA 只在 learner 主进程初始化，worker 永远使用 CPU 前向推理，不对 GPU 进行跨进程共享。

```text
CPU actor 0 (独立 IAMUEventEnv, CPU policy snapshot)  ──┐
CPU actor 1 (独立 IAMUEventEnv, CPU policy snapshot)  ──┤
CPU actor 2 (独立 IAMUEventEnv, CPU policy snapshot)  ──┼─> CPU replay buffer
CPU actor 3 (独立 IAMUEventEnv, CPU policy snapshot)  ──┘         │
                                                         DDQN learner (cuda:0)
                                                验证集评估 + best.pt/latest.pt
```

每名 worker 生成**整个回合轨迹**后，将 `(s, action, reward, s', done)` 发送到 learner，后者统一进行重放与梯度更新。为了避免 learner 更新时 CPU 空闲，主进程收到结果后会**先排入下一条工作，再执行优化器更新**；每个 worker 至多提前排队 `--prefetch-per-worker` 个回合。每隔 `--sync-every` 个被分配的回合发送一次最新策略快照。

**研究边界：这是 off-policy、异步轨迹生成；不同 worker 以可能过时的网络快照进行探索。即使每个回合的 case/seed/扰动预先按序固定，训练更新的完成顺序和策略轨迹也不能与原串行 DDQN 逐步完全相同。比较性能需要锁定独立测试集、至少运行多个学习初始化 seed，记录吞吐与回报。**

## 1. 安装和环境验证

将 ZIP 解压后的 `IAMU_dynamic` 内容合并到 `~/code/iamu/IAMU_dynamic`。或在 GitHub 仓库根目录应用增量补丁：

```bash
cd ~/code/iamu
 git switch -c experiment/v08-parallel-ddqn
 git apply --check /上传路径/IAMU_parallel_ddqn_v0.8.patch
 git apply /上传路径/IAMU_parallel_ddqn_v0.8.patch
cd IAMU_dynamic
python -m unittest tests.test_parallel_training_v08 -v
python train_budgeted_ddqn_parallel.py --help
```

可先使用以下命令确认 PyTorch CUDA 版本与 GPU 设备索引（在你的 conda 环境中执行）：

```bash
python - <<'PY'
import torch
print('PyTorch',torch.__version__)
print('CUDA enabled',torch.cuda.is_available())
print('GPU count',torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
PY
```

不要只根据 `nvidia-smi` 瞬时 0% 判断 CUDA 没有工作：小型 Q 网络使用 GPU 的计算非常短，瓶颈仍可能在 CPU 环境生成。

## 2. 正式训练命令：建议先跑 100 个真实 50 船回合

```bash
cd ~/code/iamu/IAMU_dynamic
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --episodes 100 --workers 4 --prefetch-per-worker 2 --sync-every 4 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --val-seeds 400001:400003 \
  --clock union --candidate-limit 4 \
  --budget-features absolute --value-features global \
  --candidate-screen positive --min-gain 0.1 \
  --keep-margin 0.1 --greedy-exploration-prob 0.75 \
  --batch-size 64 --warmup 512 --updates-per-step 0.25 \
  --max-updates-per-episode 64 \
  --epsilon-decay-episodes 300 --eval-every 25 \
  --device cuda:0 \
  --out results/v08_parallel_100
```

**注意:** `CUDA_VISIBLE_DEVICES=0` 后，当前进程只可见 1 张显卡，应使用 `--device cuda:0`。如果设为 `CUDA_VISIBLE_DEVICES=1`，进程内看到的那张物理 GPU 1 仍叫 `cuda:0`。千万不要把 `--device cuda:1` 与单设备可见性组合。

对于 300 回合正式消融，可以复制上述命令，将 `--episodes` 提高到 300、`--eval-every` 改成 50，并把目录换成 `results/v08_parallel_300`。首次测试不建议一开始就设 8 workers；确认内存、CPU 核心负载和进程退出正常后再增加。

`--updates-per-step 0.25` 和 `--max-updates-per-episode 64` 是**吞吐量取舍**，与 V0.7 原始每条 transition 约 1 次 SGD 更新不同。若要严格比较更新预算，请在对照组使用相同数值，也可以把 `--updates-per-step` 改回 `1.0` 并将最大更新上限设为 `0`（禁用上限），但学习器可能成为新瓶颈。**不能把 GPU 使用率上涨等同于策略提高**。

## 3. 对照测试：测真实的 worker 数量收益

先停下或避开其他大的 V0.7 训练，以免争抢 CPU。执行：

```bash
EPISODES=40 WORKERS_SET='1 4 6 8' DEVICE=cpu bash benchmark_parallel_workers_v08.sh
```

它固定案例、随机种子、扰动和初始网络设置，关闭梯度更新，专门检查环境采样的 wall time 与吞吐差异。每组的 `performance.json` 记录：

- `workers_used`：实际产出结果的独立 CPU PID 数目；
- `avg_episodes_per_hour`：含启动、经验收集及验证的端到端吞吐；
- `avg_actor_seconds`：单个轨迹平均计算时间；
- `avg_learner_seconds`：梯度更新耗时（禁用梯度时为 0）；
- `total_wall_seconds`：最终总耗时。

采用真实实例的基准结果，不能用 6 船演示样本或轻量 mock 环境的吞吐推算 50 船服务器性能。

## 4. 生成文件和实验一致性

```text
results/v08_parallel_100/
├── manifest.json           # 案例、种子、worker / 同步间隔 / 网络参数
├── train_curve.csv         # episode、worker PID、policy 版本、等待、预算、loss
├── validation_curve.csv    # 固定验证集表现
├── best.pt                 # 验证集最佳 checkpoint
├── latest.pt               # 最近一次验证 checkpoint
└── performance.json        # 工作者实际使用数、吞吐、学习器与环境耗时
```

模型仍通过 `DoubleDQNAgent.checkpoint(meta)` 保存，与 V0.7 的 `evaluate_budgeted_ddqn.py` 兼容；独立评估请继续使用现有 V0.7 独立测试命令，并保证 `case_code/10,11` 等最终测试实例不曾用于调参。

可选 `--max-idle-seconds 1800`：如果连续 30 分钟没有任何回合返回则报错并清理工作进程，默认 0（关闭），因为真实 50 船场景的单回合耗时尚无实测依据。`--worker-timeout` 仅控制进程存活检查和状态提示，不会误杀仍在执行的长回合。

**可靠性：** worker 内不共享港口实例状态、不使用 GPU、无交叉种子；使用训练 / 验证案例内容哈希检查。每条轨迹检查 `sum(rewards) * reward_scale = -weighted_wait` 和 `spent <= budget`。worker Python 异常带完整栈返回主进程，CUDA 初始化前使用 `spawn`，主进程在 `finally` 中终止并回收子进程。训练最优权重采用临时文件后原子替换，避免部分写入。**未实现完整经验回放/优化器的断点续训**；`latest.pt` 是推理/复训初始化检查点，不要当成完整训练状态。中断后不会自动恢复剩余回合。

## 5. 已执行的测试边界

当前交付的脚本已经用 Python 3、CPU PyTorch 和一个独立的**接口兼容轻量仿真替身**完成 4-worker 16-episode / 28-update 的端到端试跑，以及 2-worker 6-episode 的子进程集成测试；产生真实可加载的网络 `best.pt` 和报告。**交付环境未连接你的服务器 RTX 3090，也未在真实 50 船 `SlotPlanner` 上完成加速基准**；建议上传并运行上方真实案例的集成测试与 worker 数量消融，再根据结果选择最优 worker 数量。
