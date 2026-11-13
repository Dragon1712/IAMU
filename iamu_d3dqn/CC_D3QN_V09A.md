# IAMU-RL V0.9-A：基础 Candidate-Conditioned Dueling Double DQN（CC-D3QN）

基线：GitHub `Dragon1712/IAMU`，提交 `b5a8183e010f4062a178ed42c8501430515d37ee`。本补丁**只更换 Q 网络表示**；RL 环境（24维全局/24维候选）、奖励、ETA 场景、Keep margin、经验回放、Double DQN target、候选筛选、8+8 CPU 分阶段训练/验证及每轮 216 次验证保持不变。

## 实现

`--q-network` 选项：

- `ddqn`（默认）：原始 `CandidateQNet`；与 V0.8.2 相同的直接 Q 估计。原有 100/300 回合结果不受影响。
- `d3qn`：新 `CandidateDuelingQNet`。全局状态单独编码，Value 网络接收状态编码；Advantage 网络对 `[状态编码, 候选动作特征]` 逐项共享评分。**当前不使用候选集合编码器，不添加风险损失，属于基础版 V0.9-A。**

对于合法集合 `M(s)`：

```
V(s) = Value(StateEncoder(global_features))
A(s,a) = Advantage(concat(StateEncoder(global_features), action_features[a]))
Q(s,a) = V(s) + A(s,a) - sum_{j in M(s)} A(s,j) / |M(s)|
```

**Keep 始终合法**，均值分母至少为 1；仅 Keep 合法时 `Q_keep=V`。填充/非法动作不进入均值。网络返回所有动作的 Q，实际动作选择与 Double DQN 目标计算仍使用合法掩码。Keep margin 只对动作选择打分生效。当前和下一状态的掩码分别传入训练 Q 计算。TD Loss、梯度裁剪、target 更新等均保留原算法。

参数量（隐藏层 96）：原 DDQN 14,113；D3QN 23,522。DDQN-130 为 23,531，可用于参数量匹配对照。

## 复制/应用补丁

在 `~/code/iamu` 根目录：

```bash
# 建议从干净的指定基线提交上建立新分支
 git checkout b5a8183e010f4062a178ed42c8501430515d37ee
 git switch -c experiment/v09a-cc-d3qn
 git apply --check /上传路径/IAMU_CC_D3QN_v0.9A.patch
 git apply /上传路径/IAMU_CC_D3QN_v0.9A.patch
```

如果本地分支有修改，请先保存或提交，不要盲目覆盖。补丁对当前指定的基线构建、验证；若 GitHub `main` 之后修改了相关文件，`git apply --check` 可能提示冲突。

## 测试

```bash
cd ~/code/iamu/IAMU_dynamic
python -m unittest tests.test_cc_d3qn_v09 -v
python -m unittest tests.test_cc_d3qn_parallel_v09 -v
# 原 DDQN 回归测试仍可运行
python -m unittest tests.test_parallel_validation_v081 tests.test_sync_phased_v082 -v
```

第二组测试使用**真实 IAMU 代码的小型 TSV 案例**，完整运行 4 Actor、4 验证 Worker、216 次仿真，检查冻结权重、best、恢复；不是 50 艘船正式性能测试。

## 300 回合、8 Actor + 8 验证 Worker

```bash
cd ~/code/iamu/IAMU_dynamic
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --q-network d3qn --hidden-dim 96 \
  --episodes 300 --workers 8 --validation-workers 8 \
  --prefetch-per-worker 1 --sync-every 1 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --val-seeds 400001:400003 \
  --clock union --candidate-limit 4 \
  --budget-features absolute --value-features global \
  --candidate-screen positive --keep-margin 0.1 \
  --greedy-exploration-prob 0.75 \
  --batch-size 64 --warmup 512 --updates-per-step 0.25 \
  --max-updates-per-episode 64 \
  --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-episodes 300 \
  --eval-every 25 --validation-progress-every 6 --state-every 5 \
  --seed 42 --device cuda:0 --out results/v09a_d3qn_seed42_300 \
  2>&1 | tee v09a_d3qn_seed42_300.log
```

`CUDA_VISIBLE_DEVICES=2` 将物理 GPU 2 映射为内部 `cuda:0`。调整设备编号以适应实际服务器。若只启动 8 个 CPU 仿真进程，除 Learner/Coordinator 以外仍会有系统进程；这里是每阶段 8 个 CPU Worker，并非硬 CPU 核心亲和性绑定。

在指定的完整协议下，每轮有 `2 cases × 3 scenarios × 3 seeds × 3 budgets × 4 methods = 216` 次验证。300 回合共 12 轮，2,592 次验证。为保持原 CSV 兼容，策略行仍标为 `method=ddqn`，实际架构以 `manifest.json -> training_args.q_network`、模型检查点的 `q_network` 字段为准。运行 D3QN 与 DDQN 要使用**不同输出目录**。

## 模型加载与恢复

- 旧 DDQN 推理检查点缺少架构字段时，默认按 `ddqn` 加载。
- V0.9-A 产生的 `best.pt` / `latest.pt` / 验证快照中包含 `q_network` 和 `network_architecture_version=1`，验证 Worker 自动加载相应网络。
- `--resume` 会校验实验配置摘要；D3QN 不能从 DDQN 的 optimizer/网络状态直接续训；新旧 DDQN 状态的摘要兼容处理不改变训练语义。
- `training_state.pt` 恢复涉及非纯张量的 Transition / RNG 对象，只能加载自己信任的本地状态文件。
- 旧的序列化权重不能直接加载到新的 D3QN；D3QN 应重新训练。

## 本地已验证范围及局限

当前交付已通过 13 项网络/恢复单元测试，并在**测试替身**环境下通过 4 Actor + 4 Validation Worker 的完整 216 条验证、一轮冻结 best 和恢复测试（共 14 项）；同时通过 2+2 Worker 简短训练及验证。容器可用内存限制为 4 GiB，8+8 进程测试触发系统 OOM，被终止，因此本地**没有**完成 8+8 的端到端验证。也**没有**在真实 50 船港口案例上执行训练或比较 D3QN 与 DDQN 的优化成绩。

基于现有真实仓库的代码静态核对，原有 `parallel_validation_v081.py` 不必修改：它已通过 `DoubleDQNAgent.from_checkpoint` 从冻结快照创建验证 Agent。正式启动前，请在你的真实服务器运行上方的集成测试与较小规模试跑，确认资源、MPI/CPU 环境和完整 checkpoint 恢复均符合预期。
