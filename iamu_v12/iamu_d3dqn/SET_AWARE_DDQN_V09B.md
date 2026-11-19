# IAMU-RL V0.9-B：Set-aware Candidate-Conditioned Double DQN

## 基线和设计边界

针对 GitHub `Dragon1712/IAMU` 中 `iamu_d3dqn/` 的 V0.9-A 实现（核验源文件 Git Blob SHA：`iamu_dynamic/rl_ddqn.py` = `7ff4da56444a88e4bca82e35aea7504f7b5f84ff`，`train_budgeted_ddqn_parallel.py` = `9120e017014f25c4927646be14cc7a7698325126`）。

本版本**仅**增加 `set_ddqn` 网络选项；原 `ddqn` 和 `d3qn` 的网络、Double DQN target、Huber Loss、学习率、奖励、动作筛选、Keep margin、经验回放、CPU rollout/validation 和 frozen checkpoint 协议不变。**不是**带 Dueling 分支的 `set_d3qn`，也没有引入 CVaR/风险惩罚。

> 无需重训已有 DDQN-96、DDQN-130、CC-D3QN-96；Set-aware DDQN 必须从头训练，不能跨架构 `--resume`。

## 核心公式

当前观测：全局状态 `g ∈ R^24`，候选特征 `x_i ∈ R^24`，动作 0 是 Keep，Edit 最多四个。`m_i ∈ {0,1}` 是合法掩码。

```text
h = ReLU(W_s g + b_s)                         (24 -> H)
e_i = phi(x_i)                               (24 -> 64 -> 32)
z = sum_{i > 0} m_i e_i / max(1, sum_{i>0} m_i)     (32)
Q(g,x_i,M) = QHead([h, z, x_i])             (H+32+24 -> H -> 1)
```

- 集合编码**只包含合法 Edit**，**不包含 Keep**。
- 当没有合法 Edit 时，`z` 精确为零。
- 对所有候选使用同一个 `phi`；池化与候选顺序无关，动作 Q 随候选交换而同步交换。
- 掩码会在编码**之前**清除非法候选的特征，避免非法填充值污染合法候选（包括异常的 Inf）。
- `set_ddqn` 不含 Value / Advantage 分支，**不计算 Dueling Advantage 均值**；输出直接是候选动作条件化的 Q。
- 现有的 `obs_mask`、`next_mask` 和 Double DQN bootstrap 保持原样。Keep margin 仅作用于动作选择时的分数。
- H=96 时 Set-aware DDQN 参数量 **20,865**；DDQN-96 为 14,113，DDQN-130 为 23,531，CC-D3QN-96 为 23,522。未来论文中须控制网络容量。

## 文件清单（以 GitHub 仓库根目录为相对路径）

```text
iamu_d3dqn/
    iamu_dynamic/rl_ddqn.py               [修改，新增两类]
    train_budgeted_ddqn_parallel.py        [修改，--q-network set_ddqn]
    tests/test_set_aware_ddqn_v09b.py      [新增，15项模型/恢复测试]
    tests/test_set_aware_parallel_v09b.py  [新增，4+4 Worker真实小案例集成测试]
    SET_AWARE_DDQN_V09B.md                 [新增，本文]
```

`parallel_recovery_v081.py` 无需更改：现有 `config_digest` 会将 `set_ddqn` 纳入摘要并阻止跨架构恢复。`parallel_validation_v081.py` 无需更改：它使用 `DoubleDQNAgent.from_checkpoint()`，新增模型的类型和网络架构版本由检查点决定。

`train_budgeted_ddqn.py`（串行旧训练入口）暂未新增 `--q-network`；本版本以 V0.8.2 **并行入口**为正式训练程序。

## Git 补丁安装

在**仓库根目录**执行：

```bash
cd ~/code/iamu
git status --short
# 建议在干净工作区，从已具备 V0.9-A 的提交/分支创建新实验分支：
git switch -c experiment/v09b-set-aware-ddqn

git apply --check /path/to/IAMU_SetAware_DDQN_v09B.patch
git apply /path/to/IAMU_SetAware_DDQN_v09B.patch
```

如果上面的 `git apply --check` 失败，先核对 GitHub 与本地的源文件版本；不要盲目 `--reject` 或覆盖正在运行的 V0.9-A 进程所引用的源码。独立目录或新分支更安全。

## 单元和端到端验证

在服务器安装/应用补丁后：

```bash
cd ~/code/iamu/iamu_d3dqn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest tests.test_set_aware_ddqn_v09b -v
python -m unittest tests.test_cc_d3qn_v09 -v
python -m unittest tests.test_set_aware_parallel_v09b -v
```

最后一项通过真实 IAMU 的三个船舶的小型 TSV 测试案例，要求 4 个 rollout Actor、4 个 CPU Validation Worker、**216 次配对验证**、正确冻结最佳检查点和 `--resume`。与真实 50 船 300 回合正式实验不同。

## 300 回合、8+8 Worker 实验命令

与三组 Seed42 的原有配置一致，只变更网络类型和输出文件夹。

```bash
cd ~/code/iamu/iamu_d3dqn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --q-network set_ddqn --hidden-dim 96 \
  --episodes 300 --workers 8 --validation-workers 8 \
  --prefetch-per-worker 1 --sync-every 1 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --validation-uncertainty 8 \
  --val-seeds 400001:400003 --updates 2 \
  --clock union --period 12 --candidate-limit 4 \
  --budget-features absolute --value-features global \
  --candidate-screen positive --min-gain .1 \
  --keep-margin .1 --greedy-exploration-prob .75 \
  --reward-scale 100 --lr .0003 --gamma 1 \
  --target-update 250 --replay-capacity 50000 \
  --batch-size 64 --warmup 512 --updates-per-step .25 \
  --max-updates-per-episode 64 \
  --epsilon-start 1 --epsilon-end .05 --epsilon-decay-episodes 300 \
  --eval-every 25 --state-every 5 --validation-progress-every 6 \
  --seed 42 --device cuda:0 \
  --out results/v09b_set_ddqn_seed42_300 \
  2>&1 | tee v09b_set_ddqn_seed42_300.log
```

`CUDA_VISIBLE_DEVICES=2` 把物理 GPU 2 映射为进程内的 `cuda:0`。可按服务器实际情况调整。每轮 2 Cases × 3 Scenarios × 3 Seeds × 3 Budgets × 4 Methods = 216 次验证，12轮共2592次。

**注意**：现有验证写出的 `method=ddqn` 是 *学习策略槽位名称*，不代表网络结构是 DDQN；根据 `manifest.json` 的 `training_args.q_network` 和检查点 `q_network` 确认实际架构。

## 预期输出与注意事项

- `manifest.json` 记录 `q_network=set_ddqn`，checkpoint 记录 `q_network=set_ddqn, network_architecture_version=2`。
- `best.pt` 由冻结后的 216 次 CPU 验证选择；所有模型架构可以复用现有评估脚本的 `from_checkpoint()`。
- 旧 `ddqn` / `d3qn` 检查点维持架构版本 1，旧 DDQN 无架构字段时仍按 DDQN 读取。
- `--resume` 必须使用同一个 `set_ddqn` 实验目录和一致的训练参数，不能复用旧模型的 Optimizer、回放或完整状态。
- 当前实验评价的是泊位—岸桥动态调度的加权等待；此 RL 环境尚未包含岸电容量或碳排放目标。
- 目前仅有模型级测试结果；代码尚未在该沙盒中运行真实港口 8+8 Worker 完整训练。服务器端集成测试和正式性能比较仍需执行。

## 建议比较

保留原三个 Seed42 结果：DDQN-96、DDQN-130 和 D3QN-96。新增加 Set-aware DDQN-96，比较：所有12个验证检查点、B=2/5/10、Shock Case9、平均改约次数、相对Periodic的尾部后悔值；随后至少3个训练随机种子。Set-aware 模型的20,865个参数大于DDQN-96，后续要加参数量匹配对照才能归因于集合编码。
