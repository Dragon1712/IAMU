# IAMU-RL V0.9-C：Set-aware Candidate-Conditioned Dueling Double DQN

## 1. 基线与改动边界

本版本基于 `iamu_d3dqn/` V0.9-B Set-aware DDQN（可通过上一份 `IAMU_SetAware_DDQN_v09B.patch` 从 GitHub V0.9-A 升级）。新增第四种 `--q-network set_d3qn`。保留 `ddqn`、`d3qn`、`set_ddqn` 三种网络，且保留 IAMU 已有环境、合法动作集合、奖励、候选过滤、Replay Buffer、Double DQN Online/Target 更新、Adam、Huber loss、梯度裁剪、Keep margin、8+8 CPU 分阶段验证以及 `--resume` 语义。

**目前没有引入 CVaR、预报概率预测或改变实际港口仿真。** 用户必须从头训练新网络；不能从其他架构直接 `--resume`。

## 2. 结构

- `g`: 当前可观测全局状态，24 维；`actions`: `[B,N,24]`，`mask`: `[B,N]` Bool。
- Keep 在 `actions[:,0]`，始终合法；合法 Edit 集合最多四个。
- `h=ReLU(Linear(g, 96))`。
- `e_i=phi(x_i)`, `phi:24→64→32`，只编码合法 Edit，禁止 Keep / 非法候选参与池化。
- `z = sum_{i>0} mask_i * phi(x_i) / max(1,sum_{i>0} mask_i)`；没有合法 Edit 时 `z=0`。
- `c=[h,z]`, 128 维；`V(c): 128→96→1`; `A(c,x_i): (128+24)→96→1`。
- `Q_i=V(c)+A(c,x_i)-mean_{j:mask_j}(A(c,x_j))`。**Dueling 的 Advantage Mean 包含 Keep**，与仅 Edit 集合编码是两个不同的操作。
- 参数量 `H=96` 为 **33,346**，大于 DDQN-96 (14,113)、DDQN-130 (23,531)、D3QN-96 (23,522)、Set-aware DDQN-96 (20,865)。正式论文应增加容量匹配或解耦消融。

**重要**：在 Q 的计算之前，通过 `torch.where(mask, x, 0)` 清除非法动作填充，避免非有限非法填充值污染 Advantage 均值；集合编码器也进行相同防御。计算 Q 时无效槽仍输出一个占位分数，正式动作选择会把它们 Mask 到 `-inf`。

`forward_components(g,actions,mask)` 返回 `(value, raw_advantage, q_values)`；这与原 `CandidateDuelingQNet` 的接口相同。若要单独检查集合向量，可调用 `net.set_encoder(actions, mask)`，不应将不同网络的绝对 Q 值直接视为可比较的实际等待。

## 3. 安装选择

**A. 已经应用了 V0.9-B Set-aware DDQN 补丁**：推荐使用增量补丁：

```bash
cd ~/code/iamu
git status --short
# 建议先新建独立实验分支/工作树，避免影响正在运行的实验
git switch -c experiment/v09c-set-aware-d3qn
git apply --check /上传路径/IAMU_SetAware_D3QN_v09C_incremental.patch
git apply /上传路径/IAMU_SetAware_D3QN_v09C_incremental.patch
```

**B. GitHub 仍是 V0.9-A（没有安装 V0.9-B）**：使用 `IAMU_SetAware_D3QN_v09C_from_v09A.patch` 单次安装；**不要**与之前的 V0.9-B 补丁叠加应用。组合补丁同时包含 set_ddqn 和 set_d3qn。

## 4. 测试

```bash
cd ~/code/iamu/iamu_d3dqn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest tests.test_set_aware_d3qn_v09c -v
python -m unittest tests.test_set_aware_ddqn_v09b tests.test_cc_d3qn_v09 -v
# 真实 IAMU 小案例 4+4 Worker、冻结权重、216 次验证和断点恢复：
python -m unittest tests.test_set_aware_parallel_v09c -v
```

最后一项须在含有真实 `iamu_dynamic` 模块及原模拟器的服务器上执行；我们提供测试，但当前交付环境未执行该原环境的多进程集成测试。若服务器 Python/PyTorch 与本地版本不同，请在服务器重跑测试。

## 5. 300 回合实验（与前四种架构对照）

从 `~/code/iamu/iamu_d3dqn` 运行。GPU 的 `CUDA_VISIBLE_DEVICES=2` 对应进程内 `--device cuda:0`：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python -u train_budgeted_ddqn_parallel.py \
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt \
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
  --q-network set_d3qn --hidden-dim 96 \
  --episodes 300 --workers 8 --validation-workers 8 \
  --prefetch-per-worker 1 --sync-every 1 \
  --scenarios shock,repeated,small --budgets 2,5,10 \
  --uncertainties 4,8,12 --validation-uncertainty 8 --updates 2 \
  --val-seeds 400001:400003 --clock union --period 12 \
  --candidate-limit 4 --candidate-screen positive --min-gain 0.1 \
  --budget-features absolute --value-features global \
  --reward-scale 100 --keep-margin 0.1 --greedy-exploration-prob 0.75 \
  --lr 0.0003 --gamma 1.0 --target-update 250 \
  --replay-capacity 50000 --batch-size 64 --warmup 512 \
  --updates-per-step 0.25 --max-updates-per-episode 64 \
  --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-episodes 300 \
  --eval-every 25 --validation-progress-every 6 --state-every 5 \
  --seed 42 --device cuda:0 \
  --out results/v09c_set_d3qn_seed42_300 \
  2>&1 | tee v09c_set_d3qn_seed42_300.log
```

每次验证 `2 cases × 3 scenarios × 3 seeds × 3 budgets × 4 methods = 216`，300 回合 12 次验证共 2592。验证 CSV 中网络策略的历史标签仍为 `method=ddqn`，请使用 `manifest.json → training_args.q_network` 区分架构。

### 实验建议

先在 Seed 42 对照 DDQN-96、DDQN-130、D3QN-96、Set-aware DDQN-96 与 Set-aware D3QN-96 的验证曲线、Case 9 Shock/B=2、B=5 的尾部损失。**单种子不足以得出显著性结论**；补充 Seed 43/44 或以上。下一轮可进一步比较集合信息只进入 Value 与同时进入 Advantage 两种结构，控制参数量。

## 6. 兼容性与限制

- 保存的推理检查点包含 `q_network='set_d3qn'` 与 `network_architecture_version=2`。V0.8 旧 DDQN 缺架构字段时仍可默认加载 `ddqn`。其他网络也按照对应版本加载。
- 恢复状态 `training_state.pt` 包含网络权重/优化器/Replay Buffer。恢复配置摘要会区分四个架构。仅从可信本地恢复状态调用 `torch.load(weights_only=False)`。
- 训练日志和图表的 Q 校准诊断须采用已冻结的权重；完整训练结果不由此代码交付预先产生。
- 当前动态规划/实际仿真研究的是**有限改约下泊位—岸桥方案的加权等待时间**，没有自动实现岸电容量约束或碳排放奖励。
