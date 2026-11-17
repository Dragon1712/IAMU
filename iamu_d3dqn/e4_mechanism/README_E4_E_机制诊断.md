# E4-E：Episode 200–400 的 Keep 价值塌缩、TD-error 与 Replay 机制消融

> **交付的是实验代码，不是服务器运行结果。** 只新增 `iamu_d3dqn/e4_mechanism/`，并在运行时由生成器另外生成 `train_budgeted_ddqn_e5.py`；**不改写**原始 `train_budgeted_ddqn_parallel.py`、原 V0.9/E4 模型和旧结果。
>
> 需要在用户服务器的已安装 IAMU (`iamu_d3dqn/`) 和 PyTorch 环境中执行。当前交付环境没有 E4 完整 `training_state.pt`，所以不可声称真实训练已经跑完。

## 一、为什么做这个实验

E4-D 在 B900/Seed43 同一训练轨迹中发现，Episode 100 → 400 加权等待从 1929.17 增加到 1972.87，合法 Edit 时选择 Keep 的比例 58.2% → 98.6%。EP100 对 EP900 的固定状态 Q-gap 下降约 0.491，260 个 Edit 候选中有 150 次从超过 margin 变为不过 margin。固定反事实在一些位置显示本应改约却持续 Keep。

但这些数据只说明**相对价值出现漂移**，不能确定原因是 Bellman 自举更新、Target 同步、Replay 中可行动作样本稀缺、在线采样分布变化，或者几者叠加。

## 二、两个独立实验

### E4-E1：新的受监控在线训练，Episode 200–400

沿用原 E4 B900/Seed43 的算法、学习率、预算/到港数据、探索率衰减和 actor/validation 参数，只把本机制实验限定为 400 Episode（减少额外无关训练），验证间隔改为 50 Episode，以 EP200 作为分叉点。**这条新轨迹不是历史 B900 的逐位复现**：并行 actor 完成顺序并非确定性。

每 **25 Episode** 在统一、冻结的公开状态银行上记录：

- **Q-gap**：`Q(s,a)-Q(s,Keep)`；平均最佳候选 gap、所有合法候选平均 gap、选择 Keep 比例、超过 Keep margin(0.1)的比例、按 B=2/5/10 分层 gap。
- **TD-error**：对同一批 EP200 保存的 Replay 转移，以及对各阶段当前 Replay 样本，记录 `delta = r + gamma Q_target(s', argmax Q_online(s',·)) - Q_online(s,a)`。`argmax` 与训练代码一致：Keep 加 margin，所有非法动作 mask 掉。
- **Replay 动作组成**：`forced_keep`（没有合法 Edit）/ `choice_keep`（有合法 Edit 但选择 Keep）/ `edit` 分开计数；统计预算层、旧的20%与新的20%经验比例、上25个 Episode 的演员样本构成和 policy-staleness。
- **固定公共状态 Q 值**：不使用未来真实 ETA 或事后回放数据选择状态；Keep/Greedy 只是产生公开状态的锚定轨迹。

**关键数据保存：** 在完成 `val_000200` 验证，并确认完整状态已写入 `training_state.pt` 后，原子复制到 `e5_mechanism/full_state_ep0200.pt`。这份文件包括在线网络、目标网络、Adam、Replay 全内容及抽样 RNG、训练日志和随机数状态。附 SHA256 与 EP200 验证快照的一致性记录。

### E4-E2：EP200 同一个训练状态的冻结 Replay 消融

均加载**同一份**完整 EP200 分叉状态；**禁止**从仅有网络权重的 `validation_snapshots/val_000200*.pt` 加载后“假装”恢复 Replay/Adam。

| 组别 | Replay 数据 | Minibatch | Target 同步 | 主要作用 |
|---|---|---|---|---|
| `online` | 继续新增真实训练转移 | 原始均匀 Replay | 正常 | 原在线对照 |
| `uniform` | EP200 数据完全冻结 | 原始均匀 Replay | 正常 | 检验固定数据下，优化+自举是否足以出现漂移 |
| `fixed_target` | 同一份 EP200 数据 | 原始均匀 Replay | **EP200 目标网络固定** | 隔离目标网络更新影响 |
| `actionable_uniform` | 同一份 EP200 数据 | 在有合法 Edit 的经验中原比例均匀抽样 | 正常 | 单独检验去除被迫 Keep 的影响 |
| `balanced` | 同一份 EP200 数据 | **可选 Keep / Edit 1:1** | 正常 | 检验动作采样比例的干预效应 |

在线记录中的 `learner_updates` 按**Episode 实际完成顺序**提取。离线的 201–400 Episode 每一阶段做完全相同的梯度更新次数、使用相同原始 batch_size、从同样的 optimizer 和 target 状态开始。注意`balanced`可能有放回重复采样 Edit，因此必须报告每批不同样本数；它检验的是**采样分布干预**而非简单纠正偏差。

四种离线分支只改变上述指定因素，Replay 内容 SHA256 必须从头到尾不变，且所有输出独立目录。冻结数据造成的离线价值退化也有分布外外推风险，**不能仅凭一次结果断言训练器存在 Bug 或唯一根因**。

## 三、数学依据

### (1) 可行动作 Keep 率——不能只看总 Keep 比例

\[
P(\mathrm{Keep}\mid \exists\,\mathrm{legal\ Edit})
=\frac{N_{\mathrm{choiceKeep}}}
{N_{\mathrm{choiceKeep}}+N_{\mathrm{Edit}}}.
\]

初始状态、预算耗尽或候选不可用时的 `forced_keep` 必须排除。否则 Replay 中的大量终端/无候选 Keep 会造成**伪策略塌缩**。

### (2) 与训练一致的 Double-DQN 残差

\[
\delta_t=r_t+\gamma(1-d_t)Q_{\theta^-}\left(s_{t+1},
\arg\max_{a\in\mathcal A_{t+1}}[Q_{\theta}(s_{t+1},a)
 +m\mathbf1(a=\mathrm{Keep})]\right)-Q_\theta(s_t,a_t).
\]

其中 `m=0.1`。分别统计 `choice_keep / forced_keep / edit` 的 signed TD、MAE、RMSE、P90。这是**相对当前网络/目标网络的训练残差**，不能直接等同于相对于真实最优回报的误差。冻结 EP200 TD probe 则可减少“每次测量的数据都改变”带来的混淆。

### (3) 受控分叉的解释边界

若 `uniform` 仍出现 EP200→400 的 Q-gap 下移，则固定经验上的梯度优化/自举本身足以产生类似漂移；若 `fixed_target` 明显缓解，目标同步是强候选；若 `balanced` 缓解，则动作样本失衡值得继续检验。若只有在线分支下移，说明**新数据流或在线 actor 决策分布变化**更值得优先排查。

所有条件需要跨至少 Seed 42、43、44 重复；Case 8/9 仍是用于分析和选模的验证集，不能作为独立泛化测试。

## 四、安装

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_E4_E_TDReplay_FrozenAblation.zip
cd iamu_d3dqn
conda activate erldde
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest e4_mechanism.test_e5_mechanism -v
```

### 1. 从先前 E4 的 manifest 生成公开状态银行（一次即可）

```bash
python e4_mechanism/build_probe_bank.py \
  --source-manifest results/v10_e4_b900_seed43/manifest.json \
  --seeds 400001,400002 --anchors keep,greedy \
  --states-per-group 2 \
  --out results/v10_e5_probe/public_bank.npz
```

`public_bank.npz` 只含合法候选特征/动作 mask/公开状态编码，`.metadata.json` 记录 SHA、条件和状态覆盖信息，不保存潜在未来信息，也不把真实未来作为训练目标。其内容对所有在线/离线分支保持固定。

### 2. 生成独立的受监控训练脚本（不会覆盖原脚本）

```bash
python e4_mechanism/build_instrumented_trainer.py \
  --trainer train_budgeted_ddqn_parallel.py \
  --out train_budgeted_ddqn_e5.py
```

生成器要求原训练器源码的已知训练/保存锚点**精确匹配**，若升级了原训练器而不兼容会拒绝修改，不会默默插错位置。`train_budgeted_ddqn_parallel.py` 本身不变。

### 3. 在线训练：EP1–400，EP200 归档分叉状态

单训练种子 Seed43，GPU0、4 actor/4 validator：

```bash
python -u e4_mechanism/launch_online.py \
  --seeds 43 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --actors 4 --validators 4 \
  --episodes 400 --fork 200 \
  --validation-every 50 --monitor-every 25 \
  --probe-bank results/v10_e5_probe/public_bank.npz \
  --prefix v10_e5_online \
  2>&1 | tee v10_e5_online.log
```

使用 `--dry-run` 只打印命令。若中断，参数完全一致的情况下使用 `--resume`；不能更换 `--episodes/--monitor-every/--eval-every/--probe-bank` 或训练代码后强行恢复。同一个输出目录中如果已存在归档，程序校验并保留，绝不自动覆盖。

训练结束需检查：

```bash
ls -lh results/v10_e5_online_seed43/e5_mechanism/
# full_state_ep0200.pt / full_state_ep0200.metadata.json
# td_probe_ep0200.pt
# qgap_trace.csv, replay_trace.csv, td_trace.csv
# validation_snapshots/val_000200_* still in online run root
```

### 4. 离线消融：四 GPU 并行

```bash
bash e4_mechanism/run_offline_four_gpu.sh
```

该脚本默认：

- `actionable_uniform` → cuda:0，输出 `results/v10_e5_offline_actionable_uniform_seed43`
- `uniform` → cuda:1，输出 `results/v10_e5_offline_uniform_seed43`
- `fixed_target` → cuda:2，输出 `results/v10_e5_offline_fixed_target_seed43`
- `balanced` → cuda:3，输出 `results/v10_e5_offline_balanced_seed43`

也可单独运行一个分支：

```bash
python -u e4_mechanism/offline_branch.py \
 --fork results/v10_e5_online_seed43/e5_mechanism/full_state_ep0200.pt \
 --online-trace results/v10_e5_online_seed43/train_curve.csv \
 --probe-bank results/v10_e5_probe/public_bank.npz \
 --mode uniform --device cuda:1 --until 400 --every 25 \
 --trust-local-training-state \
 --out results/v10_e5_offline_uniform_seed43
```

**安全性**：原训练完整恢复文件包含 Python `Transition` / NumPy 对象，必须使用 PyTorch 的 pickle 反序列化。`--trust-local-training-state` 仅允许加载本地可信训练器产生的完整文件。不要对陌生来源、下载来的 `.pt` 使用此开关。

### 5. 同条件闭环验证：原 Case8/9、ETA Seeds、B=2/5/10

```bash
python -u e4_mechanism/evaluate_snapshots.py \
  --online results/v10_e5_online_seed43 \
  --offline results/v10_e5_offline_uniform_seed43 \
            results/v10_e5_offline_fixed_target_seed43 \
            results/v10_e5_offline_actionable_uniform_seed43 \
            results/v10_e5_offline_balanced_seed43 \
  --episodes 200,300,400 \
  --workers 8 \
  --out results/v10_e5_validation_seed43
```

评价会用原在线验证 `validation_details/val_.../runs.csv` 强制做 **online parity**：相同 Case、Shock/Repeated/Small、ETA seed 和 B 下等待、预算必须与在线原始记录吻合，否则实验中止。离线模型采用完全相同的物理环境和仿真未来。输出 `runs.csv/summary.csv`、模型/检查点 SHA 和完整性标记。

### 6. 自动汇总

```bash
python e4_mechanism/analyze_mechanism.py \
  --online results/v10_e5_online_seed43 \
  --offline results/v10_e5_offline_uniform_seed43 \
            results/v10_e5_offline_fixed_target_seed43 \
            results/v10_e5_offline_actionable_uniform_seed43 \
            results/v10_e5_offline_balanced_seed43 \
  --eval results/v10_e5_validation_seed43 \
  --out results/v10_e5_mechanism_analysis_seed43
```

阅读 `results/v10_e5_mechanism_analysis_seed43/report.md`。

## 五、在 4 GPU 服务器上的资源用法

**先做 Seed 43 一次因果分叉**：在线训练使用 GPU0，4 个 CPU actor、4 个验证 worker；完成后 4 个离线分支同时使用 GPU0/1/2/3，离线无需 actor，也不应占用额外 16 个 CPU 采样进程。闭环评价独立使用 `--workers 8` CPU。

跨种子时，在线任务可指定 `--seeds 42,43,44 --devices cuda:0,cuda:1,cuda:2,cuda:3 --actors 4 --validators 4` 并行运行（共 12 个 actor + 12 个验证 worker）；三个种子**不能混用同一个 EP200 fork 文件**。待每个种子的在线参考曲线存在后，逐种子运行上述四种离线分支。

## 六、错误解释防护

- **硬约束**：所有记录必须保留动作 mask、有限预算和 Keep margin。Replay 中被迫 Keep 不等于价值塌缩。
- **不额外训练**：监控时只 `no_grad` 计算 Q / TD，不修改 optimizer、target 或梯度更新次数，不消费训练 Replay RNG。在线监控仍可能增加时间，进而影响 actor 完成顺序，因此不追求与原 E4 历史运行的逐位一致。
- **同状态 Q-gap**：固定 bank 必须先于离线分叉建立，并按哈希锁定。相同状态与动作下才可以比较早期/后期 Q-gap。
- **静态 TD 数据**：EP200 TD probe 与动态 Replay TD probe 分别统计；不能把动态数据分布变化误当成模型误差漂移。
- **对照归因**：`uniform` 与 `fixed_target` 的差异是 target-update 干预的候选证据；`balanced` 的改善可能来自有效样本权重、采样重复或其他非线性优化作用，不能称为已证实根因。
- **独立测试**：Case10–14 只在模型/诊断规则冻结后测试；不要在这组实验期间用于迭代调参。
