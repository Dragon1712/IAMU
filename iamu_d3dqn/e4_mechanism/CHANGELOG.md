# E4-E 增量变更

仅新增 `iamu_d3dqn/e4_mechanism/`，不会替换原 E4/E4-D/E3 代码、配置、检查点与结果。用户运行 `build_instrumented_trainer.py` 后，另新增一个独立训练入口 `iamu_d3dqn/train_budgeted_ddqn_e5.py`（如已存在内容不一致的生成脚本则报错拒绝覆盖）。

- `build_probe_bank.py`: 固定公开状态库，Same Public Observable State，对合法 Edit mask 保持一致
- `build_instrumented_trainer.py`: 原训练源码锚点严格匹配，只插入只读监控与完成验证后的完整 EP200 状态归档
- `telemetry.py`: 每 25 Episode 的 Q-gap、Replay 动作条件分布、TD 分层动态/固定 probe
- `mechanism_core.py`: 区分可选 Keep、被迫 Keep、Edit；balanced minibatch；Replay 全内容 SHA256
- `td_probe.py`: 精确训练 Double DQN TD 目标/Keep margin；不访问隐变量；不执行梯度更新
- `launch_online.py`: 复用 E4 B900 参数，在新轨迹运行 Episode 1–400，50 Episode 验证，EP200 full state 分叉
- `offline_branch.py`: 从同一 EP200 完整状态做 uniform / fixed_target / actionable_uniform / balanced 四组离线训练，匹配在线梯度更新数，验证固定 Replay 完整性
- `run_offline_four_gpu.sh`: GPU0/1/2/3 四分支并行（每分支一个 GPU）
- `evaluate_snapshots.py`: Case8/9 同条件多进程闭环验证，对在线快照执行严格逐轨迹 parity
- `analyze_mechanism.py`: 在线/离线 Q-gap / TD / Replay / 闭环评价汇总
- `test_e5_mechanism.py`: 单元测试，包含原源码 fail-closed、Replay 条件统计、分层采样、TD 公式和完整状态 SHA 防护

**边界**：旧 E4 的网络权重快照不包含原 Replay/Adam/RNG，不能用于可信离线续训；新训练必需重新运行至 Episode 200 保存完整状态。诊断代码尚需在用户服务器上实际运行。
