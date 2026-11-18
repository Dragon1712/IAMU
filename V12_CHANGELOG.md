# V1.2 新增代码清单与验收状态

**目标 GitHub commit**：`2d96981a38434014f14e31be5641d422d28835f8`。原始 V1.1 文件没有被修改。

## 新增

- `iamu_d3dqn/v12_protocol/transitions.py`: 原 Transition 外包元数据、新 MacroTransition、前缀/压缩结果。
- `compress.py`: 三表示锚点、逐原始事件折扣与观测/预算链审计。
- `targets.py`,`agent.py`: per-transition bootstrap discount 的 DDQN、Set-aware 原网络、V1.2 专用模型 checkpoint。
- `replay.py`: 最新200完整 episode 窗口、精确索引复现、Uniform/A1-only分层。
- `actors.py`: 保持所有强制 Keep 的原 act() 调用，收集显式预算和事件元数据。
- `train.py`: V1.1 固定轮次和任务随机数、有序处理、原始步数更新额度、warmup/验证、恢复及完整配置 hash。
- `validation.py`: 明确加载 V1.2 模型的 Case8/9 Keep/Greedy/Periodic/DDQN 强基线。
- `fixed_data.py`: 同轨迹四组离线梯度预算机制实验。
- `analyze.py`: 配对指标、CVaR90 正后悔、分预算/扰动。
- `scripts/run_v12_smoke.sh`: 4组×8episode工程 smoke。
- `scripts/run_v12_main_matrix.sh`: **仅手动**触发的12组长训练脚本。
- `iamu_d3dqn/tests_v12/`: 纯逻辑、Agent、Replay恢复、真实环境测试。

## 保持未变

所有 V1.1 环境/网络/候选代码和历史实验结果；不得从 V1.1 best.pt 恢复 V1.2 optimizer/replay。

## 已发现的限制

1. 当前工具环境不能从 GitHub 下载整个代码树到本地，故**真实 V1.1 端到端集成未执行**。请运行真实环境 smoke，确认兼容后再启动400episode主实验。
2. Python/Torch 完整恢复状态使用本地 trusted pickle，禁止从不信任来源载入。
3. 目前严格恢复仅支持未扩展的 task table 与 episode 数；显式 fork 扩展工具未开发，禁止偷换。
4. Replay 的内存指标是 NumPy 数组的已测字节数，不含 Python 对象/allocator 开销。实际进程 RSS 需由服务器补测。
5. CPU连续/中断完全一致性、跨worker一致性和GPU确定性需要服务器单独验证。
6. 固定数据用预先声明的 Keep/Greedy/随机合法动作混合行为；并不是无偏或随机目标策略样本，不能替代在线评估。
