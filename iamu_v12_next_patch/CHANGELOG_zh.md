# IAMU V1.2 机制诊断（P0–P4）变更

基于提交 70606c7b；仅增量修改 V1.2 周边，不动 V1.1 Transition、环境、候选物理规则和旧结果目录。

## 已实现

- `replay.py`：成员过滤由 O(n) 列表查找替换为 set 检查；保留索引顺序、RNG、采样及 Replay 统计口径。
- `validation.py`：原接口不变；默认旧验证路径；显式 cache_dir/环境变量开启后复用 Keep/Greedy/Periodic，DDQN 始终重新评价，写 `baseline_cache_metrics.json`。
- `agent.py`：默认不开启的 `diagnostic_callback`，基于原更新 chosen/y/TD/梯度范数，不添加反向传播。
- `fixed_data.py`：新增 `--probe-bank`, `--diagnostics`, `--checkpoint-every`, `--probe-every`, `--baseline-cache-dir`；无probe时完全保留原固定数据训练逻辑。
- `baseline_cache.py`：基于案例与代码内容的强指纹，Linux flock 并发串行，原子写入，读时内容校验。
- `probe_bank.py`：独立场景种子的共同可行动状态分层，行动前缀重放，真实候选方案指纹。
- `probe_eval.py`：批量跨checkpoint的每合法Edit Q-gap，对保存的模型保持只读，不声称历史Target/TD可恢复。
- `counterfactual_probe.py`：同一真实未来，固定续演政策，独立深复制分支，存路径条件收益标签。
- `diagnostics.py`：25步抽样、250步窗口聚合；源状态四类别、宏长度五分箱及分位数。
- `fixed_f1.py`：A0–A3同 train_raw/probe_raw 分组1000步，从相同参数初始化；源、代码、任务信息、模型快照及明确验证状态。
- `analyze_diagnostics.py`：条件路径损失、错误Keep/有害Edit、按场景与预算分层汇总。
- `tests_v12/test_diagnostic_protocol_next.py` 与 `test_probe_restore_next.py`：11项确定性逻辑/接口测试，需服务器真实环境补测。
- `apply_next.py`：离线 Git blob SHA 核验、拒绝覆盖、源码备份；适用于没有 `.git` 的 Linux 目录。

## 尚未执行

- 服务器真实环境下8状态restore/counterfactual smoke、完整144状态probe构建、F1 4×seed42×1000更新、缓存与原验证在真实Case上的逐行等价、服务器实际加速、CPU确定性训练比对、GPU性能基准。
- F2条件对照、Seed43/44、第三独立训练数据与在线400 episode均属于后续阶段。
