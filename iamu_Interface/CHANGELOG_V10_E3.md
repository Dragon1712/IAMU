# IAMU V1.0-E3 增量补丁说明

## 修改范围

仅新增 `iamu_Interface/` 文件；不修改已有 `iamu_dynamic/v10_policy.py`、`v10_budget_lookahead.py`、`experiments/eval_risk_budget_v10.py`、训练代码、模型或历史结果。

## 新增功能

- `iamu_dynamic/v10_e3_confidence.py`：配对 MPC 收益均值/样本标准误/近似 Student-t 置信区间、方向一致性门控、固定/自适应/纯 MPC 消融评分。
- `iamu_dynamic/v10_e3_policy.py`：在原 `IAMUEventEnv` 和 E2 的 `evaluate_multistage` 上使用同一组风险样本与 MPC 样本执行 E3 策略，输出逐候选统计。
- `experiments/eval_e3_value_calibration_v10.py`：冻结公开状态，比较在线低 M MPC、独立高 M MPC、合成未来 ETA 更新下的共享后续策略反事实，以及原测试场景下的事后反事实；H=0/H=2 消融；在**同一估计数据与同一场景**下评价八种融合策略。
- `experiments/eval_e3_fusion_v10.py`：8 种策略在相同 ETA 条件下闭环配对；自动比对冻结 V0.9 基线；记录改约、运行时间和门控使用情况。
- `experiments/analyze_e3_v10.py`：分种子、预算、扰动汇总，以及保留训练种子和 Case×ETA 相关性的探索性分层 Bootstrap。
- `scripts/run_e3_parallel.py`：多进程 CPU 分片运行、严格不同目录、恢复配置校验、空候选分片记录及全部成功后的合并。
- `scripts/check_e3_e2_parity.py`：确保 E3 q_fixed/ q_only 在相同配置下精确复现 E2 的 q_joint / set_ddqn。
- `scripts/run_e3_smoke.sh` / `run_e3_calibration_validation.sh` / `run_e3_fusion_validation.sh`：常用运行入口。
- `tests/test_v10_e3_confidence.py` / `test_v10_e3_real_env.py`：单元与服务器端物理环境集成测试。
- `README_E3_置信度预算前瞻与价值校准.md`：完整协议、公式、命令及科研解释边界。

## 不作为结论的事项

- 13 项纯单元测试通过，不等于服务器真实仿真已经完成。
- 未来场景的反事实收益是**已指定信号模型下的模拟标签**；模型预测误差和外部真实 ETA 分布偏差不能混为一谈。
- 置信门控在重尾样本下的近似区间不保证 95% 真实覆盖。
- 旧验证集 Case8/9 用于机制探索，不得据此证明独立泛化或已发表创新性。
