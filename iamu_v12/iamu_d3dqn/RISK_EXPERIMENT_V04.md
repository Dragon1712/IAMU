# IAMU Dynamic V0.4 风险感知改约选择器

## 实现内容

- `deterministic`：继承 V0.3 的点预测 ETA 贪心，不改变已有默认行为。
- `expected`：对每次 ETA 公告时当前可观测的预测信息进行蒙特卡洛采样，比较候选计划对样本的平均改善量。
- `risk_lookahead`：计算期望改善、下侧改善 CVaR（负收益尾部损失），并用**一阶段未来修复**估计保留预算的期权价值。动作集包含 Keep，未来假设公开信号为模型生成；不使用真实未来 ETA。
- 三种方法共享候选生成、硬预算约束、原始动态执行器与场景/种子。可选 `alns2` 两船候选。
- 自动输出 `runs.csv`、`paired_runs.csv`、`summary.csv`：平均值、中位数、退化率、最差场景、运行时间、尾部遗憾 CVaR。

## 科学边界

1. 随机 ETA 采样是**未校准的条件高斯代理模型**，不能声称真实港口概率分布。
2. `risk_lookahead` 只规划一个假想的未来决策时刻，是两阶段 recourse 近似，**不是完整长序列 MPC 或强化学习**。
3. 不读取实际未来到港时间、未来真实更新消息；可行性检查和真实执行仍沿用 V0.3 简化泊位—岸桥物理层。
4. 当前没有岸电约束、排放模型和冻结 ERL-DDE 权重的端到端实测。
5. 原 V0.3 `run_benchmark.py` 不改；新实验用 `run_risk_benchmark.py`，旧结果不覆盖。

## 测试

```bash
cd ~/code/iamu/IAMU_dynamic
python -m unittest discover -s tests -v
```

## 快速复现 15-run 风险诊断（可从实验目录执行）

```bash
python run_risk_benchmark.py \
 --cases ../2024LiERLDDE/case_code/1.txt \
 --scenarios shock --seeds 44 --budgets 0,2,5 \
 --methods deterministic,expected,risk_lookahead \
 --risk-aversions 0.5 --risk-samples 3 --forecast-std 8 \
 --candidate-limit 4 --lookahead-candidates 1 \
 --next-move-candidates 1 --out results/v04_case1_shock44
```

## 正式配对评估建议

建议先选择 3 个案例、每类扰动 5 个开发种子，风险抽样数 4、候选数 4，固定并留出独立测试集。耗时可能较长；先运行 1 个案例、1 个种子小规模验证，再逐步增加。

```bash
python run_risk_benchmark.py \
 --cases ../2024LiERLDDE/case_code/0.txt \
         ../2024LiERLDDE/case_code/1.txt \
         ../2024LiERLDDE/case_code/2.txt \
 --scenarios shock,repeated --seeds 11,22,33,44,55 \
 --budgets 0,2,5 --methods deterministic,expected,risk_lookahead \
 --risk-aversions 0,0.5,1 --risk-samples 4 \
 --forecast-std 8 --candidate-limit 4 \
 --lookahead-candidates 1 --next-move-candidates 1 \
 --out results/v04_risk_full
```

**风险研究应汇报**：平均总等待、30 个配对场景的最差损失与 CVaR、预算消耗、相对 Periodic 的胜/负场景数与运行时间。不要将训练或模型选择时见过的种子作为泛化测试。
