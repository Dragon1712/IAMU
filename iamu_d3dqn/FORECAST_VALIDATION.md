# IAMU V0.5：风险预测模型的独立验证（不运行调度器）

## 目的与验证对象

本工具验证 **V0.4 ETA 采样分布本身**，而不是通过等待时间、岸桥优化或改约策略来选择预测模型。

- `censored_sigmaX`：**完全对应 V0.4 已有采样形式的分布**：`max(now, Normal(current_public_eta, X))`。`censored` 在 `now` 有概率质量，并非条件截断正态。
- `truncated_sigmaX`：**研究用对照模型，并非当前 V0.4 已部署的采样器**。表示在已收到某艘船的到港前 ETA 通知、确认该船尚未到港时，正态分布条件化于 `arrival > now` 的分布。
- `sigma_train`：在训练划分的 ETA 更新残差上估计 `sqrt(mean((actual-public_eta)^2))`，只用训练标签。它会成为以上两类分布的额外候选参数。

严格要求：所有预测分布 `ArrivalBelief` 的输入只有 `current_public_eta`、`now`、`sigma` 和模型家族。真实到港时间仅在离线评分步骤中作为标签。**不调用** `simulate()`、`RealizedRuntime`、`SlotPlanner`、`true_etas` 进行在线决策，也不使用改约等待作为选择目标。

## 安装

把压缩包里的两个 `.py` 文件与本说明文件覆盖到服务器的 `IAMU_dynamic/` 对应相对路径（均为新增文件，不改原有模型和结果）。仅依赖 Python 标准库及仓库原有 `iamu_dynamic` 包。

```bash
cd ~/code/iamu/IAMU_dynamic
python -m unittest discover -s tests -v
python validate_eta_forecast.py --help
```

## 推荐：独立测试执行命令

**从 `~/code/iamu/IAMU_dynamic` 目录运行：**

```bash
python -u validate_eta_forecast.py \
  --train-cases ../2024LiERLDDE/case_code/0.txt \
  --val-cases ../2024LiERLDDE/case_code/1.txt \
  --test-cases ../2024LiERLDDE/case_code/2.txt \
  --train-seeds 101:105 \
  --val-seeds 201:205 \
  --test-seeds 1001:1020 \
  --scenarios shock,repeated,small \
  --sigmas 8,14.4,20 \
  --families censored,truncated \
  --uncertainty 8 \
  --ood-uncertainty 12 \
  --updates 2 \
  --bootstrap 500 \
  --out results/v05_forecast_validation
```

- **训练划分** Case 0 × Seeds 101–105：仅拟合额外的 σ 参数。
- **验证划分** Case 1 × Seeds 201–205：仅按更新时刻的轨迹宏平均 CRPS 选择模型，锁定模型身份。
- **独立测试** Case 2 × Seeds 1001–1020：所有调参与模型选择结束后，仅评估一次。
- **额外扰动强度** `test_ood`：仍使用 Case 2 和相同测试种子，但把扰动强度从 8 改为 12。它是独立于模型选择的强度压力测试，**不应将它与普通测试集当作相互独立的轨迹来合并做 CI**。
- `--sigmas` 是预先指定的固定候选，不能根据测试集结果重新挑选。
- `--save-predictions` 可额外保存每条合成预报记录及其离线真值，数据量较大。
- `--grid 64` 或 `128` 可用于检查分位数积分近似 CRPS 的数值稳定性，默认 32。

**安全性：** 程序同时检查三组的案例路径、案例文件内容 SHA256、随机种子集合全部互斥。只有真正不同的案例和种子才能进入独立测试。

## 输出文件

```text
results/v05_forecast_validation/
├── manifest.json                  # 参数、案例哈希、样本数和数据划分
├── selected_model.json            # 验证集锁定的模型、训练拟合 sigma
├── model_metrics.csv              # 训练/验证/测试/OOD；按场景与阶段分组
├── selected_model_metrics.csv     # 全部最优及 V0.4 可直接采用的最优模型
├── test_bootstrap_ci.csv          # 测试集 CRPS、覆盖率和超限率的轨迹 bootstrap CI
└── test_pit_bins.csv              # PIT 十等分直方图（按模型）
```

主要指标：

- **CRPS（越低越好）**：严格适当的概率评分，使用分位数积分的确定性数值近似；不受每次 Monte Carlo 4/16/32 样本数量的噪声影响。
- **NLL（越低越好）**：连续到港真值的负对数密度分数；条件截断分布考虑条件化归一化。
- **Coverage 50/80/90/95%**：预测中心区间的实测覆盖率；90% 区间理想接近 0.90。
- **high90、high95**：真实到港时间超过模型对应上分位数的发生比例；理想值分别接近 0.10/0.05。
- **excess90、excess95**：超过上分位数后的平均额外延误幅度（包含未超出时的零），体现尾部超限严重性。
- **pinball90/95**：上分位数的严格适当分位数损失。
- **PIT**：概率积分变换的直方图，用于观察分布系统性偏差；现有删失模型在 `now` 有点质量，评估这里所有更新真实到港均严格晚于 `now`，仍需结合覆盖率判断。
- **Bootstrap**：以整个 `(case,scenario,seed)` 轨迹为单位重采样，而不是把同一船反复更新的预报当作独立观测。一个未见过的 Case 2 不能代表跨港口全总体泛化，不能过度解读置信区间。

## 注意：验证目标与局限

1. **验证的是预报条件分布，不是改约动作的 CVaR。** 即使 CRPS 或预测覆盖率更好，也不保证调度等待或改约尾部风险同步改善。验证通过后需要再做配对调度试验。
2. **只评估公开 ETA 公告所对应的船舶和起始预报。** 更新公告本身表明船舶尚未到港。它不覆盖 V0.4 当前采样器对“未开始作业但实际上已经到港的其他船舶”的信念误差；那需要在环境中显式公布 `arrived/not arrived` 状态后才能无泄露地全面检验。
3. **全部是合成场景。** Shock 的初始到港分布带右偏尾部，Repeated 的更新误差较大；不会自动等同于真实港口 ETA 误差。
4. **避免测试集调参。** 输出中的所有测试模型仅作透明展示，**正式部署候选只能由验证集选出**。`selected_model.json` 中的 `best_v04_deployable_censored_model` 可作为维持 V0.4 采样形式时的最优选择；如果 `selected_model` 为 `truncated`，必须另外修改采样器才可以部署。
5. **采样数不等于预测分布质量。** `risk_samples=4/16/32` 影响候选动作的 Monte Carlo 近似，而本工具测试的是这些样本所来自的理论分布质量，两者需分别报告。
6. OOD σ 校准会发生明显变化；不能把预测标准差 14.4 恰好等于 Shock 场景真值生成器基础标准差，就直接当成当前 ETA 条件残差的校准证据。

## 本地校验（非论文结论）

在 2026-11-12 GitHub `Dragon1712/IAMU` 主分支的原始 `case_code/{0,1,2}.txt` 上，使用上方默认配置完成合成验证：

| 阶段 | 当前所选诊断模型 | CRPS | Coverage90 | upper-90 超限率 |
|---|---|---:|---:|---:|
| Validation（Case 1） | truncated σ≈7.1246 | 4.1518 | 88.67% | 11.40% |
| Independent test（Case 2） | 同一模型，未再训练 | 3.6459 | 90.99% | 10.20% |
| Test OOD（Case 2，强度 12） | 同一模型，未再训练 | 5.3271 | 82.60% | 16.50% |

注意所选 `truncated` 为诊断对照，不是现有 V0.4 可直接运行的采样器。验证集内 V0.4 可直接采用的最优 `censored` 候选为 σ≈7.1246，在测试集的 CRPS 为 3.6446、90% 覆盖率约为 91.26%。两者 OOS 性能十分接近，不能据此断言条件截断一定优于删失模型。

**后续真正的模型改进**：使用训练集建立状态相关、可能带非对称尾部的 ETA 条件预测分布，再严格按相同切分方案测试 CRPS、上分位预测与置信区间；最后将锁定模型接回改约算法做独立调度评估。

### 关键：总体覆盖率可能掩盖 Shock 场景的明显失校准

使用相同的、验证集已锁定的 `truncated σ≈7.1246`，独立测试集按扰动类型拆开后：

| 测试强度 | 扰动类型 | 90% 区间实测覆盖率 | 真值超过预报第 90 分位数的比例 |
|---|---|---:|---:|
| 8 | Shock | **76.63%** | **24.96%** |
| 8 | Repeated | 96.48% | 5.51% |
| 8 | Small | 99.85% | 0.00% |
| 12（压力测试） | Shock | **62.37%** | **35.94%** |
| 12（压力测试） | Repeated | 85.63% | 13.45% |
| 12（压力测试） | Small | 99.80% | 0.00% |

因此，合并后的测试覆盖率约 91% **并不能证明 Shock 尾部预测已经校准**。若后续要给风险前瞻使用预测模型，优先建立只利用可观测 ETA 历史、剩余到港时间和可公开的扰动信息进行条件预测的模型。不要直接把测试集的 `scenario=shock` 真值标签作为在线输入，否则会泄露合成场景类型。
