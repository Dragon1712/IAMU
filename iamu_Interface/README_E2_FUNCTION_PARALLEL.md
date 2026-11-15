# IAMU V1.0-E2：按功能并行实验（不改变科学实验结果）

## 0. 交付定位

本包**只新增** `iamu_Interface/scripts/` 中的三个脚本和一个纯标准库测试文件，不修改底层仿真、Set-aware DDQN、风险采样或多阶段预算前瞻逻辑。

前置条件：**先安装上一个 `iamu_Interface_V10_E2.zip`**，因为当前 GitHub 主分支可能仍然是 E1。调度器会检查 `q_joint_heuristic` 与 `--mpc-samples` 等 E2 专有实现；未升级的工程会直接拒绝启动。

并行粒度：**Case × 扰动类型 × ETA 种子 × 改约预算 × 实验功能**。每一个独立分片启动一个独立 Python 进程，内部保留原有事件顺序、风险采样器、Keep/Edit 规则和预算扣减。不同分片没有共享运行中的 `IAMUEventEnv`、随机数生成器或 CSV 输出。

**保证范围：** 当源码、检查点、案例、超参数和随机种子一致，并且仿真/推理均确定性时，串行与并行应得到相同的加权等待和动作；程序内置逐分片核验和串并行数值对比。因 CPU 调度和内存争用，并行程序的**耗时指标不能直接与串行耗时作算法性能比较**。

## 1. 功能分工和推荐并发

| 功能 | `--workflows` 名称 | 内部操作 | 推荐 worker 数（资源宽裕时） |
|---|---|---|---:|
| E2 联合预算实验 | `multistage` | 原始 Q、风险 Q、H 阶段 MPC/Q、旧启发式等 | 8 |
| 风险抽样稳定性 | `risk_stability` | M=16/64/128，独立 N=256 参考，3 批随机数 | 12 |
| 联合策略风险稳定性 | `risk_stability_joint` | 固定 MPC 分量，只测试风险抽样变化 | 8 |
| MPC 抽样敏感性 | `mpc_samples` | M_mpc=4/8/16，其他条件不变 | 6 |
| 前瞻深度消融 | `h_depth` | H=0/1/2/3，其他条件不变 | 6 |

配置示例合计 40 个 CPU worker。每个 worker 默认限制为一个 CPU 核、一个 OpenMP/BLAS 线程；需要至少 40 个实际可使用的逻辑 CPU。

即使 GPU 资源丰富，**第一轮建议全部用 CPU**：便于隔离工作流，也减少 CUDA 设备竞争、不同内核归约顺序等因素干扰。确认数值完全一致后，才考虑把单独的网络推理工作流迁移至 GPU。

## 2. 安装和测试

在已有 `iamu_Interface/` 上叠加新的小补丁。ZIP 包内自带顶层 `iamu_Interface/` 文件夹，避免解压嵌套。

```bash
cd ~/code/iamu
unzip -o /path/to/iamu_Interface_E2_parallel.zip
cd iamu_Interface
conda activate erldde

# 原始 E2 模块测试和真实环境集成检查（先完成）
python -m unittest discover -s tests -p 'test_v10_e2*.py' -v
# 本次新增调度器、合并器与结果比较器的独立测试
python -m unittest discover -s tests -p 'test_parallel_e2.py' -v
```

本交付已通过 6 个纯标准库模块测试，并使用**模拟的测试进程**执行了完整的并行启动、CSV 聚合和 `--resume` 恢复验证。该模拟验证**不等于真实 IAMU 物理仿真验证**；实际 E2 端到端结果需要在服务器运行。

## 3. 预览分片计划（不执行仿真、不写入文件）

```bash
python scripts/run_e2_parallel.py \
  --workflows multistage,risk_stability \
  --workers multistage=4,risk_stability=6 \
  --cases 8,9 --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --dry-run
```

默认两条功能线共 **108 个分片**：每条工作线 2 Cases × 3 扰动 × 3 ETA Seeds × 3 Budgets = 54。

当使用全部五类功能时，配合 M_mpc=4/8/16 与 H=0/1/2/3，共 **540 个分片**。建议先完成前两条主功能，后面三条属于更重的敏感性工作流。

## 4. 第一轮：小规模串并行结果对比

先运行**严格一致**的单分片串行版本：

```bash
python -u experiments/eval_risk_budget_v10.py \
  --cases assets/cases/8.txt \
  --scenarios shock --seeds 400001 --budgets 2 \
  --policies set_ddqn,q_risk,q_budget,q_joint,q_budget_heuristic,q_joint_heuristic \
  --risk-samples 16 --mpc-samples 8 --lookahead-stages 2 \
  --lookahead-step-hours 12 \
  --beta-risk 0.25 --beta-budget 0.25 --legacy-beta-budget 0.10 \
  --risk-alpha 0.9 --sigma 7.12463033710398 --belief censored \
  --verify-baseline --audit --device cpu \
  --out results/e2_check_serial
```

再使用同一个条件运行**并行调度器**（这里 1 worker，用于做数值基准）：

```bash
python scripts/run_e2_parallel.py \
  --workflows multistage --workers multistage=1 \
  --cases 8 --scenarios shock --eta-seeds 400001 --budgets 2 \
  --out results/e2_check_parallel
```

对比：

```bash
python scripts/compare_e2_parallel.py \
  --serial results/e2_check_serial/runs.csv \
  --parallel results/e2_check_parallel/merged/multistage/default/runs.csv
```

程序会按 `(case, scenario, eta_seed, budget, policy)` 匹配每一条记录，比较等待、消耗预算、改约次数、配对收益、决策数量等字段，**只忽略运行时间**。默认绝对公差 `1e-7`、相对公差 `1e-9`。未通过时不要继续扩大规模。

如果本地尚未复制案例，请先 `python setup_interface.py --source ../iamu_d3dqn --copy-assets`，或者将串行命令中的 `assets/cases/8.txt` 换成服务器上真实 Case 8 文件绝对路径。

## 5. 先开展两类核心功能并行

示例：利用 20 个不与其他训练任务竞争的 CPU 核心。注意 `--cpus` 必须按照服务器实际可使用、实际空闲的 CPU 核编号调整，**不要照抄 32-51 这一示例而不检查硬件**。

```bash
# 可用 CPU 列表及已有任务情况
lscpu -e=CPU,SOCKET,CORE,NODE,ONLINE | head -50
ps -eo pid,psr,%cpu,args | grep -E 'train_budgeted|eval_risk|run_e2' | grep -v grep

python -u scripts/run_e2_parallel.py \
  --workflows multistage,risk_stability \
  --workers multistage=8,risk_stability=12 \
  --cpus 32-51 \
  --cases 8,9 --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --risk-samples 16 --mpc-samples 8 --h 2 \
  --beta-risk 0.25 --beta-budget 0.25 \
  --risk-sizes 16,64,128 --reference-samples 256 \
  --repeats 3 --states-per-group 2 --stride 10 \
  --out results/v10_e2_parallel_primary \
  2>&1 | tee e2_parallel_primary.log
```

该命令两个工作流同时运行；每个实际工作进程独占分配的 CPU 亲和性列表，并行数不能超过 `--cpus` 提供的核数。如果只想使用当前可用 CPU，可**省略** `--cpus`。请根据 RAM 负载逐步增大 worker 数。

## 6. 后续独立的稳定性与前瞻敏感性功能

```bash
python -u scripts/run_e2_parallel.py \
  --workflows risk_stability_joint,mpc_samples,h_depth \
  --workers risk_stability_joint=8,mpc_samples=8,h_depth=8 \
  --cases 8,9 --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --risk-samples 16 --mpc-samples 8 \
  --risk-sizes 16,64,128 --reference-samples 256 \
  --mpc-sizes 4,8,16 --h-sizes 0,1,2,3 \
  --out results/v10_e2_parallel_sensitivity \
  2>&1 | tee e2_parallel_sensitivity.log
```

不同 `--out` 确保两批功能可以相互独立启动，或在服务器资源允许时同时运行。当前设置每个 ETA 条件的 `risk_stability_joint` 使用固定 MPC 样本，`mpc_samples` 只改变 MPC 样本数，`h_depth` 只改变 H 的值；**不要把不同实验参数混到同一个平均值里**。

## 7. 输出与汇总方式

```text
results/v10_e2_parallel_primary/
├── parallel_plan.json             # 输入、源码、checkpoint、分片定义和哈希
├── shards/
│   ├── multistage/default/case8__shock__seed400001__B2/
│   │   ├── runs.csv
│   │   ├── paired_summary.csv
│   │   ├── decision_audit.csv
│   │   ├── manifest.json
│   │   └── _success.json          # 完整验证通过后写入
│   └── risk_stability/default/.../
├── logs/<workflow>/<variant>/<shard>.log
├── merged/
│   ├── multistage/default/
│   │   ├── runs.csv
│   │   ├── paired_summary.csv
│   │   └── decision_audit.csv
│   └── risk_stability/default/
│       ├── stability_details.csv
│       └── stability_summary.csv
└── merge_report.json
```

关键特点：**合并的是所有独立实例的原始行，再重新计算均值、尾部 CVaR 和决策稳定性**；不是简单平均各分片的 `paired_summary.csv`，因此样本权重与原串行全量实验一致。多阶段深度 H0/H1/H2/H3 或 MPC M4/M8/M16 分别合并，互不混合。

## 8. 失败恢复：不覆盖已完成的分片

如果发生 OOM、单个子任务退出、服务器重启等：

```bash
# 同原始命令，仅增加以下选项；可以调整 --workers 数量
python scripts/run_e2_parallel.py \
  --workflows multistage,risk_stability \
  --workers multistage=6,risk_stability=8 \
  --cases 8,9 --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --risk-samples 16 --mpc-samples 8 --h 2 \
  --risk-sizes 16,64,128 --reference-samples 256 \
  --repeats 3 --states-per-group 2 --stride 10 \
  --out results/v10_e2_parallel_primary \
  --resume --retry-failed
```

**必须保持所有科学超参数相同**（包括 beta、H、样本数、抽样类型等）。改变 worker 数和 CPU 核绑定允许，不影响实验定义；改变模型、案例或源代码哈希时自动拒绝恢复，必须建立新结果目录。失败时保留已完成分片，只有显式设置 `--retry-failed` 才会删除重建**未完成**的分片目录。

若分片已完成，但合并因中断未执行：

```bash
# 沿用原科学参数，并添加 --resume --merge-only
python scripts/run_e2_parallel.py ... --resume --merge-only
```

`--merge-only` 的完整命令必须与首次提交时的科学参数一致，不能把省略号原样粘贴给 shell。

## 9. 可靠性保证与不能保证的部分

- **确定性**：逐分片固定 Case、scenario、ETA seed、预算、checkpoint、风险采样算法和模型；相同条件下 RNG 不依赖任务提交顺序，结果应与串行一致。
- **隔离性**：每个功能独立进程池、不同进程互不共享 Python 对象、专属输出和日志、使用 `taskset` 绑定 disjoint CPU 集（如果服务器安装了 taskset）。
- **完整性**：每个分片验证预算上限、策略集合、记录唯一性、输入源哈希和输出哈希；全部成功后才能合并。
- **统计一致性**：保持场景的配对关系，每个 H/M 单独汇总；不中途只挑成功案例形成貌似完整的结果。
- **局限**：若底层有未固定种子的全局 RNG、文件 I/O 副作用、异构 CPU 的浮点计算差异，仍可能导致数值波动；使用 `compare_e2_parallel.py` 发现并停止。**并行环境下测得的时间、峰值内存和吞吐不可与单任务运行时间直接作公平算法效率比较**，需要独占资源另跑计时基准。
- **版本边界**：不修改 V0.9/DDQN、V1.0-E2 风险和预算策略实现；若 E2 文件尚未安装，启动时会报错。

## 10. 测试报告

- `python -m unittest discover -s tests -p 'test_parallel_e2.py' -v`：6 个纯标准库测试通过。
- 用测试专用、无物理意义的模拟实验脚本进行 8 个独立分片的真实子进程并行运行；均退出 0，生成 4 组汇总；随后 `--resume` 正确跳过了全部 8 个已完成分片。
- **未在当前交付环境运行真实 IAMU E2 仿真**，因此不能声称真实闭环性能或完整集成测试通过。
