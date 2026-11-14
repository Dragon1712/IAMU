# iamu_Interface — IAMU V1.0 独立实验工程

**目的：** 将 V1.0 风险—预算策略、单船候选接口及其实验代码统一放在仓库根目录的 `iamu_Interface/` 内运行，**不修改 `iamu_d3dqn/` 的现有训练、诊断和结果目录**。

## 1. 目录布局

```text
IAMU/
├── iamu_d3dqn/                 # 现有 V0.9 训练项目，保持不变
├── 2024LiERLDDE/               # 现有港口数据和 C++ 探针源码
├── 2024LiERLDDE_GPU/           # 原始冻结 ERL-DDE 模型（可选）
└── iamu_Interface/             # 新增，V1.0 的唯一实验工作目录
    ├── setup_interface.py      # 一次性复制 V0.9 引擎源文件快照
    ├── dependency_snapshot.json  # setup 后自动生成版本与 SHA256
    ├── iamu_dynamic/
    │   ├── core.py, rl_env.py, ...  # setup 后保存的只读 V0.9 源码快照
    │   ├── v10_risk_features.py
    │   ├── v10_budget_lookahead.py
    │   ├── v10_policy.py
    │   ├── v10_candidate_api.py
    │   └── v10_candidate_integration.py
    ├── experiments/
    │   └── eval_risk_budget_v10.py
    ├── tests/
    │   ├── test_v10_modules.py
    │   ├── test_v10_integration.py
    │   └── test_interface_workspace.py
    ├── scripts/
    │   ├── run_smoke.sh
    │   ├── run_e1_validation.sh
    │   ├── run_budget_lookahead.sh
    │   ├── build_erldde_probe.py
    │   └── run_erldde_hint_smoke.sh
    ├── assets/                 # 可选：拷贝本地案例、冻结 best.pt
    └── results/                # 所有 V1.0 实验输出位于这里
```

## 2. 在服务器上安装

在 `~/code/iamu` 解压提供的 ZIP。ZIP 内**已包含顶层目录 `iamu_Interface/`**，请不要解压到 `iamu_Interface/` 里面（避免多一层嵌套）。

```bash
cd ~/code/iamu
unzip /path/to/iamu_Interface_V1.0.zip
cd iamu_Interface
```

**首次必须执行以下命令，复制 V0.9 物理环境引擎快照：**

```bash
python setup_interface.py --source ../iamu_d3dqn
```

该命令只读取 `../iamu_d3dqn/iamu_dynamic/`，然后将 8 个必要 Python 文件复制到本目录。不会写入原目录，也不会重命名原训练输出或检查点。运行时 V1.0 **只导入 `iamu_Interface/iamu_dynamic` 中的快照**，不再动态引用原训练源码。

可选：同时将 Seed 42 的 `best.pt`、训练 `manifest.json` 和 Case 8/9 复制到本目录，完全避免实验时读取外部权重与案例：

```bash
python setup_interface.py --source ../iamu_d3dqn --copy-assets
```

若文件已复制且哈希一致，会显示 `unchanged`；如原始源码发生变化，默认拒绝覆盖已有快照，防止同一系列实验混用代码版本。仅在准备开始**新的实验版本**时才手工使用 `--force`，并采用新的输出目录。

仓库里既有的 PyTorch、NumPy 环境可直接沿用（例如 `conda activate erldde`）。通常不必重新安装 CUDA 版 PyTorch。程序对冻结 V1.0 实验默认用 CPU。

## 3. 测试

```bash
cd ~/code/iamu/iamu_Interface
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m unittest discover -s tests -p test_interface_workspace.py -v
python -m unittest discover -s tests -p test_v10_modules.py -v
python -m unittest discover -s tests -p test_v10_integration.py -v
```

本包的源码语法和工作目录快照安装测试已在当前交付环境通过；真实 V0.9 物理仿真及检查点评估需要在服务器完成（本交付环境没有模型权重与完整仿真文件）。

## 4. E1：冻结模型风险—预算实验

启动冒烟 + 原始 V0.9 基线一致性检查：

```bash
bash scripts/run_smoke.sh
```

全量验证集（Case 8/9，3 场景，3 ETA 种子，预算 2/5/10）：

```bash
bash scripts/run_e1_validation.sh
```

比较 Keep、Periodic、Greedy、Set-DDQN、Q+Risk、Q+Budget、Q+Joint、Expected。每个脚本允许在后面追加命令行覆盖参数，如：

```bash
bash scripts/run_smoke.sh --beta-risk 0.5 --out results/v10_smoke_risk05
```

多阶段预算前瞻计算较慢，先单独运行冒烟：

```bash
bash scripts/run_budget_lookahead.sh
```

也可直接运行 Python 入口，指定不同的 checkpoint 和案例：

```bash
python -u experiments/eval_risk_budget_v10.py \
  --checkpoint ../iamu_d3dqn/results/v09b_set_ddqn_seed42_300/best.pt \
  --manifest ../iamu_d3dqn/results/v09b_set_ddqn_seed42_300/manifest.json \
  --cases ../2024LiERLDDE/case_code/8.txt ../2024LiERLDDE/case_code/9.txt \
  --smoke --risk-samples 8 --verify-baseline \
  --out results/v10_manual_smoke
```

**默认路径：** 如果存在 `assets/checkpoints/set_ddqn_seed42/{best.pt,manifest.json}` 和 `assets/cases/{8,9}.txt` 则优先使用，否则读取兄弟目录中现有 V0.9 检查点与案例。`--out` 默认位于本目录下的 `results/`，再次运行同一输出目录会拒绝覆盖。

输出文件都位于 `iamu_Interface/results/<实验名称>/`：`runs.csv`、`paired_summary.csv`、可选 `decision_audit.csv`、`manifest.json`。脚本日志位于 `iamu_Interface/` 根目录。

## 5. E3：冻结 ERL-DDE Hints 候选实验（可选）

先构建 **独立**的只读 C++ 探针，不修改原有 `libBACAP_env.so`：

```bash
python scripts/build_erldde_probe.py --source ../2024LiERLDDE
```

随后指定与 `2024LiERLDDE_GPU/model.py` 结构兼容的受信任冻结权重：

```bash
export ERLDDE_REPO="$(realpath ../2024LiERLDDE_GPU)"
export ERLDDE_CHECKPOINT='/abs/path/to/compatible/synced_model_0.pth'
# 可选 export ERLDDE_PROBE=...
bash scripts/run_erldde_hint_smoke.sh
```

ERL-DDE 目前仅提供船舶顺序与位置偏好，通过现有 SlotPlanner 生成新计划，**尚非完整 ERL-DDE 动态重优化**。该候选源改变了原 RL 的候选分布，因此冻结模型结果属于探索性诊断。

## 6. 安全、版本与统计约束

1. 保持 `iamu_d3dqn` 原训练不变；不在原目录写入 V1.0 文件；所有新结果都写在 `iamu_Interface/results/`。
2. 首先通过 `--verify-baseline`，确保新目录冻结 Set-aware DDQN 与原 V0.9 行为一致。
3. 风险采样仅依赖当前公开 ETA 和可观测是否到港标志；不使用尚未公布的未来事件/到港真值。
4. 当前 V1.0 是冻结推理策略与单船候选接口，不包含重训练、多船硬预算修改、岸电/排放物理模型。
5. Case 8/9 用于诊断与参数筛选；Case 10–14 的现有独立测试安排不要用于 V1.0 反复调参。
6. `dependency_snapshot.json` 记录源文件哈希，正式结果 `manifest.json` 记录实验配置和风险模块哈希。

**方法依据和已上传实验分析：** 见同目录 `V10_实验依据与当前结果.md`。
