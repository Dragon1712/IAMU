# V1.0 迁移到 `iamu_Interface/`

旧版补丁以 `iamu_d3dqn/` 为运行目录，新增文件位于其 `iamu_dynamic/`, `experiments/`, `tests/`。

新版文件全部移至 `IAMU/iamu_Interface/`，且只使用此目录下的 V0.9 引擎快照。风险算法、预算算法、候选接口及评估逻辑均保留，主要改变运行路径及实验结果隔离方式。

- 旧：`cd iamu_d3dqn && python experiments/eval_risk_budget_v10.py ...`
- 新：`cd iamu_Interface && python setup_interface.py --copy-assets && bash scripts/run_smoke.sh`
- 旧输出：`iamu_d3dqn/results/v10_*`
- 新输出：`iamu_Interface/results/v10_*`

**为什么不把 V0.9 的全部训练/验证代码再复制一遍？** V1.0 冻结实验只需要最小闭环环境和模型推理，因此 `setup_interface.py` 拷贝八个必要的版本化引擎文件，避免将训练脚本与新实验不相关的历史模型目录搬进来。复制完成后无需通过 Python import 依赖原目录；只需指定模型权重/案例位置（可选本地复制）。

如果服务器当前已经把旧 V1.0 补丁放入 `iamu_d3dqn/`，可保留作历史备份。新工程无需从旧补丁中再拷贝任何代码，直接解压新版 ZIP 即可。
