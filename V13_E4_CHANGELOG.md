# IAMU V13-E4 增量包（重新生成）

基准提交 `c49e8d09517da632433afe270d9a2d01649f5cb9`。

新增：`iamu_d3dqn/v13_boundary/`、`iamu_d3dqn/tests_v13_boundary/`。
不覆盖 V11/V12/V13 算法或已有实验数据。

功能：
- B1：40 个 source_budget=1 探针，Keep/Edit 同一未来轨迹、相同模型自策略或 Greedy 续行；EP100/EP400/Best 权重对照、逐条 SMDP 恒等式、零差状态后续首次 Edit 的实际修改语义。
- B1 汇总：八条 tape 先在每个状态内求均值，再对源 episode 聚类 Bootstrap；量化正号 Keep cost understatement。
- F：同原始轨迹 A0/A2、同初始化 G/Gshort 两条离线训练曲线，保存诊断 Q-gap、每个采样窗口曝光和 target 同步次数。
- R：从分离的训练标签银行生成共同 A0 frozen continuation 的配对 rollout，R1 监督差距，R2 A2+均匀预算辅助，R3 A2+预算一次辅助。
- `scripts/run_v13_boundary.sh` 默认 Dry Run，显式 RUN=1 + STAGE 执行单阶段。

当前代码文件与先前链接失效的版本按相同实验规格重新构建；没有原 ZIP 的二进制字节副本，故**不声明二者逐字节完全相同**。

限制：当前工作容器未装完整 IAMU 物理仿真库和服务器 `.pt` 文件；尚未完成真实仿真快照、GPU 训练与完整回归测试，绝不宣称修复已有效。
