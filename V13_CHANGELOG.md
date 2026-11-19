# V1.3 P0–P3 交付清单

基础版本：`53b8c2e37bf73da46b637b41ccd739274cf604ab`。

本次只新增 `iamu_d3dqn/v13_diagnostics/` 和 `iamu_d3dqn/tests_v13/`，不覆盖 V1.1 / V1.2 文件，不包含或改动已有实验数据和模型 checkpoint。

- P0：同原始数据的三表示宏转移与浮点回报审计、动作语义/公共锚点、terminal/gamma、精确小型 MDP Q-gap，双层场景校验（运行环境 + 快照重放）。
- P1：100 frozen A0 epsilon 探索 + 100 均匀候选混合行为的单份固定原始数据，4-CPU 有序任务归并，所有外生 scenario 独立哈希。
- P2：独立样本种子、预算/场景配额的 120-state probe bank；预生成且同分支共享的时间索引未来；8 磁带配对 Keep/Edit，Greedy 或冻结 A0 continuation，输出原始已生成未来作为额外敏感性结果。
- P3：固定轨迹的分组 DDQN TD 目标审计 + 同状态双分支的第一个宏区间 TD-gap；汇总 Q-gap 与反事实的分预算和分场景误差。

严格限制：未实现第二批 P4 固定数据学习强度消融；未实现第三批势函数 shaping；未启动 V1.3 在线 400-episode 训练；不将合成未来视为已校准条件后验或 Q*；不将 model-only checkpoint 的 target=online 视为历史 target network。
