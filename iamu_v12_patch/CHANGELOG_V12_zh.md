# IAMU V1.2 代码变更清单

本包并未写入 GitHub；基于 `2d96981a38434014f14e31be5641d422d28835f8` 的增量修改通过 `apply_v12.py` 安装。

## 既有文件修改（6）

1. `iamu_dynamic/rl_env_v11.py`：修复未搜索事件的 `search_exhausted`，新增 `search_performed` 与测时。
2. `iamu_dynamic/rl_ddqn_v11.py`：带版本的 Transition 预算字段。
3. `v11_protocol/v11_actors.py`：显式每 episode 探索随机种子、预算变化与测时记录。
4. `v11_protocol/indexed_replay.py`：完整 ring/pool 顺序及采样 RNG 保存恢复。
5. `v11_protocol/fixed_label_probe.py`：从 fork 还原 gamma、bootstrap margin，区分 selector、evaluator 来源。
6. `v11_protocol/train_v11.py`：新科学指纹、独立恢复、Uniform 三类计数、原始事件更新额度、可选 tail collapse 和可选协调器。

## 新增文件

`v11_protocol/` 中新增研究诊断、分层 Replay 对照配置、恢复模块、派发协调器、反事实与 margin 扫描共 10 个文件；`tests_v11/` 新增 7 个专项测试文件（具体以 `PATCH_MANIFEST.json` 为准）。

## 验收范围

本地共执行 19 项测试，14 项通过，5 项因缺少完整 IAMU 仿真器和模型依赖明确跳过；针对 Replay 恢复的测试使用了从审查版本按相同源结构重建的本地 Replay 测试夹具。本地测试不能替代真实仓库、服务器 CPU/GPU 和历史 checkpoint 的完整验收。没有启动 400 episode 的正式训练。

## 未宣称完成

未验证线上 GitHub 分支写入（API 返回 403）；未在目标服务器应用补丁；未执行真实 Case 1–9 与 GPU 联调；未形成新的实验性能数据、统计结论或独立 Case10–14 测试。`collapse` 只是新方法开关，不能当作训练正确性修复。缓存目前在一次冻结 margin 扫描内复用基线，尚未提供跨运行的结果级共享缓存或全部任务的自动断点跳过。
