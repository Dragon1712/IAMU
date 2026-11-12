# ERL-DDE Linux 移植补丁（根据用户上传的 BACAP_env.cpp 制作）

## 文件清单

- `BACAP_env.cpp`：用户上传原始 C++ 源码的 Linux 兼容版本；**未修改核心泊位/岸桥调度逻辑**。
- `env.py`：参考原作者公开 `env.py` 的接口重新适配，保持 `Bay_Env(file)`、`reset()`、`step()` 的原有返回约定。注意本文件并非从用户本地 `env.py` 修改，因为用户未提供它。
- `build_linux.sh`：在目标 Linux 上编译 `libBACAP_env.so`。
- `smoke_test.py`：测试结构体 ABI、加载、`reset` 与 `step`，支持 `--synthetic` 合成最小案例。
- `BACAP_env_linux.patch`：从用户上传的原文件到新版 C++ 源码的统一 diff，便于审计修改。

**不要在 Ubuntu 上使用 Windows 的 `BACAP_env.dll`。** 压缩包不包含在其他 Linux 系统上编译的 `.so`，请在目标服务器用其 GCC 13.3 本地编译。

## 部署

```bash
# 1. 在原仓库里备份（替换路径）
cd /实际路径/2024LiERLDDE
mkdir -p backup_before_linux
cp BACAP_env.cpp env.py backup_before_linux/

# 2. 解压后的 erldde_linux_ready/ 中四个文件覆盖原仓库对应位置
cp /解压目录/erldde_linux_ready/BACAP_env.cpp ./
cp /解压目录/erldde_linux_ready/env.py ./
cp /解压目录/erldde_linux_ready/build_linux.sh ./
cp /解压目录/erldde_linux_ready/smoke_test.py ./

# 3. GCC 编译
bash build_linux.sh

# 4. 安装 Python 依赖，推荐在原项目专用 Conda 环境
python -m pip install numpy  # smoke test 必需；训练还需 torch、matplotlib

# 5. 先做不依赖原案例数据的最小冒烟测试
python -X faulthandler smoke_test.py --synthetic

# 6. 然后用真实原始案例测试
python -X faulthandler smoke_test.py case_code/0.txt
```

## 兼容与修改摘要

1. 移除 4 处 `__declspec(dllexport)`，仍保留 `extern "C"`。
2. 导出 `step` 从 C++ `std::array<double,5>` 按值参数改为 `const double*`，在导出包装函数内复制成 `std::array<double,5>` 再调用原 `env->step(action)`。
3. 补充原文件使用 `std::round`、`std::numeric_limits` 等的必要标准头文件。
4. 用 C++ `static_assert` 和 Python `ctypes.sizeof/offset` 检查状态及回传结构体布局。
5. Python 加载脚本所在目录的 `.so`，修正参数类型，显式传入 `ctypes.byref(result)`，并增加 `close()` 释放资源。
6. 没有增加岸电、累计改约预算，也未修改核心调度策略或算法奖励。

## 已完成验证

在当前构建环境（Linux x86_64，GCC 14.2，glibc 2.41）已经：

- 成功运行 `g++ -std=c++17 -O2 -fPIC -shared`；
- 通过 `nm -D` 检查四个导出函数；
- `Array2D` 大小 1920 字节、`StepResult` 大小 1936 字节，reward/done 偏移 1920/1928；
- 使用合成三船案例执行 `reset` 和三步 `step`，没有崩溃并得到终止状态。

**未在用户 Ubuntu 24.04/G++13.3 上实际编译；也未运行原始案例和完整训练。** 请以服务器实测为准。

## 重要注意事项

- 原作者的 C++ `reset()` 主要重置时间等字段，并未完整恢复所有修改过的船舶/岸桥内部状态；独立 episode 建议重新创建 `Bay_Env` 对象，使用后调用 `close()`。
- 原始 `load()` 假定案例是有效的制表符分隔文本，并且至少有一条完整船舶数据。新版 Python 仅检查文件存在和非空，不会自动修复格式不合法的案例。
- `smoke_test.py` 的固定动作仅测试跨语言调用，没有训练收益意义。
- 原始 Windows DLL 不能直接转换成兼容 Linux 的二进制；必须从 C++ 源码本地编译。
