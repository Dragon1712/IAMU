# IAMU V1.2 Linux 双实验：完全离线部署

当前**已经存在**的研究源码根目录：

```
/home/moco/code/iamu/iamu_v12/
  iamu_d3dqn/
  2024LiERLDDE/
```

假定源码与 GitHub `Dragon1712/IAMU` 提交 `2d96981a38434014f14e31be5641d422d28835f8` 对应，新的增量补丁只安装在 `iamu_v12` 中。无须 `git clone`、`git pull`、`git worktree`，服务器可完全离线。

## 一、上传、解压

将 `iamu_v12_offline_bundle.zip` 上传到 `/home/moco/code/iamu/`，然后：

```bash
cd /home/moco/code/iamu
unzip -n iamu_v12_offline_bundle.zip
ls -l iamu_v12_patch/apply_v12_offline.py iamu_v12_two_experiments_offline.sh
```

不要将压缩包直接解压覆盖到 `iamu_v12/iamu_d3dqn/`。它包括补丁目录和独立启动脚本。

## 二、只读检查、离线安装

```bash
export IAMU_REPO=/home/moco/code/iamu/iamu_v12
cd /home/moco/code/iamu
python iamu_v12_patch/apply_v12_offline.py --repo "$IAMU_REPO" --check
```

**只有检查通过才安装**：

```bash
python iamu_v12_patch/apply_v12_offline.py --repo "$IAMU_REPO" --apply
```

离线安装器对六个待改旧源码进行**Git blob SHA-1**严格校验（与 GitHub 已审计版本的文件内容逐字节一致），同时验证训练/验证数据的九个文件存在，并先在 `/home/moco/code/iamu/v12_patch_backups/` 创建源码 ZIP 备份，再写入源码。不会读取/改写 `.pt` 或历史 `results/`。如果旧源码不一致或 V1.2 文件已经安装，直接拒绝，不使用 `--force`。

这个离线安装器**不会创建 Git 分支**；若研究目录内有 Git，可在安装后另行手工提交。V1.2 实验内部的代码、数据、任务列表指纹仍然会保存；无 Git 时 git.commit 可能为 null。

## 三、准备环境与测试

```bash
cd "$IAMU_REPO/iamu_d3dqn"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -m unittest discover -s tests_v11 -p 'test_*.py' -v
```

任何失败先检查再运行正式实验。推荐先做 8 episodes raw/collapse 冒烟，参见 `iamu_v12_patch/README_zh.md` 第三部分；实验目录务必首次新建，不可复用已有结果目录。

## 四、只并行两个实验

第一批为 `Uniform / Seed43` 与 `rho=0.50 / Seed43`：

```bash
cd "$IAMU_REPO/iamu_d3dqn"
bash /home/moco/code/iamu/iamu_v12_two_experiments_offline.sh dry seed43
```

确认只打印两个任务，且原样输出目录 **不存在** 后：

```bash
mkdir -p results/logs
nohup bash /home/moco/code/iamu/iamu_v12_two_experiments_offline.sh run seed43 \
  > results/logs/two_seed43_launcher.log 2>&1 &
echo $! > results/logs/two_seed43_launcher.pid
```

```bash
# 查看进度
tail -F results/logs/two_seed43_launcher.log
# GPU 监控
watch -n 2 nvidia-smi
```

默认 GPU0 / GPU1 各一个 Learner，4 actors + 2 validation workers。GPU2 / GPU3 空闲。第二批（Seed42）将脚本参数 `seed43` 改为 `seed42`，建议第一批完成后再运行。

历史 `.pt` 继续留在服务器原位置（例如 `/home/moco/code/iamu/iamu_d3dqn/results/v11_*`，以实际位置为准）；**新训练不需要复制旧 `.pt`**。已有 checkpoint 的 Q-gap 与反事实诊断应使用其绝对路径读取原 `.pt`，不要修改。

## 五、中断和恢复

正常结束已启动的旧训练进程后，确认两个实验的 `results/v12_*/training_state.pt` 存在，才能：

```bash
cd "$IAMU_REPO/iamu_d3dqn"
bash /home/moco/code/iamu/iamu_v12_two_experiments_offline.sh resume-dry seed43
bash /home/moco/code/iamu/iamu_v12_two_experiments_offline.sh resume seed43
```

如果已运行完 400 EP，勿再次恢复；先确认 `complete.json`、验证账本、结果完整性。恢复时科学代码、案例数据和关键参数必须与原始运行保持一致。

## 六、常见问题

1. **提示 Source mismatch**：服务器旧源码与已审计 GitHub 提交不同，或已安装过补丁。不要重复应用；先比较文件。
2. **提示 V1.2 patch not installed**：离线应用尚未成功，或者使用了错误的研究根目录。
3. **提示 `results/v12_xxx` already exists**：这是防止重复训练覆盖数据的保护。检查已有结果，选择续跑还是创建新实验矩阵与目录，不能删旧 `.pt` 图省事。
4. **使用原先的两实验脚本**：它默认查找 `~/code/iamu_v12`，不适合现有路径。请用本包的 `iamu_v12_two_experiments_offline.sh`，或显式 `export IAMU_REPO=/home/moco/code/iamu/iamu_v12`。
5. **没有 Git**：安装及训练并不依赖网络或 Git；`apply_v12_offline.py` 不调用 Git。
6. **PyTorch CUDA**：`nvidia-smi` 显示 CUDA 13.0 是驱动支持版本，检查 `torch.cuda.is_available()` 和实际 PyTorch 构建；无需为了驱动显示版本而强行安装相同版本的 PyTorch。
