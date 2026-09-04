# StarVLA 部署与 LIBERO 推理

本文档记录如何在本仓库中启动 StarVLA policy server，并用 LIBERO 仿真客户端做推理/评测。LIBERO 评测采用 client-server 架构：

- `starVLA` 环境：加载 checkpoint，启动 websocket policy server。
- `libero` 环境：启动 LIBERO 仿真，向 policy server 发送观测并接收动作。

以下命令默认从仓库根目录执行：

```bash
cd /path/to/TacPAC
```

## 1. 环境检查

本机需要两个 conda 环境：

```bash
conda env list
```

应能看到：

- `starVLA`：用于模型推理服务。
- `libero`：用于 LIBERO 仿真评测。

不要直接使用 base 环境的 `python`。本机 base Python 是 3.13，通常不适合这套依赖；请激活对应环境，或显式设置 `STARVLA_PYTHON` / `LIBERO_PYTHON`。

快速检查：

```bash
conda run -n starVLA python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

LIBERO_HOME=/path/to/LIBERO \
LIBERO_CONFIG_PATH=/path/to/LIBERO/libero \
PYTHONPATH=/path/to/LIBERO:/path/to/TacPAC \
conda run -n libero python -c "from libero.libero import benchmark; import mujoco; print('LIBERO OK', mujoco.__version__)"
```

如果 `../LIBERO` 不存在，可安装 LIBERO 评测环境：

```bash
conda create -n libero python=3.10 -y
conda activate libero
bash examples/LIBERO/eval_files/install_libero.sh
```

## 2. Checkpoint

本机已有可用 checkpoint：

```bash
results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
```

也可以使用训练中间保存的 checkpoint，例如：

```bash
results/Checkpoints/vla/0610_WanMoT_libero_all/checkpoints/steps_20000_pytorch_model.pt
```

如果使用自己的模型，把下面命令中的 `CKPT` 改成对应 `.pt` 文件即可。

## 3. 单卡 smoke test

先做最小化测试：1 个任务、1 条 episode、不保存视频。这个测试用于确认 server 能加载模型、client 能连接 server、LIBERO 能正常 step。

### 终端 1：启动 policy server

```bash
conda activate starVLA

export CKPT=results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
export GPU_ID=0
export PORT=6694

bash examples/LIBERO/eval_files/run_policy_server.sh "$CKPT"
```

等待日志显示 server 已监听端口，例如：

```text
server running on ws://...:6694
```

### 终端 2：启动 LIBERO 客户端

```bash
conda activate libero

export LIBERO_HOME=/path/to/LIBERO
export PORT=6694

NUM_TRIALS_PER_TASK=1 MAX_TASKS=1 SAVE_VIDEO=0 \
bash examples/LIBERO/eval_files/eval_libero.sh "$CKPT" libero_goal
```

如果两个终端不共享 shell 变量，请在终端 2 重新设置 `CKPT`：

```bash
export CKPT=results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
```

## 4. 单 suite 完整评测

可选 suite：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

示例：完整评测 `libero_goal`，每个任务 50 条 episode。

```bash
conda activate libero

export LIBERO_HOME=/path/to/LIBERO
export CKPT=results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
export PORT=6694

NUM_TRIALS_PER_TASK=50 MAX_TASKS=0 SAVE_VIDEO=1 \
bash examples/LIBERO/eval_files/eval_libero.sh "$CKPT" libero_goal
```

## 5. 多卡完整四套评测

本机有 8 张 A100 时，可以为每张 GPU 起一个 policy server，然后用 DDP 客户端并行跑四个 suite。

### 终端 1：启动 8 个 policy server

```bash
conda activate starVLA

export CKPT=results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
export PORT=6694

bash examples/LIBERO/eval_files/run_policy_server_ddp.sh 8 "$CKPT"
```

默认会启动端口 `6694` 到 `6701` 的 8 个 server。

### 终端 2：跑四套 LIBERO

```bash
conda activate libero

export LIBERO_HOME=/path/to/LIBERO
export CKPT=results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt
export PORT=6694

EVAL_USE_CPU=1 NUM_TRIALS_PER_TASK=50 SAVE_VIDEO=0 \
bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh 8 8 "$CKPT"
```

参数含义：

- 第一个 `8`：policy server 数量。
- 第二个 `8`：LIBERO eval worker 数量。
- `SAVE_VIDEO=0`：完整大规模评测时建议关闭视频，减少 IO。

## 6. 输出位置

结果默认保存在 checkpoint 所在实验目录下：

```bash
results/Checkpoints/vla/0610_WanMoT_libero_all/results/pytorch_model/libero/
```

单 suite 输出：

```bash
<run_root>/<suite>/evaluation_results.json
<run_root>/<suite>.log
```

四套聚合输出：

```bash
<run_root>/overall_results.json
<run_root>/eval_summary.log
```

如果开启 `SAVE_VIDEO=1`，视频会保存到：

```bash
<run_root>/<suite>/video/
```

## 7. 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CKPT` | 脚本内默认 checkpoint | 待评测 `.pt` 文件 |
| `GPU_ID` | `0` | 单 server 使用的 GPU |
| `GPU_IDS` | 空 | 多 server 指定 GPU 列表，例如 `0,1,2,3` |
| `PORT` | `6694` | policy server 起始端口 |
| `LIBERO_HOME` | `../LIBERO` | LIBERO 仓库路径 |
| `NUM_TRIALS_PER_TASK` | `50` | 每个任务 episode 数 |
| `MAX_TASKS` | `0` | 限制任务数，`0` 表示全量 |
| `SAVE_VIDEO` | `1` | 是否保存 rollout 视频 |
| `ACTION_HORIZON` | `32` | 客户端执行动作 chunk 的 horizon |
| `USE_BF16` | `1` | server 端是否用 bf16 |
| `COMPILE` | `0` | server 端是否启用 compile |

## 8. 常见问题

### 连接失败

确认 server 已经启动，并且 client 使用同一个端口：

```bash
export PORT=6694
```

如果 server 跑在另一台机器，将 client 端的 `EVAL_HOST` 设置为 server IP：

```bash
export EVAL_HOST=<server-ip>
```

### Python 版本不对

不要用 base 环境直接跑脚本。使用：

```bash
conda activate starVLA
conda activate libero
```

或显式指定解释器：

```bash
STARVLA_PYTHON=/path/to/miniconda3/envs/starVLA/bin/python \
bash examples/LIBERO/eval_files/run_policy_server.sh "$CKPT"

LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python \
bash examples/LIBERO/eval_files/eval_libero.sh "$CKPT" libero_goal
```

### MuJoCo/OpenGL 报错

评测脚本默认设置了：

```bash
PYOPENGL_PLATFORM=egl
MUJOCO_GL=egl
```

如果仍然失败，先确认 `libero` 环境内能 import：

```bash
conda activate libero
python -c "import mujoco; print(mujoco.__version__)"
```

### 多卡进程残留

`run_policy_server_ddp.sh` 会把 PID 写到：

```bash
/tmp/starvla_policy_server_ddp/latest.pid
```

需要停止时可以按 PID 清理：

```bash
awk '{print $1}' /tmp/starvla_policy_server_ddp/latest.pid | xargs -r kill
```
