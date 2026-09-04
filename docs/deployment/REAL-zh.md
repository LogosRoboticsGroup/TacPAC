# StarVLA 真机推理部署指南

本文档说明如何用 StarVLA 在 GPU 机器上启动 InferSystem 兼容的推理服务，并让真机端的 InferSystem 通过 ZMQ/msgpack 获取 action chunk。

默认从 StarVLA 仓库根目录执行：

```bash
cd /cfs/starVLA
```

## 1. 架构

真机部署采用 client-server 架构：

```text
机器人端 InferSystem
  Robot.observe() + Sensor.read_images()
  -> InferenceClient.predict_chunk()/get_action()
  -> ZMQ/msgpack

GPU 端 StarVLA
  deployment/model_server/server_infersystem.py
  -> baseframework.preprocess()
  -> predict_action()
  -> postprocess()
  -> unnormalized actions

机器人端 InferSystem
  -> ActionDispatcher.dispatch()
  -> robot.act() + gripper.set()/move()
```

协议由 StarVLA 侧适配。InferSystem 真机端继续使用现有 `InferenceClient`，不需要改成 WebSocket。

触觉预处理同样统一在 GPU 端的 `baseframework.preprocess()` 完成。机器人端始终发送 raw tactile；server 会从 checkpoint 的 `config.yaml` 和 `data_mix` 识别全部触觉视角，并自动应用 `stress`、相对第 0 帧 residual 或 raw 模式。InferSystem 无需增加触觉模式配置，episode reset 会清空各触觉流的在线基准。

## 2. 上线前硬性检查

真机前必须确认下面几项，否则不要执行机器人动作：

- checkpoint 是为当前机器人和当前 action space 训练的。
- `stat_key` 对应当前机器人数据统计。
- InferSystem `build_state_vector()` 输出维度和顺序与训练时 state 一致。
- StarVLA 输出 action 维度和 InferSystem `ActionDispatcher` 的 `arm_dim/gripper_index/action_space` 一致。
- `camera_order` 与 InferSystem YAML 里的 `inference.enabled_cameras` 顺序完全一致。
- 相机视角、颜色、分辨率与训练数据一致或已在训练/预处理里覆盖。
- 首次真机测试必须低速、短 horizon、空场地，并保留急停。

尤其注意：仿真里的 Cartesian delta checkpoint 不能直接驱动真机 joint-position 控制。

## 3. GPU 端环境检查

在 GPU 机器上使用 `starVLA` 环境：

```bash
conda activate starVLA

python scripts/test/test_install_dependencies.py --groups deploy
python scripts/test/test_infersystem_protocol.py
```

如果缺 `pyzmq`，安装后再检查：

```bash
pip install pyzmq
```

如果当前 pip 源没有 `pyzmq`，可以用 conda-forge：

```bash
conda install -n starVLA -c conda-forge --override-channels pyzmq -y
```

## 4. 确认部署参数

准备以下变量：

```bash
export CKPT=results/Checkpoints/vla/<run>/final_model/pytorch_model.pt
export STAT_KEY=<your_robot_stat_key>
export CAMERA_ORDER=top,left_wrist,right_wrist
export PORT=5555
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `CKPT` | StarVLA checkpoint 路径 |
| `STAT_KEY` | checkpoint 中当前机器人对应的数据统计 key |
| `CAMERA_ORDER` | StarVLA 模型输入视角顺序，必须匹配 InferSystem `enabled_cameras` |
| `PORT` | ZMQ 推理服务端口，InferSystem YAML 中也要使用同一端口 |

可选地检查 checkpoint metadata：

```bash
python - <<'PY'
import os
from starVLA.model.framework.base_framework import baseframework

ckpt = os.environ["CKPT"]
stat_key = os.environ.get("STAT_KEY") or None

model = baseframework.from_pretrained(ckpt)
model.set_dataconfig(stat_key)
print("metadata:", model.get_metadata())
print("available stat keys:", list(model.norm_stats.keys()))
print("state stats shape:", len(model.norm_stats[model.stat_key]["state"]["mean"]))
print("action stats shape:", len(model.norm_stats[model.stat_key]["action"]["mean"]))
PY
```

## 5. 启动 StarVLA 推理服务

在 GPU 机器上启动 InferSystem 兼容 server：

```bash
conda activate starVLA

python deployment/model_server/server_infersystem.py \
  --ckpt_path "$CKPT" \
  --bind "tcp://*:${PORT}" \
  --stat_key "$STAT_KEY" \
  --camera_order "$CAMERA_ORDER" \
  --use_bf16 \
  --log_timing_every 20
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--bind` | `tcp://*:5555` | ZMQ REP 监听地址 |
| `--camera_order` | 必填 | 相机输入顺序 |
| `--stat_key` | `None` | 数据统计 key；多数据集 checkpoint 建议显式传入 |
| `--use_bf16` | 关闭 | GPU 推理用 bf16 |
| `--strict_cameras/--no-strict_cameras` | 开启 | 是否拒绝缺失或多余相机 |
| `--default_fps` | `None` | client 未传 fps 时给模型的默认 fps |

生产真机建议保持 `--strict_cameras` 默认开启。

## 6. Server-only smoke test

先不要接机器人。另开一个终端，用随机图像测试 server 通信和 action shape：

```bash
conda activate starVLA

python deployment/model_server/tools/debug_infersystem_client.py \
  --server "127.0.0.1:${PORT}" \
  --camera_order "$CAMERA_ORDER" \
  --state_dim <state_dim> \
  --prompt "pick up the cup" \
  --stat_key "$STAT_KEY"
```

成功时会看到类似：

```text
status=ok
actions_shape=(T, D)
first_action=[...]
```

这个测试只验证协议、模型加载和 shape。随机图像得到的动作不能用于判断策略质量，也不能直接发给真机。

如果 server 在另一台机器，先确认网络连通：

```bash
nc -vz <gpu_server_ip> "$PORT"
```

## 7. 配置 InferSystem 真机端

在 InferSystem YAML 的 `inference` 段设置 server 和相机顺序。例如 ARX5：

```yaml
inference:
  server: "<gpu_server_ip>:5555"
  fps: 30
  jpeg_quality: 90
  n_execute: 20
  enabled_cameras:
    - top
    - left_wrist
    - right_wrist

  action_space: joint_position
  arm_dim: 6
  gripper_index: -1
  gripper_threshold: 0.5
```

要求：

- `enabled_cameras` 顺序必须等于 GPU server 的 `--camera_order`。
- `arm_dim` 必须等于模型 action 中手臂部分维度。
- `gripper_index=-1` 表示模型 action 不单独交给 `BaseGripper` 控制。
- 如果 action 中有夹爪维度，确认 `gripper_index` 和 `gripper_threshold` 与训练语义一致。

## 8. 真机端分步检查

先检查硬件和相机，不要立刻跑闭环策略：

```bash
cd /cfs/starVLA

# 相机画面检查
python InferSystem/Example/realsense_visualize.py \
  InferSystem/Config/arx5_example.yaml

# 机器人连接检查
python InferSystem/Example/arx5/probe.py \
  InferSystem/Config/arx5_example.yaml --polls 3 --enable
```

不同机器人使用对应 example：

```bash
python InferSystem/Example/flexiv/probe.py InferSystem/Config/rizon4_example.yaml --polls 3 --enable
python InferSystem/Example/arx5/probe.py InferSystem/Config/arx5_example.yaml --polls 3 --enable
```

确认相机、机器人状态、夹爪都正常后，再运行推理控制：

```bash
python InferSystem/Example/robot_inference.py \
  InferSystem/Config/arx5_example.yaml \
  --prompt "pick up the cup" \
  --show-cameras \
  --save-video
```

首次执行建议：

- 降低机器人控制速度和加速度。
- 把 `n_execute` 设小，例如 `5` 到 `10`。
- 关闭夹爪或设置 `gripper_index=-1`，先验证手臂动作方向。
- 在空场地执行，确认动作尺度和方向后再放物体。

## 9. 常见问题

### 连接失败

确认 server 在 GPU 端已经启动，并且 InferSystem YAML 使用同一 IP 和端口：

```yaml
inference:
  server: "<gpu_server_ip>:5555"
```

如果使用防火墙或容器，确认端口可从机器人端访问。

### server 报 missing camera image

`--camera_order` 中有相机名没有被 InferSystem client 发过来。检查：

- InferSystem YAML `inference.enabled_cameras`
- GPU server `--camera_order`
- 相机是否打开成功

服务端不会为缺失相机补图；让机器人端和 GPU server 的相机列表完全一致。

### server 报 unexpected camera image

InferSystem 发了不在 `--camera_order` 中的相机。让两边相机列表完全一致，或确认是否有触觉相机也被编码进请求。

### action shape 不对

检查 `STAT_KEY`、checkpoint 和 InferSystem `ActionDispatcher` 配置。重点看：

- `action stats shape`
- `arm_dim`
- `gripper_index`
- `action_space`

### 动作方向或尺度明显不对

先停止真机。常见原因：

- checkpoint action space 与真机控制方式不一致。
- state 维度顺序不一致。
- 相机顺序或视角不一致。
- 使用了错误的 `stat_key`。

## 10. 推荐上线顺序

1. GPU 端依赖检查通过。
2. `test_infersystem_protocol.py` 通过。
3. StarVLA server 能启动并返回 metadata。
4. debug client 随机图 smoke test 返回 `status=ok` 和正确 action shape。
5. InferSystem 相机可视化通过。
6. 机器人 probe 和 home 流程通过。
7. 空场地低速、短 horizon 执行。
8. 加入夹爪。
9. 加入物体并开始任务测试。
