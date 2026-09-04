<h1 align="center">🖐️ TacPAC</h1>

<p align="center"><strong>面向接触密集型操作的触觉预测与实时动作修正世界-动作模型</strong></p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="#安装">安装</a> ·
  <a href="#训练">训练</a> ·
  <a href="#推理">推理</a>
</p>

<p align="center">
  <img src="assets/tacpac_teaser.png" width="100%" alt="TacPAC 方法概览与定性结果">
</p>

TacPAC 是一个用于接触密集型机器人操作的触觉世界-动作模型。传统世界-动作模型可以在
执行前预测一个动作块将产生的接触，但动作一旦开始执行，这个预测就无法再修改已经规划
好的后续动作。TacPAC 将触觉预测转化为实时闭环修正：动作块执行期间，触觉专家持续读取
最新触觉图像，并结合规划阶段预测的接触信息，修正尚未执行的动作。

核心机制是可复用的逐层 tactile-action KV cache。缓存同时保留预测的未来触觉接触，以及
在该预测条件下生成的动作表征。每次修正只需对固定缓存做一次前向计算，无需重新生成整个
动作块。

> 论文、数据集和 checkpoint 正在整理并将后续发布。目前仓库包含模型、训练、触觉预处理、
> 部署协议和测试代码。

## ✨ 核心特点

- **以预测为参照的触觉修正：** 当前接触反馈不是孤立输入，而是与规划所预期的接触进行比较。
- **异步闭环执行：** 基础模型每个 action chunk 规划一次；机器人持续运动的同时，触觉帧只更新
  尚未执行的动作后缀。
- **低延迟缓存复用：** 实验中单次修正耗时 **30.4 ms（32.9 Hz）**，比重新生成动作块快
  **20.7 倍**。
- **真机效果：** 在五个接触密集型任务上，平均成功率由纯视觉基础模型的 **22%** 提升到
  **64%**，比最强对比方法高 16 个百分点。
- **训推一致的触觉处理：** 训练 dataloader 与推理 server 共用 raw、frame residual 和 stress
  触觉表示，并针对视觉/触觉采用不同的 resize 策略。

## 🧠 方法

<p align="center">
  <img src="assets/tacpac_pipeline.png" width="100%" alt="TacPAC 两阶段结构">
</p>

TacPAC 分两阶段训练：

1. **触觉预测世界-动作模型：** video expert 预测未来视觉与触觉观测，action expert 通过
   Mixture-of-Transformers attention 联合去噪动作块。
2. **触觉专家：** 冻结基础模型，为每个规划完成的动作块缓存干净的触觉与动作 K/V；训练时
   在动作块内采样多个执行偏移，只监督仍可修改的动作后缀上的 delta action。

推理时，`predict_action()` 先返回动作块，`prefill_tactile_cache()` 随后建立缓存；每个新的
触觉帧通过 `correct_action()` 修正当前计划。

## 📊 结果

每种方法、每个任务均进行 20 次真机实验。TacPAC 在五个任务上均取得最高成功率。

| 变体 | 触觉预测 | 在线修正 | 触觉缓存 | 插头 | 水果 | 薯片 | 空瓶 | 扩展卡 | 平均 |
| --- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯视觉 | ✗ | ✗ | — | 15 | 30 | 60 | 5 | 0 | 22 |
| 无触觉专家 | ✓ | ✗ | — | 35 | 50 | 65 | 20 | 15 | 37 |
| 无触觉预测 | ✗ | ✓ | ✗ | 40 | 25 | 45 | 30 | 25 | 33 |
| 无触觉缓存 | ✓ | ✓ | ✗ | 40 | 50 | 75 | **40** | 30 | 47 |
| **TacPAC** | ✓ | ✓ | ✓ | **80** | **65** | **90** | **40** | **45** | **64** |

五个任务分别为充电插头插入、多物体水果转移、易碎薯片转移、空瓶扶正和扩展卡插入。

## 🗺️ 代码结构

| 功能 | 路径 |
| --- | --- |
| TacPAC 主框架、缓存与修正流程 | `starVLA/model/framework/WM4A/WanMoTJointTacExpert.py` |
| 单步触觉专家 | `starVLA/model/modules/wan_mot/tactile_expert.py` |
| 世界-动作 Mixture-of-Transformers | `starVLA/model/modules/wan_mot/` |
| 训推共用触觉预处理 | `starVLA/dataloader/vla/tactile_stress.py` |
| LeRobot 数据与执行偏移采样 | `starVLA/dataloader/vla/dataset/` |
| 训练配置 | `starVLA/config/training/vla/starvla_wam.yaml` |
| `prepare` / `refine` 有状态推理协议 | `deployment/model_server/server_infersystem.py` |

<a id="安装"></a>

## 🛠️ 安装

```bash
git clone git@github.com:LogosRoboticsGroup/TacPAC.git
cd TacPAC

conda create -n tacpac python=3.10 -y
conda activate tacpac

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

下载 Wan2.2 TI2V backbone，并在 `starVLA/config/training/vla/starvla_wam.yaml` 中检查
`dit_path`、`vae_path`、`text_encoder_path` 和 `tokenizer_path`。

## 📦 数据准备

TacPAC 使用 LeRobot 格式示范数据。Flexiv 配置默认包含两个 RGB 视角和两个触觉视角：

```text
observation.images.third_view
observation.images.left_wrist_view
observation.images.left_wrist_left_tactile
observation.images.left_wrist_right_tactile
```

请在 `starVLA/dataloader/vla/mixtures.py` 中注册本地数据路径及 `video_keys`。包含
`tactile` 的 key 会被识别为触觉视图。论文使用的私有数据暂未包含在仓库中，在公开发布前
请将示例 `data_root` 替换为自己的 LeRobot 数据集路径。

先预计算两个训练阶段共用的文本 embedding：

```bash
python scripts/vla/precompute_text_embeds.py \
  --config_yaml starVLA/config/training/vla/starvla_wam.yaml \
  --datasets.vla_data.data_mix flexiv_plug_4views \
  --datasets.vla_data.text_embedding_cache_dir data/text_embeds_cache/flexiv_plug_4views
```

<a id="训练"></a>

## 🚀 训练

先激活环境。脚本默认使用 8 个进程，可通过 `NPROC_PER_NODE` 修改 GPU 数量，并在命令末尾
追加 Hydra 风格参数覆盖。如果缓存不在默认的 `data/text_embeds_cache/<data_mix>`，请设置
`TEXT_EMBEDDING_CACHE_DIR`。

第一阶段，训练触觉预测世界-动作模型：

```bash
NPROC_PER_NODE=8 bash scripts/vla/train_WanMoTJoint.sh \
  flexiv_plug_4views \
  stage1
```

第二阶段，冻结基础模型并训练异步触觉专家：

```bash
NPROC_PER_NODE=8 bash scripts/vla/train_WanMoTJoint-TacExpert.sh \
  flexiv_plug_4views \
  results/Checkpoints/vla/<stage1-run>/final_model/pytorch_model.pt
```

<a id="推理"></a>

## ⚡ 推理

```bash
CKPT=results/Checkpoints/vla/<tacpac-run>/final_model/pytorch_model.pt \
PORT=5556 \
bash deployment/local_infer-wan-tac.sh
```

有状态协议包含两个操作：`prepare` 生成新动作块并建立 `plan_id`；`refine` 发送最新触觉视图、
执行偏移和已执行动作前缀，返回该计划修正后的剩余动作。请求格式见
`deployment/model_server/README.md` 和 `scripts/test/test_infersystem_stateful_tactile.py`。

## 🧪 测试

```bash
python -m unittest \
  scripts.test.test_tactile_stress \
  scripts.test.test_tactile_expert \
  scripts.test.test_tactile_view_size \
  scripts.test.test_infersystem_stateful_tactile \
  scripts.test.test_infersystem_prefill_order
```

GPU 端到端 smoke test 位于 `scripts/test/smoke_tac_expert_gpu.py` 和
`scripts/test/smoke_stateful_tactile_server_gpu.py`。

<a id="致谢"></a>

## 🙏 致谢

TacPAC 的开发深受以下优秀开源工作的启发：[StarVLA](https://github.com/starVLA/starVLA)、
[FastWAM](https://github.com/yuantianyuan01/FastWAM)、
[T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex) 和
[Dream-Tac](https://github.com/LYFCLOUDFAN/Dream-Tac)。

<a id="引用"></a>

## 📝 引用

```bibtex
@article{ma2026tacpac,
  title   = {TacPAC: Tactile Prediction and Real-Time Action Correction in World-Action Models for Contact-Rich Manipulation},
  author  = {Ma, Zipei and Wei, Xiaofei and Jiang, Junzhe and Lu, Shunlin and Zhang, Li},
  year    = {2026},
  journal = {arXiv preprint}
}
```

使用本项目时，也请引用上游框架和实验所使用的 backbone。

## ⚖️ 许可证

代码采用 [MIT License](LICENSE)。第三方模型、数据集、机器人驱动和素材遵循其各自许可证。
