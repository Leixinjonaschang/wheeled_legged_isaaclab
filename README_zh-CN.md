<!--
  SPDX-FileCopyrightText: 2024-2026 LimX Dynamics Technology Co., Ltd.
  SPDX-License-Identifier: Apache-2.0
-->

# tron2_rl_lab

[English](README.md) | [中文](README_zh-CN.md)

> **发布渠道：** 本仓库的开源主副本托管于
> [`github.com/limx-tron2/tron2_rl_lab`](https://github.com/limx-tron2/tron2_rl_lab)。
> 内网 GitLab 地址
> `192.168.2.65:8022/rl/poc/tron/tron2_open_source/tron2_rl_lab` 仅作为
> LimX 内部开发使用的私有镜像。

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 LimX **WF-TRON1B** 轮足双足机器人（腿部位置控制、轮子速度控制）强化学习训练栈。仓库提供从 `wheeled_legged_mjlab` 迁移来的 teacher-student 速度跟踪任务：一个纯本体感知（blind）策略，和一个带深度相机、带 latent dynamics 预测的策略，均在生成的崎岖地形上训练。迁移与 mjlab 原实现的对应关系及其中的决策见 [`doc/mjlab_migration.md`](doc/mjlab_migration.md)。

## 许可与归属

本项目遵循 **Apache License, Version 2.0**（2004 年 1 月）发布。完整
许可文本位于仓库根目录 [`LICENSE`](LICENSE)。SPDX 标识符：`Apache-2.0`。

- [`NOTICE`](NOTICE) — 必需的归属声明。
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — 各组件来源
  说明：源自 Isaac Lab 的扩展、内嵌的 `rsl_rl/` fork（BSD-3-Clause）、
  开发工具的许可文本（包含一项在发布前必须分类的 GPL 条目）、
  随附的 STL / USD 资产以及文档媒体。
- [`SECURITY.md`](SECURITY.md) — 报告安全漏洞或与安全相关的
  RL / 奖励 / 终止问题的方法。
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — 开发环境搭建、内嵌
  fork 的更新策略、PR 检查清单、DCO 签署。
- [`CHANGELOG.md`](CHANGELOG.md) — 发布说明及当前未解决的
  **`⚠ TO CONFIRM`** 项（首个公开 tag 的阻塞项）。

关于子目录的说明：

- `exts/bipedal_locomotion/setup.py` 声明
  `license="Apache-2.0"`，与顶层 LICENSE 保持一致。
  基于 Isaac Lab 派生的逐文件归属审计仍标记为
  `⚠ TO CONFIRM`（见 `THIRD_PARTY_NOTICES.md` §1）。
- `rsl_rl/` 是
  [`leggedrobotics/rsl_rl`](https://github.com/leggedrobotics/rsl_rl)
  的 **内嵌 fork**，遵循 **BSD-3-Clause**；保留了上游的版权声明
  （见 `THIRD_PARTY_NOTICES.md` §2 及
  `CONTRIBUTING.md#vendored-rsl_rl-update-policy`）。

对上游的本地修改摘要见
[`CHANGES_VS_UPSTREAM.md`](CHANGES_VS_UPSTREAM.md) 与
[`licenses/dependencies/README.md`](licenses/dependencies/README.md)。

## 适用范围与除外

本仓库 **包含**：

- Isaac Lab 扩展 `exts/bipedal_locomotion/`，含 TRON2A 的 env /
  MDP / actuator / robot cfg。
- 内嵌的 `rsl_rl/` PPO 训练器与 on-policy runner。
- 入口脚本 `scripts/rsl_rl/{train,play}.py`，含 ONNX / JIT 策略
  导出（`scripts/rsl_rl/play.py:108-118`）。
- `SF_TRON2A` 与 `WF_TRON2A` 变体的 USD / STL 资产，位于
  `exts/bipedal_locomotion/bipedal_locomotion/assets/usd/`。
- 位于 `doc/` 的仿真回放 GIF（MuJoCo、Gazebo、实机）。

**不包含** — 有意为之：

- **不含训练后的策略**（`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、
  `*.jit`）。请自行训练，或从 LimX 内部制品库 / GitHub Release
  获取。
- **不含训练日志、TensorBoard events 或 Weights & Biases 记录。**
  `logs/`、`wandb/`、`events.out.tfevents.*` 均被 `.gitignore`
  屏蔽并在 CI 中拦截。
- **CI 不跑仿真测试。** CI 仅覆盖 `python -m py_compile`、`ruff`、
  违禁产物扫描与许可扫描；`tests/` 下的单元测试无需 Isaac Sim，但未接入 CI。
  在无头 GPU worker 上跑一次冒烟训练
  作为后续工作跟踪。
- **不含 Isaac Sim 的再分发。** 必须自行从 NVIDIA 获取 Isaac Sim
  5.1 并接受 NVIDIA 专有 EULA，才可安装本训练栈。本仓库不授予
  Isaac Sim 的任何许可。
- **不含 GPU / 驱动。** 4096-env 训练需要 CUDA 兼容 GPU
  （推荐显存 ≥ 12 GB）。本仓库不附带驱动、CUDA 或 PyTorch wheel。
- **不含 SDK 二进制、固件、标定值、客户配置或厂商 CAD。**

实机部署代码（实机上电、ROS 驱动、MuJoCo / Gazebo 集成）请参考
下文
[MuJoCo 仿真与实机部署](#mujoco-仿真与实机部署) 与
[Gazebo 仿真与实机部署](#gazebo-仿真与实机部署) 中引用的姊妹仓库。

## 仓库结构

```
.
├── exts/bipedal_locomotion/   # Isaac Lab extension：env/asset/MDP/sensor/terrain/robot cfg
├── rsl_rl/                    # 项目内 vendored 的 rsl_rl fork（teacher-student representation PPO）
├── scripts/rsl_rl/            # 训练 / play 入口（train.py / play.py / cli_args.py）
├── scripts/tools/             # MJCF -> USD 转换、冒烟 rollout
├── tests/                     # 单元测试与 mjlab 对齐测试（无需 Isaac Sim）
├── doc/                       # 迁移说明、媒体文件
└── logs/                      # 训练日志与模型权重
```

## 环境要求

- **Isaac Sim 5.1** + **Isaac Lab 2.3.1**
- Python 3.11
- GPU（推荐 ≥ 12 GB 显存，4096 envs 训练）
- `libGLU.so.1`（`apt install libglu1-mesa`）：缺少时 Isaac Sim 会在加载地形材质时卡死。

## 安装

```bash
# 1. clone 仓库
git clone <repo-url> tron1_rl_lab
cd tron1_rl_lab

# 2. editable install extension 与 vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

## IsaacSim 训练

任务 ID 均在 [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py) 中注册。

### 1. 训练模型

```bash
# 纯本体感知的 teacher-student 策略
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0 --headless

# 深度相机 + latent dynamics 预测的策略
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-Depth-Predict-OursGP-v0 --headless
```

*常用选项：*
- `--num_envs N`: 覆盖任务默认的 4096 个 env。
- `--max_iterations N`: 覆盖 runner 默认的 30,000 次迭代。
- `--logger wandb --log_project_name <project>`: 用 W&B 代替 TensorBoard 记录。
- `--video`: 开启录像。

日志路径：`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`。每次保存 checkpoint 都会在 run 目录下同时导出 ONNX，并附带 `wf_tron1b_deploy` 部署代码会校验的元数据。

### 2. 运行推理 (Play)

用 `-Play-v0` 后缀的任务 ID：关闭观测噪声、动作延迟、外推、相机标定扰动和各课程，机器人在随机地形块上重生。

```bash
python scripts/rsl_rl/play.py --task Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-Play-v0 \
    --checkpoint_path <model.pt>
```

*选项：* `--policy_role teacher` 运行特权 teacher 而不是可部署的 student；`--export_onnx` 在 checkpoint 旁写出 `exported/policy.onnx` 后退出。

### 3. Resume 续训

必须显式带 `--resume True` 才会加载 checkpoint。环境的步数计数器会一起恢复，课程从中断处继续。

```bash
# 方式 A：直接给 .pt 路径（推荐）
python scripts/rsl_rl/train.py --task <task> --resume True --checkpoint_path <path_to_model>

# 方式 B：按 run 名查找
python scripts/rsl_rl/train.py --task <task> --resume True --load_run <run_name>
```

## 机器人形态

| 形态 | 末端 | 驱动 | task id 前缀 |
|---|---|---|---|
| WF_TRON1B | wheel | 腿部位置控制 + 轮子速度控制 | `Isaac-Limx-WF-TRON1B-...` |
| SF_TRON2A | sole foot (ankle pitch) | 位置控制 | *（仅保留资产，任务已移除）* |
| WF_TRON2A | wheel | 位置控制 | *（仅保留资产，任务已移除）* |

### 已注册任务

| 任务 ID（play 时在 `-v0` 前加 `-Play`） | mjlab 原任务 | student 策略输入 |
|---|---|---|
| `Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0` | `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP` | 本体感知历史 + 指令 |
| `Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-Depth-Predict-OursGP-v0` | `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP` | + d435 深度图（CNN + GRU）；latent dynamics 预测 |

两个任务使用同一套生成地形（20 列：台阶、斜坡、障碍、随机起伏、随机方块、梅花桩、倾斜网格，与平地交错），并带地形等级课程。

机器人 USD 由仓库内的 MJCF 生成并已提交；重新生成：

```bash
python scripts/tools/convert_wf_tron1b_mjcf.py --headless
```

## 架构概览

项目分为三个主要部分：

1. **`exts/bipedal_locomotion/`** — Isaac Lab extension。包含 env / asset / MDP / robot cfg。
2. **`rsl_rl/`** — vendored fork。`scripts/rsl_rl/train.py` 会优先加载此路径下的算法库。
3. **`scripts/rsl_rl/`** — 训练与推理的入口脚本。

### 任务 Wiring 流程

以 `Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0` 为例：

1. **Gym 注册**：`tasks/locomotion/robots/__init__.py` 将任务 ID 绑定到 env cfg 类和 runner cfg 工厂函数。
2. **环境配置 (Env cfg)**：`tasks/locomotion/robots/wf_tron1b_env_cfg.py`（训练 / play / 深度变体）基于 `tasks/locomotion/cfg/WF_TRON1B/wf_tron1b_base_env_cfg.py`（场景、观测、动作、指令、事件、奖励、终止、课程），可调参数集中在 `cfg/WF_TRON1B/constants.py`。
3. **Runner 配置**：`tasks/locomotion/agents/wf_tron1b_rsl_rl_cfg.py`。
4. **资产配置 (Asset cfg)**：`assets/config/wheelfoot_tron1b_cfg.py`（USD 路径与执行器参数）。

## MuJoCo 仿真与实机部署

- [MuJoCo 仿真仓库](https://github.com/example/tron1-mujoco-sim)
- [Python实机部署代码](https://github.com/example/tron1-deploy-python)

### MuJoCo 部署效果（SF / WF）

<p align="center">
  <img src="doc/mujoco_sf.gif" alt="MuJoCo SF" width="48%" />
  <img src="doc/mujoco_wf.gif" alt="MuJoCo WF" width="48%" />
</p>

- 无法直接预览时可下载查看：
  - [mujoco_sf.gif](doc/mujoco_sf.gif)
  - [mujoco_wf.gif](doc/mujoco_wf.gif)

## Gazebo 仿真与实机部署

- [Gazebo 仿真仓库](https://github.com/example/tron1-gazebo-sim)
- [ROS实机部署代码](https://github.com/example/tron1-deploy-cpp)

### Gazebo 部署效果（SF / WF）

<p align="center">
  <img src="doc/gazebo_sf.gif" alt="Gazebo SF" width="48%" />
  <img src="doc/gazebo_wf.gif" alt="Gazebo WF" width="48%" />
</p>

- 无法直接预览时可下载查看：
  - [gazebo_sf.gif](doc/gazebo_sf.gif)
  - [gazebo_wf.gif](doc/gazebo_wf.gif)

## 实机部署效果（办公室场景）

<p align="center">
  <img src="doc/real_wf.GIF" alt="TRON2A 实机部署效果 1" width="48%" />
  <img src="doc/real_sf.GIF" alt="TRON2A 实机部署效果 2" width="48%" />
</p>

## 实机运行注意事项（强烈建议）

启动与落地流程建议固定为以下顺序，避免切策略瞬间冲击：

1. **将机器人吊起**，确保双脚不承重，先检查关节状态、急停和通信是否正常。
2. **先进入 IK 模式**，确认逆解控制稳定、姿态与期望一致。
3. **再缓慢将机器人放到地面**，观察接触是否平稳、是否出现异常抖动。
4. **最后切换到 walk 策略**，先低速小步验证，再逐步提升速度与动作幅度。

如出现异常（突发抖动、姿态发散、落地后冲击过大），请立即急停并回到吊起状态重新检查。

## 验证

下列命令与 CI 运行的内容一致（详见
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)），在提交 PR
前本地跑一遍可减少来回沟通。

```bash
# 1. 对全部一方 Python 文件做字节码编译。
python -m py_compile $(git ls-files '*.py')

# 2. Lint（仅一方子目录；内嵌的 rsl_rl fork 不做风格约束）。
pip install ruff
ruff check --select F exts scripts

# 3. Editable-install dry run。Isaac Sim / Isaac Lab 不在 PyPI，
#    传递依赖解析可能失败；此步目的是校验 setup.py / pyproject.toml
#    的元数据。
pip install --dry-run -e exts/bipedal_locomotion
pip install --dry-run -e rsl_rl

# 4. 在装有 Isaac Sim / Isaac Lab 的机器上做可选的 import 冒烟：
python -c "import bipedal_locomotion; print(bipedal_locomotion.__file__)"

# 5. 确认没有训练产物 / SDK 二进制被暂存。
git ls-files | grep -iE \
  '(^|/)logs/|(^|/)wandb/|\.pt$|\.pth$|\.ckpt$|\.onnx$|events\.out\.tfevents\.' \
  && echo "!! training artifacts staged" && exit 1 || echo "ok"
```

单元测试（MDP kernel 与 mjlab 参考实现对比、地形、深度缓冲、域随机化、
ONNX 元数据）需要训练环境中的 Python 包，但不需要 Isaac Sim：

```bash
python -m pytest tests
```

在装有 Isaac Sim 5.1 / Isaac Lab 2.3.1 的机器上，可以跑一次简短的
端到端冒烟：

```bash
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0 \
  --num_envs 16 --headless --max_iterations 2
```

## 参考资料

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — 内嵌 fork
  [`rsl_rl/`](rsl_rl) 的上游。

## 引用与支持

如在学术工作或公开成果中使用本训练栈，请引用本仓库：

```
@misc{limx_tron2_rl_lab_2026,
  title  = {tron2_rl_lab: RL training stack for the LimX TRON2A bipedal robot},
  author = {LimX Dynamics},
  year   = {2026},
  howpublished = {\url{https://github.com/limx-tron2/tron2_rl_lab}}
}
```

- **Bug 反馈 / 功能建议：** [GitHub Issues](https://github.com/limx-tron2/tron2_rl_lab/issues)。
- **问题咨询 / 集成协助：** [GitHub Discussions](https://github.com/limx-tron2/tron2_rl_lab/discussions)。
- **安全漏洞报告：** 邮件 `contact@limxdynamics.com`，另见
  [`SECURITY.md`](SECURITY.md)。
- **硬件 / 实机安全事件：** 邮件 `contact@limxdynamics.com`，
  主题前缀 `[tron2_rl_lab hardware]`，请勿在公开 Issue 中提交。
- **公司 / 商务联系：** <https://www.limxdynamics.com>。
