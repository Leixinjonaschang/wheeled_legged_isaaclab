# mjlab → Isaac Lab 迁移：计划与决策记录

本文件记录把 `wheeled_legged_mjlab` 中两个训练任务迁移到本仓库的范围、结构对应关系，以及迁移过程中
所有"本该问一下 owner"的决策（按要求不中途提问，先按下列默认执行，owner 可逐条推翻）。

## 1. 范围

| mjlab 任务（参考源） | Isaac Lab 任务（本仓库） |
|---|---|
| `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP` | `Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0`（+`-Play-v0`） |
| `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP` | `Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-Depth-Predict-OursGP-v0`（+`-Play-v0`） |

- 参考版本：mjlab `exp_setting` 分支 HEAD `5e7c1af`。工作区里未提交的改动只新增了 eval-only 的
  `-BlindGP-Predict-DepthEnv` 任务，不影响上面两个任务。
- 不迁移：mjlab 的实验 / 评估 / 画图脚本（`scripts/eval`、`scripts/vis`、`run_ablation.py` 等）、
  ablation 任务（`LPGP`、`RGGP`、`*-no_dyn_ctx`、`*-DepthEnv`）。
- 已在仓库内的旧 port（`Isaac-Limx-WF-TRON1B-Velocity-{Flat,Rough}-v0`，plain PPO）是基于更早的
  mjlab 版本移植的（奖励权重、DR、地形、传感器都已过时）。见决策 D1。

## 2. 结构对应

| mjlab | Isaac Lab |
|---|---|
| `tasks/velocity/config/wf_tron1b/env_cfgs.py` 模块常量 | `tasks/locomotion/cfg/WF_TRON1B/constants.py` |
| `env_cfgs.py::make_*()` | `cfg/WF_TRON1B/wf_tron1b_base_env_cfg.py`（rough rep-ts lin-vel 超集） |
| `wf_tron1b_rough_rep_ts_lin_vel_[depth_]env_cfg()` | `tasks/locomotion/robots/wf_tron1b_env_cfg.py` |
| `terrain_cfg.py` | `cfg/WF_TRON1B/terrain_cfg.py` + 自定义 box 地形 |
| `mdp/*.py` | `mdp/wheeled_*.py`、`mdp/kernels.py`（纯 torch，可无 Isaac Sim 单测） |
| `rl_cfg.py` | `tasks/locomotion/agents/wf_tron1b_rsl_rl_cfg.py` |
| `rl/runner.py`、`rl/vecenv_wrapper.py` | `utils/wrappers/rsl_rl/` |
| `rsl_rl/`（fork） | `rsl_rl/`（同一 fork，整体同步） |
| `scripts/rsl_rl/{train,play}.py` | `scripts/rsl_rl/{train,play}.py` |

## 3. 工作分支

所有分支从 `feature/develop` 切出，review 后本地合并回 `feature/develop`（不 push）。

| 分支 | 内容 |
|---|---|
| `mjlab-port/rsl-rl-resync` | 用 mjlab HEAD 的 `rsl_rl` 整体替换旧快照 |
| `mjlab-port/wf-tron1b-usd` | 由 MJCF 生成并校验 USD（关节/刚体/碰撞体/惯量/轮子滚动） |
| `mjlab-port/terrain` | 20 列地形（含 4 种 mjlab box 地形）+ mjlab 课程语义 |
| `mjlab-port/mdp-resync` | 奖励 / 终止 / 指令 / 课程 / 动作 / 观测与当前 mjlab 对齐 |
| `mjlab-port/domain-randomization` | 87 维 dynamics context 对应的全部 DR |
| `mjlab-port/blind-task` | 组装 env、注册 BlindGP、脚本、运行时验证 |
| `mjlab-port/depth-task` | 深度相机、异步深度缓存、相机 DR、OursGP、运行时验证 |

## 4. 运行环境

- Isaac Sim 5.1 + Isaac Lab 2.3.1（本仓库 README 声明的版本），Python 3.11，torch 2.7.0+cu128，
  装在 `/home/ubuntu/venvs/wl_isaaclab`（uv venv）。机器上原有的 Isaac Lab 只有 Arena docker 里的 3.0，
  API 不同，不用。
- 自动化运行时通过 `OMNI_KIT_ACCEPT_EULA=YES` 接受 Omniverse EULA（机器上已在使用 Isaac Sim）。
- 没有接受 Anaconda channel ToS（其商业条款对大公司有约束），改用 uv。
- 本机缺 `libGLU.so.1`，Isaac Sim 在加载 MDL 材质（地形）时会卡死在 100% CPU。用户态解决：从
  `libglu1-mesa` deb 解出库放在 `/home/ubuntu/venvs/wl_isaaclab/extra_libs`，运行时加
  `LD_LIBRARY_PATH=/home/ubuntu/venvs/wl_isaaclab/extra_libs:$LD_LIBRARY_PATH`（有 root 时直接
  `apt install libglu1-mesa` 即可）。

## 5. 验证方法

1. 纯 torch 的 kernel 与 mjlab 参考实现逐项数值对比（`tests/`，无需 Isaac Sim）。
2. 参数对齐：以 mjlab 论文 run 保存的 `params/env.yaml` / `agent.yaml` 为准逐项核对。
3. 运行时 smoke：各观测组宽度、所有奖励/终止项有限、接触/DR 读回与记录一致。
4. sim2sim：把 mjlab 训好的 OursGP checkpoint（`logs/exp_ckpt/ckpt_with_dyn_ctx/model_29999_ours_seed*.pt`）
   直接放进 Isaac Lab 环境运行，检验观测 / 动作 / 深度图约定是否一致。
5. 短训练：与 mjlab 训练曲线（`logs/paper_train_logs/ours_seed*`；BlindGP 需另跑 mjlab 参考）在相同迭代数对比。

## 6. 决策记录（值得问 owner 的）

| # | 问题 | 默认决策 | 理由 / 影响 |
|---|---|---|---|
| D1 | 旧的 plain-PPO `Flat/Rough` 任务怎么处理？ | 删除注册及 flat 相关配置 | mjlab 已不再注册它们；它们需要第二套观测布局；保留会在基座重同步后静默改变行为。需要时可从 git 历史恢复。 |
| D2 | `BlindGP` 按注册的任务迁，还是按论文 blind 基线迁？ | 按注册的任务（`RepresentationVelocityActorCritic` + `RepresentationVelocityTeacherStudentPPO`） | `logs/paper_train_logs/blind_*` 实际用的是未提交的 `PredictBlindGP`（blind + latent dynamics predictor）。若要的是论文基线，只需再加 model 文件和 runner cfg。 |
| D3 | 轮-地摩擦的组合规则 | PhysX `friction_combine_mode="max"`，地形 μ=1.0 | MuJoCo 默认取两 geom 摩擦的 max（已实测验证）。mjlab 的轮子摩擦 DR 范围 [0.41, 1.14]，所以等效摩擦只在 μ_wheel>1 时才变化。为忠实复现采用 max；若本意是让 DR 覆盖低摩擦，应改 multiply（会改变训练分布）。 |
| D4 | `leg_joint_torque` 惩罚哪些关节？ | 全部 8 个执行器（腿+轮） | mjlab 的 `SceneEntityCfg(joint_names=腿)` 不会解析 `actuator_ids`（保持 `slice(None)`），`joint_torques_l2` 实际对所有执行器求和；论文 run 的 `params/env.yaml` 已确认。忠实复现。 |
| D5 | 地形课程语义 | 按 mjlab：每种子地形恰好一列、行难度 `row/(rows-1)`、按 proportion 把 env 分配到列 | Isaac Lab 默认按 proportion 分配列、env 均分、难度带随机抖动，三点都与 mjlab 不同，平地列占比会从 28% 变成 50%。 |
| D6 | Isaac Lab 没有的 4 种 box 地形 | 按 mjlab 几何重写为 mesh 地形 | `random_stairs`、`random_spread`、`stepping_stones`、`tilted_grid`。 |
| D7 | 深度相机实现 | 待 spike（TiledCamera vs RayCasterCamera） | mjlab 渲染 geom group 0/1（地形 + 机器人可视 mesh），TiledCamera 才能看到机器人自身；RayCasterCamera 快但只看地形。 |
| D8 | 默认 logger | tensorboard | 本仓库惯例；mjlab 默认 wandb（project `mjlab`）。命令行 `--logger wandb` 可切换。 |
| D9 | `num_envs` | 4096 | 与 mjlab 默认一致（旧 port 为 2048）。 |
| D10 | 接触检测 | 轮子/非法接触只统计与地形的接触力；self-collision 单独统计机器人自身接触 | mjlab 的接触传感器显式过滤 `secondary=terrain` / 自身子树。 |
| D11 | 编码器偏差语义 | 从位置目标中减去偏差，观测用真实关节角 | mjlab `joint_pos_rel(biased=False)`。 |
| D12 | 训练无关的 mjlab 部件 | 不迁移：`prepare_quantities`、`clear_non_finite_sim_data`、`mean_action_acc` metric、play 时的 debug 可视化、MuJoCo solver 参数（iterations/ccd/nconmax） | 无 PhysX 对应物或不影响训练。 |
| D13 | `init_velocity_prob`、零权重的 `heading_progress` | 删除 | 目标任务中都不生效（0），mjlab 已删 `heading_progress`。 |
| D14 | done 环境的 `applied_actions` | 使用 reset 前快照（`terminal_applied_action`） | 与 mjlab wrapper 一致，latent dynamics 训练用到。 |
| D15 | 是否提交生成的 WF_TRON1B USD | 提交（28.9 MB，可由 converter 逐字节复现） | 开箱即训；link 平铺在 `/robot/<link>`，碰撞体对渲染隐藏（与 mjlab 深度相机只看可视 geom 一致）。 |
| D16 | push 到远端 | 不 push | 对外操作，留给 owner。 |
| D17 | 轮子摩擦 DR 的实现 | 合成一个 event：每 env 一次共享采样 U(0.45,1.1) + 每轮独立 U(-0.04,0.04) | 分布与 mjlab 的两个 event 完全等价；特权观测的归一化范围仍是两者之和 [0.41,1.14]。 |
| D18 | `dynamics_context` 的来源 | startup 时记录采样值并据此构造，不每步从 PhysX 读回；布局与归一化逐项对齐 mjlab（MJCF 顺序） | 只在 startup 随机化，记录值与 PhysX 读回值一致（已运行时验证）；每步读回代价高。惯量 27 维 = 每个 body 的质量比 ×3（只随机 α 时与 mjlab 排序后的主惯量比严格相等）。 |
| D19 | MJCF 被动关节阻尼 0.01 | 放进 PhysX `viscous_friction`，PD 增益 DR 只缩放驱动 kp/kv | 与 mjlab 只缩放执行器 kp/kv 一致。 |
| D20 | MuJoCo 执行器 → PhysX 映射 | kp/kv → drive stiffness/damping；`frictionloss` → static+dynamic friction；`forcerange` → `effort_limit_sim`；不设关节速度上限 | 实测 Isaac Sim 5.1 的关节摩擦字段为与载荷无关的力矩，语义与 MuJoCo 一致；旧 cfg 的 `velocity_limit` 实际不生效，已删。 |
| D21 | 刚体最大角速度 | `max_angular_velocity` = 100 rad/s | 原 1000 是 deg/s（17.5 rad/s），会把轮速卡在动作可达的 20 rad/s 以下；MuJoCo 无此上限。 |
| D22 | PhysX 轮速稳态偏差 | 接受 | 带载时 PhysX 轮速比目标高约 +0.3%~+2.3%（MuJoCo −0.15%），调 solver/armature 无法消除，远小于 ±20% 的增益 DR。 |
