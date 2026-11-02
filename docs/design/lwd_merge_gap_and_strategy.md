# LWD 边云分支合并：差异分析与融合策略

> 分析日期：2026-11-02。分析方式：对两侧分支做 bare-clone + worktree 只读对比，并用 `git merge-tree` 做无工作区试合并。
> 本文档为汇总与结论层，逐文件/逐类/逐冲突块的明细已归纳进正文各节。

---

## 1. 任务背景与对象

| 侧 | 仓 | 分支（下称） |
|---|---|---|
| A 侧（源） | `vllm` | `cr_mi_to_23.0_azy_1029_memc` |
| A 侧（源） | `vllm-ascend` | 同上 |
| B 侧（目标） | `vllm`（本仓） | `main` |
| B 侧（目标） | `vllm-ascend` | `main` |

目标：把 A 侧（云复用、多实例/多边多云等）的代码合并进 B 侧主干。本文档说明差异与融合策略，实际合并不在本次范围内。

## 2. 血缘关系（决定合并策略的根本事实）

**两侧 LWD 代码同源、分叉演进，并非两套独立发明。**

- vllm 仓共同基线：`0fc695fc6d`（2026-06-14，上游 vLLM 提交）。
- vllm-ascend 仓共同基线：`97b216a6b7`（2026-07-31，v0.23.0 系）。
- **B 侧 `vllm/v1/lwd_control/` 的首个提交（`291586c9b0`，2026-10-12）明确写着 "port lwd_control control plane onto ghdc/main from `prefill_only_ww_1009_refact`"** —— 即 B 的控制面是 A 家族分支（约 2026-10-09 状态）的移植，之后重写了通信层（PR #11）。
- vllm-ascend 侧同理：B 的数据面首提交 `82ce44839` "port the edge-cloud data plane package from the mm branch"（内部 mm 分支）。
- 铁证：两侧存在同名同值枚举 `LwdBatchType`、同名函数 `_lwd_build_first_pass_embeds`、中英文对照注释、`lwd_hash` 两文件语义 diff = 0。

分叉后的演化方向：

| | A 侧（qxxxw） | B 侧（datacomm main） |
|---|---|---|
| 演化主线 | prefill_only → cloud_reuse → multi_instance/多边多云 | 控制面移植 → 重写通信层 → 多 DP → host staging → 持续吞并上游 |
| 特征 | 内部交付形态：功能纵深、性能直达路径、中文注释 | 社区主干形态：工程规范（英文/SPDX/CI/守卫测试）、上游大合流 |

**推论**：同名目录/同名类（如 `LwdEdgeScheduler`）是同源分叉的必然结果；这些文件不能按文本三方合并处理，必须先做"主干二选一"的架构决策（见 §7）。

## 3. 总体规模差异

| 指标 | vllm 仓 A | vllm 仓 B | va 仓 A | va 仓 B |
|---|---|---|---|---|
| 独有提交 | **454**（2026-11-02 又并入 `baseline_update1031`，+13） | **4123**（其中自有 lwd 功能约 69，其余为上游同步） | **219**（+5） | **1730**（大部分上游同步） |
| 改动文件/行（相对基线） | **103** 文件 **+28693/-175** | 5055 文件 +917596/-184414 | **62** 文件 **+11901/-455** | 3379 文件 +638921/-239322 |
| LWD 专属代码 | `lwd_control` 27 文件 13600 行，生态合计 ≈**2.2 万行**（11-02 前缀模块重构后） | `lwd_control` 25 文件 5980 行 + kv_offload 扩展 ≈**1.6 万行** | ≈7066 行（worker 4401 + lwd_comm 1976 + lwd_wire 689） | ≈5678 行（worker 4141 + lwd_data_plane 1537） |
| LWD 测试 | 5733 行（vllm）；2216 行（va）（11-02 后） | 18 文件 / 3319 行（vllm）；17 文件 / **3821 行**（va） | | |
| 部署示例 | **3 套**（va `examples/`: 2e1c/2e2i/multi_instance） | 0（工具/文档在 vllm core：`tools/gen_lwd_topology.py`） | | |
| CI | 无（内部代码托管环境） | **lwd_fast / lwd_image / lwd_checklist** 三个专用 workflow | 无 | 守卫测试进 CI |

规模悬殊的本质：B 侧吞并了 vLLM / vLLM-Ascend 上游大版本合流（worker/gpu 子包化、launchers 重构、rust 栈、V2 model runner、DSpark/dflash spec decode、w4a4/w4a8 量化、kvpp/ec_transfer/mooncake 等 0.24+ 能力），A 侧钉在 0.23/6 月基线上做功能纵深。**这是合并的动机之一：A 的功能必须"搬进"新得多的 B 基座。**

### 依赖差异（requirements）

| 依赖 | A | B | 融合后 |
|---|---|---|---|
| torch-npu (va) | 2.10.0.post2 | 2.10.0.post4 | 取 B |
| triton-ascend (va) | 3.2.1 | 3.2.2 | 取 B |
| transformers (va) | 5.5.4 | 5.14.1 | 取 B |
| transformers (vllm) | >=4.56 | >=5.10.4 | 取 B |
| memfabric_hybrid / memcache_hybrid (va) | 无版本号 | ==1.2.0 | 取 B 固定版本 |
| fastokens (va) | 无 | ==0.2.0 | 取 B |
| diskcache (vllm) | ==5.6.3 | 已移除（上游） | 随 B，确认 A 无直接依赖 |
| fastapi / starlette / prometheus (vllm) | 旧区间 | 新区间（含 CVE 修复） | 取 B |

## 4. 架构差异

### 4.1 vllm 仓：LWD 控制面两套实现（最大冲突域）

| 维度 | 融合前：A（`vllm/v1/lwd_control/` 26 文件） | 融合前：B（`vllm/v1/lwd_control/` 25 文件） | 融合后（建议） |
|---|---|---|---|
| 通信层 | `control_communication/` 13 组件，**三代传输并存**（PUSH/PULL → ROUTER/DEALER → ROUTER-ROUTER mesh），HELLO/WELCOME 裸帧 | `lwd_comm/` 6 组件，**一代收敛**：云 bind ROUTER / 边 per-DP DEALER，版本化 Hello/Accept/Reject 握手 + keepalive + IMMEDIATE | **以 B `lwd_comm` 为骨架**，扩展出 ROUTER-ROUTER mesh transport 子类承载 A 的多边多云（A 的不可丢 publish 语义 + per-identity FIFO 一并带入） |
| 消息协议 | `lwd_notify.py`：12 种消息，`LWD_WIRE_VERSION=**5**`（2026-11-02 起含 prefix nonce/READY/REFILL/is_refill + `structured_outputs`） | `lwd_messages.py`：8 种消息，`LWD_MESSAGE_VERSION=3`，含 prefix_nonce/refill | **统一到 v5**：B 的消息集 + A 的多云维 + `structured_outputs`，按"追加带默认值字段"策略合入 |
| 拓扑/身份 | `LwdRoleRegistry`（YAML，进程级单例）+ **CLI 数字推导**（`--edge-npu-count` 等按公式算 rank） | pydantic `LwdTopology`（topology YAML，`extra=forbid`，sha256 digest，scene 四态）+ `tools/gen_lwd_topology.py` | **B 拓扑为唯一事实源**；A 的 CLI 推导降级为"生成 topology YAML 的工具层"，registry 格式一次性迁移 |
| 云侧调度 | 双调度器：`LwdCloudMixedScheduler`（默认混排）+ `LwdCloudPhaseScheduler`（旧相位机） | 单调度器 `LwdCloudScheduler`（arrival 驱动） | B 调度器为主干（与 A 的 mixed 高度同源，B 即移植后精化）；A 的 phase 实验臂可选保留 |
| 边侧调度 | `LwdEdgeScheduler` 2998 行：**多云路由/健康冷却/RR 兜底/多实例桶**；probe 续算分支已随 HTTP 死链路失效（代码待删） | 1046 行 + 拆件：`lwd_edge_picker`（组批）/`lwd_cloud_dp_account`（DP 记账）/`lwd_edge_executor_proxy` | **按 B 的拆件结构重组**，A 的多云路由独立成模块移植 |
| 前缀协调 | **2026-11-02 起已迁移为控制面 nonce 预留**（`lwd_prefix_cache.py`/`lwd_edge_prefix_control.py`，与 B 同构）；HTTP/SSE 探针与 HMAC 租户链代码仍在树内但**运行时未接线**（`LwdCloudControlProcessor` 无实例化点，探针必超时→边侧 fail-open） | **控制面 nonce 预留**（`LwdPrefixReservations` 挂 kv_cache_manager 5 处钩子）+ `LwdRefillNotify` 重嵌入 | **A 已收敛到 B 的设计**：以 B 账本/状态机为唯一实现，保留 A 的多云路由维度（cloud_id/engine）与 refill MQ；A 的 HTTP/HMAC 死链路在合并时删除（若未来需要跨网关隐私探针，另立插件）。详见 §7.4.1（2026-11-02 修订） |
| 引擎入口 | `lwd_resolve_engine_cls`/`lwd_serve_guard` 钩子（`__init__.py` 150 行挂接） | `core.py` 内联分流 + `lwd_base_engine.py` 基类（203 行统一启动/收发骨架） | 取 B 基类分层；A 的 serve_guard fail-fast 校验作为薄钩子保留 |
| 诊断 | `vllm/v1/lwd_debug/`（309 行，7 文件） | `lwd_control_diag/`（358 行单文件枢纽）+ `lwd_logger` | 取 B 收敛形态 |

### 4.2 vllm 仓：数据面集成深度
> **PP 并行（LWD）**：B 主干已移除 LWD 的 PP 支持——`lwd_check_parallel_config` 显式拒绝 `pipeline_parallel_size > 1`；`LwdParallelGroupsSpec` 无 PP 组（PP/DCP/PCP 用 singleton）；无 `_LWD_FULL_HEAD_TAIL`/`head_tail` 模式；va 基类零织入。因此合并后必须删除 A 的 LWD-PP 代码（完整清单与验收见 §7.4.7-D7），但 **B 原生（非 LWD）PP 代码一行不动**。

| 集成点 | A | B | 融合后 |
|---|---|---|---|
| `multiproc_executor.py` | **+455 行**（early-recv MQ、本侧 broadcast、instance rank 集） | 零改动 | A 块向 B 上游新 executor 结构重写式移植 |
| `distributed/parallel_state.py` | **+547 行**：`LwdParallelGroups`（CLI 推导建组，instance-major 布局） | `distributed/lwd/lwd_parallel_groups.py`（237 行，`LwdParallelGroupsSpec.from_topology`） | **统一为 topology 派生（B 形态）**，A 的 registry 作为拓扑源之一 |
| `gpu_model_runner.py` | **212 行改动 / 44 处 lwd 引用**（角色标志、spec 草稿 stash、layer trace、占位 embeds） | 零 lwd 改动（runner 干净，数据面在 `distributed/lwd`） | A 的 44 处向 B 的 `worker/gpu/` 新 runner 结构重落，**最大重定位成本** |
| `kv_cache_manager.py` | +30 行（`lwd_resume_cap` 续算边界 + probe pin 校验，fail-closed） | +408 行（`lwd_prefix_caching` 5 处钩子 + reservations） | 语义裁决后统一（见 D1） |
| `scheduler.py` / `sched/output.py` / `outputs.py` | 小增量（`lwd_window_cap` 两处、批型三件套 +123、`LwdC2eMeta`） | 同名同源小分叉（B 另加 mrope 直传、CoW 字段） | 同源可机械合并，取 B 为底并入 A 的 cloud_id 字段 |

**结论**：A 是"深侵入换性能直达"，B 是"窄接缝换可维护性"。融合必须按 B 的集成纪律重放 A 的数据面逻辑，**这部分不可文本合并**。

### 4.3 vllm-ascend 仓：数据面两套实现

| 维度 | 融合前：A | 融合前：B | 融合后（建议） |
|---|---|---|---|
| 目录组织 | `worker/lwd_cloud/`（4401 行）、`worker/lwd_edge_worker.py`（679 行单文件）、`distributed/lwd_comm/` + `lwd_wire.py` | **目录已重命名**：`worker/lwd_cloud_model/`、`worker/lwd_edge_model/`（worker/runner/mm_cache 三分）、`distributed/lwd_data_plane/` | 取 B 新目录结构（A 的 `platform.py` worker_cls 指旧路径，直接沿用会 ImportError） |
| 数据面键控 | `(edge_id, cloud_id, engine_idx, channel)` 四元组全 mesh + echo 回声握手 + watchdog | `(dp_idx, channel)` 二元组，DP 连接来自 `LwdParallelGroupsSpec.from_topology` | 统一为 B 的 `LwdDPConnection{edge_id, cloud_id, dp_idx}` 连接抽象（已含 A 的三维语义） |
| 基类纪律 | **直接改基类**：`worker.py` +139、`model_runner_v1.py` +89 行织入 LWD 分支 | **零织入**：`LwdWorkerBase`/`LwdBaseModelRunner` 子类化，AST weave-guard 测试强制基类零 LWD 符号 | 取 B 纪律（守卫测试进 CI）；A 的织入下沉到子类再迁回 |
| host staging | `lwd_host_stage.py` 770 行（D2H 分段 + per-request pinned） | 741 行且是 A 的**超集**：新增 prefix 语义（`seed_prefix`/`_note_refill_arrival`/`skip_consumed`/`take_refill_done`）+ 589 行 22 用例测试 | **直接取 B** |
| 云复用续算 | `LwdPrefixCoordinator`（probe/claim/sweep/finish_chain）+ `_lwd_resume_base` 续算基线 | 无跨边复用（云内 nonce 预留） | A 独有，整体移植；`_lwd_resume_base` 与 B 的 `stage_floor` 语义同构，**二者合一** |
| 早收 early-recv | per-edge MQ + 配额 + 边摘除 | 单 MQ + 句柄序列化传递 + done-MQ + fenced harvest（专用流 + 池化 pinned） | 取 B 单 MQ 架构，多边形态回加 per-edge 维度 |
| spec 恢复链 | `LwdDraftCache` + `_lwd_draft_stash` + spec persist（PhaseScheduler 抢占臂配套） | **已删除**（改用 defer prefill 不逐出 decode） | 按调度策略取舍：若采 B defer 策略则丢弃；若保留 A 相位臂则子类化重放 |
| hash MoE（DSv4） | `experts_selector.py` 整文件重写（+438/-417）+ 0.23 双 runner 注入 | 上游已重构为 `ops/fused_moe/{router/,dataclass/,...}` 子包，注入点移到 `models/deepseek_v4/model.py:391` 前向 | **语义重放**：把 `merge_hash_expert_ids`/`moe_gating_top_k_hash` 钩子移植进 B 新 router 结构（modify/delete 冲突，行级合并必错） |
| worker 选择（platform.py） | `lwd_config.enabled` + `is_edge`，插在 310p/xlite 之前 | `lwd_config is not None` + `role`，选择独占（硬件默认不可覆盖） | 取 B 独占式；A 的 prefill_only PP 豁免逻辑重新评估 |

### 4.4 "memc" 的真相（避免误解的重要澄清）

**分支名后缀 `_memc` 不是 qxxxw 的功能开发量。** 实测：`git diff 97b216a6b7..HEAD -- vllm_ascend/device_allocator/ vllm_ascend/simple_kv_offload/ ...` 为空——memcache/memfabric KV 池、camem 分配器、ascend_store 连接器是 **2026-07-31 基线自带的上游能力**，分支名含义是"把 cloud_reuse/multi_instance 移植到带 memc 的 23.0 基线"。vllm 仓 103 文件 diff 中亦无任何 memc 标识。

B 侧反而通过 1730 个上游同步提交让 memc 生态更完整（ec_transfer、mooncake、kv_offload native/preempt、kvpp_cache 等）。**融合不需要任何"memc 合并"动作。**

### 4.5 配置体系差异

| 配置面 | 融合前：A | 融合前：B | 融合后 |
|---|---|---|---|
| LwdConfig 定义（vllm） | `vllm/config/lwd.py` 单文件（256 行 dataclass）+ `lwd_coordination.py`（189 行）+ `parallel.py` 里 `LwdParallelConfig`（+434 行） | `vllm/config/lwd/` **pydantic 包**（6 文件 671 行：entry/topology/api/coordination/logging） | ⚠️ **同名遮蔽隐藏冲突**：Python 包优先于同名模块，合并后 A 的 `lwd.py` 会被静默遮蔽。必须显式删除 A 单文件版，把 `LwdCoordinationConfig`（云复用）并入包内 |
| 拓扑声明 | registry YAML / CLI 数字推导 | topology YAML（pydantic forbid + digest） | B 为准，A 源做迁移工具 |
| CLI 参数（vllm） | `--edge-npu-count/--cloud-npu-count/--instance-parallel-size/--cloud-dp-size/--role-registry/--edge-id/--cloud-id` | `--api-server-rpc-port/--api-server-attach`（仅元数据） | B 参数框架为主干，A 的拓扑参数并入（或由 gen_lwd_topology.py 承接） |
| 环境变量（vllm） | `VLLM_LWD_CLOUD_API_KEY`、`VLLM_LWD_TOPOLOGY_CHECK_PORT` + `VLLM_ASCEND_LWD_{DISABLE_DOWN,SKIP_SAMPLE,HOST_STAGE,DEBUG,...}` 系列 | `VLLM_LWD_LOG_LEVEL`；ascend 侧 envs.py 零 LWD 变量（内联读取） | 双方 env 块机械合并全保留；A 的 8+3 个诊断 env 保留为调试开关 |
| ascend_config（va） | +8 行 `payload_scramble` 块（双静态因子 fail-fast） | 无 LWD 字段（`lwd_config` 剔除给核心层），payload 混淆从拓扑 `feature_ctrl` 初始化 | 取 B（拓扑驱动），A 的显式块作为兼容入口评估 |
| 模式开关 | `enabled` + `mode ∈ {head_tail, embedding_only, prefill_only}` + `edge_head_tail_layers` | 存在即启用；scene 四态（single_instance/edge_share/cloud_share/lwd_cluster） | B 语义为底；A 的 head_tail/embedding_only 模式作为扩展保留 |

### 4.6 CI / 文档 / 规范差异

| 项 | A | B | 融合后 |
|---|---|---|---|
| LWD CI | 无 | `lwd_fast/lwd_image/lwd_checklist` workflow + PR/issue 模板 | 取 B，A 的跨进程部署测试补进 CI |
| 设计文档 | `lwd_control_url_multi_value.md`(337 行)、`lwd_host_staging.md` 中文详版(858 行)、`lwd_pre_recv.md` | `lwd_control_plane/data_plane/host_staging/multi_dp.md` + refactor v4/v5 系列 | B 文档体系为主干；A 中文详版改名为附录（同名 `lwd_host_staging.md` 两版描述两套实现，add/add 冲突需处理） |
| 代码规范 | 中文注释、无守卫 | 英文 + SPDX + `lwd_style_guard`/`weave_guard`/`s1s2_refactor_guard` 守卫测试 | 取 B（守卫进 CI 阻止回退） |

## 5. 功能矩阵：融合前差异 × 融合后效果

> ✅=已实现且保留；➕=B 缺、需从 A 移植；🦴=以 B 为骨架；❌=不迁移。

| 功能 | 融合前：A（qxxxw） | 融合前：B（main） | 融合后效果 |
|---|---|---|---|
| LWD prefill_only 边云协同 | ✅（28693 行体系） | ✅（同源移植版） | 单套统一实现（B 骨架），避免双栈并存 |
| **多边多云 mesh（N 边×N 云）** | ✅ ROUTER-ROUTER + registry | ⚠️ topology instance_links（规则 mesh） | ➕ B 拓扑扩展任意 mesh，A 的路由/健康冷却并入 edge 模块 |
| **多实例/多前端 attach** | ✅ 完整（`frontend_attach.py` 1029 行，REGISTER/GRANT 协议） | ⚠️ 仅 `LwdAPIConfig` 元数据预留（102 行） | ➕ A 实现对接 B 的 edge_share 场景配置（B 已留好落点） |
| **云复用（跨边 KV/计费回边）** | ⚠️ 2026-11-02 起改为 nonce/READY/REFILL 预留（与 B 同构）；HTTP/HMAC 链路保留但运行时未接线 | ❌（云内 nonce 预留，无跨边计费） | 🦴 B 账本/状态机 + A 的多云路由维度；计费/HTTP 待定（见 D1 修订） |
| control_url 多值直连 + 死云冷却 | ✅（dev/debug 形态，含 2e2i 示例） | ❌ | ⚠️ 依赖 HTTP 探针；探针链路已死，保留需重新接线（工作量另计） |
| 前缀协调 | ✅ 已迁移为 `LwdPrefixReservations` + nonce/READY/REFILL（与 B 同构，API 命名不同） | ✅ nonce 预留 + refill | 🦴 单账本单状态机（§7.4.1 修订），A 的多云维并入 |
| **结构化输出（LWD prefill_only）** | ✅ **2026-11-02 新增**：wire v5 + 云侧 grammar_init + 边侧 picker 过滤 + 切分脚本带 tokenizer | ❌ 明确拒绝（`ValueError`） | ➕ 按 A 的 5 文件 +95/-24 方案移植（§7.4.6） |
| **权重预读跳过（safetensors skip_name）** | ✅ 新增：进 `get_tensor` 前过滤占位模块 | ❌ | ➕ 可选移植（独立显存/IO 优化，见 §4.2） |
| **UP 水位多云键** | ✅ `dict[(cloud_id, engine_index), int]` | ✅ `dict[dp_idx, int]` | 统一为二元组键（A 为超集，B 的 dp 映射为 (0, dp)） |
| host staging | ✅ v7（文档深、有开关）；2026-11-02 起补齐 `seed_prefix/skip_consumed/take_refill_done` | ✅ 现行实现 + prefix refill + 22 用例 | 🦴 B 版（A 现已基本对齐），A 文档归档 |
| 云内多 DP | ✅（engine 维） | ✅（per-DP 引擎 + DpAccount 账本，更完整） | 🦴 B 形态，A 的 engine 维映射为 DP |
| 云抢占恢复 | 原生抢占 + 草稿 stash | `LwdRefillNotify` 重嵌入 + SKIP 分支 | 🦴 B 协议；A spec 恢复链按调度策略取舍 |
| spec 恢复链（draft stash + accepted persist） | ✅（2026-11-02 修补 remap，新增 `test_lwd_accepted_state_restore`） | 已删除（defer prefill 策略） | ❌/可选：除非保留 A 相位实验臂（见 D4） |
| **PP 并行（LWD）** | ✅ head_tail/embedding_only + PP 对建组 + `_LWD_FULL_HEAD_TAIL` 中性化 | ❌ 已移除（config 显式拒绝 PP>1；组表无 PP；va 零织入） | **删除 A 的 LWD-PP 代码**，只保留 prefill_only；原生 PP 不动（§7.4.7） |
| 早收 early-recv | ✅ per-edge MQ/配额/摘除 | ✅ 单 MQ + done-MQ + fenced harvest | 🦴 B 架构，回加 per-edge 维度 |
| echo 回声握手 + HCCL 拉起校验 | ✅（150 行 + 5 用例） | ❌（仅 warmup） | ➕ 保留（解决多实例拉起期静默 hang，与 B warmup 不冲突） |
| 权重按角色过滤/检查点切分 | ✅（`lwd_weight_filter` + `split_lwd_checkpoint.py` 1016 行） | ❌（边固定零层） | ➕ 全套保留（部署刚需） |
| 部署示例 | ✅ 3 套（2e1c/2e2i/multi_instance） | ❌ | ➕ 保留并适配 B 拓扑 |
| hash MoE 专家 ID（DSv4） | ✅ 0.23 结构实现 | ✅ 0.24 router 结构实现 | 🦴 B 结构 + A 钩子重放 |
| MM 多模态 | ✅ 更深（mm_identity 指纹/pdmix 混批） | ✅ mrope 直传 | 取并集 |
| KV offload p2p/kvcr/tiering | ❌ | ✅（+5834 行 + 10160 行测试） | 🦴 B 全量保留 |
| hisparse / fault_tolerance | ❌ | ✅（2757 / 240 行） | 🦴 B 全量保留 |
| 上游 0.24+ 生态 | ❌（钉 0.23 基线） | ✅（4000+ 提交合流） | 🦴 B 基座 |
| memc（MemCache/MemFabric KV 池） | 基线继承 | 基线 + 上游大幅演进 | 🦴 取 B，零合并动作 |
| LWD CI / 守卫测试 | ❌ | ✅ | 🦴 B 全量保留，补 A 部署测试 |
| 测试体系 | 5733 行（vllm：前缀 nonce/READY/REFILL、多云、probe/registry/前端）+ 2216 行（va） | 3319 + 3821 行（进程内单元/守卫） | 双方互补合并（近零重叠），同名测试取 B 版 + A 独有用例并入 |
| 语言/规范 | 中文注释 | 英文 + SPDX + 守卫 | 🦴 B 规范 |

## 6. 实现冲突预测（试合并实测）

`git merge-tree` 试合并结果（2026-11-02 新提交后重跑）：**61 个冲突文件**（vllm 41 + vllm-ascend 20），其中真实文本冲突 46 个、git 已自动合并 15 个（需语义确认）。较上一轮（56：37+19）新增 5 个：vllm `weight_utils.py`、`renderers/base.py`、`lwd_prefix_cache.py`、`lwd_edge_prefix_control.py`；va `test_lwd_mtp_proposer.py`。**注意：A 新提交把两个前缀文件改成与 B 同名，原本"静默并存"的风险变成了显式冲突——这是好事，强制在合并时收口为单套实现。**

### 6.1 冲突类型分布

| 类型 | vllm | va | 合计 | 占比 | 处理方式 |
|---|---|---|---|---|---|
| 1 增量可合（含自动合并） | 12 | 4 | 16 | 26.2% | 机械合并 + 回归确认 |
| 2 同功能重写 | 10 | 8 | 18 | 29.5% | **架构决策二选一/融合**（含新增的 2 个 prefix add/add + va test） |
| 3 一方重构一方增量 | 8 | 5 | 13 | 21.3% | 以 B 新结构为主干重放 A 语义（含 `weight_utils`/`renderers`） |
| 4 双方同点修改（语义冲突） | 11 | 3 | 14 | 23.0% | 逐冲突块人工裁决 |

### 6.2 隐藏冲突（merge-tree 不报、但会破坏语义，共 5 项）

1. **`vllm/config/lwd.py`（A 单文件）vs `vllm/config/lwd/`（B 包）同名遮蔽**：包优先于模块，A 的 `LwdCoordinationConfig`/`lwd_edge_batch_queue_depth` 将"看起来在、实际不生效"。必须显式删除一个形态。
2. **va worker 目录重命名**：B 把 `lwd_cloud/`→`lwd_cloud_model/`、`lwd_edge_worker.py`→`lwd_edge_model/`；A 的 `platform.py` worker_cls 仍指旧路径 → ImportError。
3. **lwd_control 双栈静默并存**：A 的 `control_communication/`（13 文件）与 B 的 `lwd_comm/`（7 文件）不同名不冲突会同时入库，装配点指向不完整实现 → 必须显式 `git rm` 弃用侧。**2026-11-02 后前缀子模块已改为同名（会显式冲突），但通信层双栈问题仍在。**
4. **va 子包重组的 modify/delete 陷阱**：B 删除了 `experts_selector.py`、`fused_moe_0_23_0.py`、`methods/w8a8_dynamic.py`（挪进子包）；merge 解冲突时直接 `git add` 会把 A 的旧结构文件复活到 B 新目录旁 → 双实现。
5. **A 的 HTTP 前缀探针已成为"启动但无人消费"的死链路（2026-11-02 新增风险）**：`engine/utils.py` 仍会在配置了 `lwd_coordination` 时启动 HTTP 服务与队列桥，但 `LwdCloudControlProcessor` 在新的 A 树中**没有任何实例化点**（运行时与测试均无）→ 命令队列无人消费、探针必超时、边侧 fail-open；同时留下常驻线程与监听端口。合并时必须显式删除这条链路（或重新接线后再启用），不能原样搬进主干。
6. **A 的 LWD-PP 代码回流风险（B 已删除该能力）**：A 的 `head_tail`/`embedding_only` 模式、`LwdParallelConfig.lwd_group_ranks` 的 `pp_groups`、`parallel_state.py` 的 `_LWD_FULL_HEAD_TAIL` 中性化、va/core 的 `enable_lwd` PP 门控，若按"保留 A 能力"惯性整体搬迁，会把 B 已拒绝的 PP 语义重新引入。合并时按 §7.4.7 的删除清单显式清理。

### 6.3 热点文件（完整 61 文件逐项定性已归纳于本节与 §6.4）

| 热点 | 冲突性质 | 裁决要点 |
|---|---|---|
| `vllm/v1/lwd_control/` 整包（含新增 2 个 prefix 同名 add/add） | 同功能两套实现（前缀已同构收敛） | P0：主干二选一 + 删除清单；前缀按 §7.4.1 修订版收口 |
| `vllm/config/vllm.py`（3 块）+ config/lwd 形态 | 双方各自定义 LwdConfig 字段与解析；A 新增结构化输出相关改动 | P0：B 包为准 |
| `vllm/distributed/parallel_state.py`（2 块） | 两套 LWD 世界布局（A instance-major vs B inner-DP grid） | P1：统一 topology 派生 |
| `vllm/engine/arg_utils.py`（5 块） | 双方都实现 LWD master_addr 逻辑 + 各自 CLI 参数集 | P1：master_addr 二选一，A 参数增量重放 |
| `vllm/v1/core/kv_cache_manager.py`（2 块） | 2026-11-02 后两边均为 `lwd_prefix_caching` + 预留快照，仅方法名不同（`get_computed_blocks` vs `snapshot`） | P1：机械收口（§7.4.1 修订） |
| `va worker/model_runner_v1.py`（7 块）、`worker.py`（5 块） | A 基类织入 vs B 上游大重构；A 新增 accepted-state remap 修补 | P1：A 分支下沉子类重放 |
| `va experts_selector/fused_moe_0_23_0/w8a8_dynamic` | modify/delete（B 重组子包） | P2：钩子/日志挪新路径 |
| `tests` 同名文件（vllm `lwd_control` 4 组近似名、va 4 个同名 + 新增 `test_lwd_mtp_proposer`） | 同主题两版 | 取 B 版（超集/新路径），A 独有用例并入 |
| **新增：`vllm/renderers/base.py`、`model_loader/weight_utils.py`** | 结构化输出/权重预读跳过（A 新增）vs B 上游重构 | 按 §7.4.6 移植或取 B，逐块裁决 |

### 6.4 2026-11-02 新提交带来的冲突增量

| 文件 | 类型 | 说明 | 处置 |
|---|---|---|---|
| `vllm/v1/lwd_control/control_cloud_scheduler/lwd_prefix_cache.py` | 2（add/add） | A 新迁移的预留账本，与 B 同名同构（A 302 行 / B 219 行，API 命名不同） | 取 B 为骨架，并入 A 的多云字段（§7.4.1 修订） |
| `vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_prefix_control.py` | 2（add/add） | A 的边侧 nonce/READY 驱动，与 B 同构（A 173 / B 144 行，`configure(hello)` vs `configure(params)`） | 取 B 结构，并入 A 的 HELLO 能力位解析 |
| `vllm/model_executor/model_loader/weight_utils.py` | 3 | A 新增 `skip_name` 预读过滤（进 `get_tensor` 前跳过占位模块），B 上游也改过该文件 | 可选移植 A 的优化，按 B 新签名重放 |
| `vllm/renderers/base.py` | 3 | A 的结构化输出/`tokenizer=None` 适配；B 上游重构 | 随 §7.4.6 决策，默认取 B + 必要 guard |
| `tests/ut/worker/test_lwd_mtp_proposer.py`（va） | 2（add/add） | A 新增，适配其旧目录结构；B 有同名测试但路径/契约更新 | 取 B 版，A 独有用例并入 |

**结论**：增量没有引入新的"架构级互斥点"；相反，前缀协调从"两套范式"收敛为"一套设计 + 多云扩展"，合并难度下降。真正新增的工作量是**结构化输出移植（D6）**与**HTTP 死链路清理**。

## 7. 合并策略

### 7.1 总体策略：B 为骨架，A 为能力包，"重放式移植"而非文本融合

理由：
1. B 基座新（上游 0.24+ 合流）且工程体系完整（CI/守卫/英文规范/评审），A 的 454 提交全部基于旧结构；
2. 两侧 LWD 同源分叉，同名文件已是两套实现，逐文件文本融合会得到互相残缺的双栈；
3. A 的差异化能力（多边多云/多实例/云复用/前端 attach/工具链/结构化输出）边界清晰，可模块化移植；
4. B 已为 A 的部分能力留好接口（`LwdAPIConfig` 元数据 ↔ A 的 `frontend_attach` 实现；`LwdTopology.instance_links` ↔ A 的 mesh）。

反向方案（A 为主干移植 B 的 host-staging/prefix）工作量更大且丢上游基座，不推荐。

### 7.2 两仓合并顺序

**先 vllm 后 vllm-ascend**：va 的 worker 选择、数据面键控、payload 混淆种子全部依赖 vllm 核心层的 LwdConfig 形态，P0 决策（§7.3）必须先在 vllm 仓落定。va 侧相对独立的件（lwd_hash 钩子、诊断日志挪路径）可并行预备。

### 7.3 分阶段执行（P0→P3）

| 阶段 | 内容 | 产出/退出条件 |
|---|---|---|
| **P0 架构决策**（阻塞一切） | ① lwd_control 主干二选一（建议 B）；② vllm/config/lwd 模块形态（建议 B 包 + A 并入 coordination）；③ va worker 目录形态（建议 B 新目录）；④ 前缀单实现收口 + 删除 A 的 HTTP/HMAC 死链路（§7.4.1 修订）；⑤ 结构化输出移植范围确认（§7.4.6）；⑥ LWD-PP 删除清单确认（§7.4.7）。产出**删除清单**（弃用侧全部文件显式 `git rm`）+ **移植清单**（A 能力 → B 落点映射表） | 六份清单评审通过 |
| **P1 语义裁决** | 逐冲突块人工裁决：parallel_state 世界布局、arg_utils master_addr、kv_cache_manager 前缀 API（`snapshot` vs `get_computed_blocks`）、engine core/utils DP rank 与 GPU 分配、va worker/model_runner 织入块（7+5 块）、ascend_config、outputs 水位键 | 每块有裁决记录 |
| **P2 结构重放** | A 语义按 B 新结构落位：vllm `multiproc_executor/serve/qwen3_vl/gpu_model_runner/single_type_kv_cache_manager`；前缀 API 统一 + 多云维度并入；结构化输出（消息/云 grammar_init/边 picker 过滤）；权重预读跳过（可选）；**删除 A 的 LWD-PP 代码**（模式/PP 建组/中性化/门控）；va `fused_moe` 三件（lwd_hash 钩子 3 处 + 13 行日志挪路径）、`platform.py` worker_cls 挂新点 | A 能力包全部落位 |
| **P3 机械收尾** | 16 个类型 1 文件直接合并；env 块双方保留；4 组同名测试改名共存；61 个冲突文件清零 | 全部冲突清零 |

### 7.4 关键决策点（D1–D7 均已给出推荐方案，按影响排序）

| # | 决策 | 建议 |
|---|---|---|
| D1 | **前缀协调如何融合**（2026-11-02 修订：A 已迁移为与 B 同构的 nonce/READY/REFILL） | **单账本 + 单哈希域 + 单协商状态机**：以 B 的 `LwdPrefixReservations` + `LwdEdgePrefixControl` 为唯一实现，并入 A 的多云路由维度（cloud_id/engine、is_refill、refill MQ、UP 水位二元组键）；A 的 HTTP/HMAC 链路已成为无消费者的死链路，合并时删除（未来若需跨网关隐私探针再另立插件）；消息统一升 v5。详见 §7.4.1（修订版） |
| D2 | 数据面世界布局（rank/HCCL 组） | **已决策**：B `LwdParallelGroupsSpec.from_topology` 为唯一布局引擎；把 A 的共享边（edge_share）布局与"每 rank 多连接"能力实现为 spec 扩展（解掉 B 现有两处 `NotImplementedError`）；保留 A 的 pre-init 拓扑一致性守卫（改为比较 topology digest）。详见 §7.4.2 |
| D3 | master_addr 与 CLI 参数体系 | **已决策**：master_addr 取 B（与 A 语义已收敛）；CLI 以 topology YAML 为唯一源，A 的数字参数降级为 `gen_lwd_topology.py` 输入 + 一版兼容 shim，`--role-registry` 一次性导入迁移。详见 §7.4.3 |
| D4 | spec 恢复链去留 | **已决策**：主干丢弃 A 的 draft cache/persist 链（与 B 的 defer-prefill 一致，且 B 已判定其为死代码）；A 在 2026-11-02 修补了 accepted-state remap，若保留 PhaseScheduler 实验臂需一并带入该修补。详见 §7.4.4 |
| D5 | host staging 文档双版本 | **已决策**：B 英文文档为 canonical；A 中文 v7 详版改名归档，其独有根因分析/决策推导并入 canonical 的 Design notes。详见 §7.4.5 |
| **D6** | **结构化输出（LWD prefill_only）如何移植**（2026-11-02 A 新增） | **按 A 的 5 文件 +95/-24 方案移植进 B**：消息加 `structured_outputs`（wire v5）、云侧重建 SamplingParams 并在 promote 前 `grammar_init`、边侧取消拒绝并给 picker 加 grammar-ready 过滤；B 云侧加载完整 checkpoint，无需 A 的切分脚本改动。详见 §7.4.6 |
| **D7** | **LWD 的 PP 并行清理**（B 已移除，合并后不得回流） | **删除 A 的全部 LWD-PP 代码**：`head_tail`/`embedding_only` 模式、PP 对建组、`_LWD_FULL_HEAD_TAIL` 中性化、va/core 各处 `enable_lwd` PP 门控；只保留 prefill_only 语义与权重裁剪工具，原生 PP 不动。详见 §7.4.7 |

### 7.4.1 D1 决策细化：前缀协调融合方案（2026-11-02 修订版）

> **修订说明**：A 在 2026-11-02 并入 `baseline_update1031`（merge `c81803fe6e`），把前缀缓存从"HTTP/SSE 探针 + HMAC 协调摘要"迁移为"控制面 nonce + READY/REFILL + `LwdPrefixReservations`（原生块哈希）"，删除 `lwd_prefix_coordinator.py`，新增与 B 同名的 `lwd_prefix_cache.py` / `lwd_edge_prefix_control.py`；va 侧 host staging 同步补齐 `seed_prefix/skip_consumed/take_refill_done`。因此上一版"双哈希域（native/hmac）+ 双协商策略（in-band/http-probe）"的前提已不存在，以下为修订后的实施依据。

#### （1）融合前：A 侧实现（新提交后）

**端到端流程（与 B 同构）**

1. 云侧握手/HELLO 通告前缀能力：`prefix_caching / hash_block_size`（A 用 `LwdHelloNotify` 扩展字段，B 用 `LwdCloudParamsNotify`，语义等价）。
2. 边侧按同一 hash 算法为请求分配 `prefix_nonce`、计算 `block_hashes`，`LwdRequestNotify` 携带 `block_hashes + prefix_nonce + skip_reading_prefix_cache` 上行。
3. 云侧 `add_request` 路径 `reservations.reserve(request)`：`find_longest_cache_hit` + touch 命中块 + 冻结 groups 快照 + 设 floor（`lwd_resume_cap_tokens`），随后回 `LwdPrefixReadyNotify(request_id, hit_tokens, nonce)`。
4. 边侧 `LwdEdgePrefixControl` 以 nonce 对账；READY 到达前**不入队**（A 新增的 defer-until-READY 修复），error/超时 → 回滚 + abort。
5. 准入走 `KVCacheManager.get_computed_blocks` 的 `lwd_prefix_caching` 分支 → `reservations.get_computed_blocks(request)`（B 同名方法为 `snapshot`），fail-closed，禁止重查。
6. 抢占重准入：按 recv/对齐 clamp + truncate 后 `reserve_hit`；命中 < floor → `LwdRefillNotify(start_row, end_row, nonce)` → 边重嵌入 → refill done 抬闸。
7. host staging：`seed_prefix / skip_consumed / take_refill_done`（A 新提交已补齐，与 B 同语义）。

**A 相对 B 的增量（需要保留的）**

| 增量 | 位置 | 说明 |
|---|---|---|
| 多云路由维度 | `LwdRequestNotify.cloud_id/engine/prefix_hit_tokens`、`LwdC2eNotify.prefix_nonces`、`LwdRangeItem.is_refill` | 支持 N 云 × N DP 的定向回执与按云归账 |
| HELLO 能力位 | `LwdHelloNotify.prefix_caching/hash_block_size` | 边侧逐 engine 严格互校（B 走独立 ParamsNotify） |
| UP 水位二元组键 | `vllm/v1/outputs.py: lwd_up_send_watermarks: dict[(cloud_id, engine_index), int]` | B 为 `dict[dp_idx, int]`；A 是超集 |
| refill done MQ | va `lwd_cloud_worker` / `pre_recv` | 云侧补发完成后通知引擎抬闸 |
| 结构化输出字段 | `LwdRequestNotify.structured_outputs`（wire v5） | 与前缀共用消息层，见 §7.4.6 |
| 新增测试 | `test_lwd_prefix_cache.py` 587 行、`test_lwd_edge_prefix_control.py` 680 行、`test_lwd_multi_dp_prefix.py` 187 行 | 覆盖 nonce/READY/REFILL/多 DP |

**A 侧的遗留死链路（合并时必须处置）**

- `lwd_cloud_control.py`（FastAPI/Bridge）、`lwd_edge_client.py`（SSE 客户端）、`lwd_prefix.py`（HMAC manifest）仍在树内；`engine/utils.py` 在配置 `lwd_coordination` 时仍会启动 HTTP 服务与队列桥。
- 但 `LwdCloudControlProcessor` **在全仓没有任何实例化点**（运行时与测试均无）→ 命令队列无人消费 → 探针必超时 → 边侧 fail-open；同时留下常驻线程与监听端口。
- 结论：这条链路已不是有效能力，应作为**删除项**而不是迁移项。若未来确实需要"跨网关/租户隐私探针"，应作为独立插件重新立项（其 HMAC/manifest 设计仍有参考价值）。

#### （2）融合前：B 侧实现

B 为当前主干：`LwdPrefixReservations.reserve/reserve_hit/snapshot/truncate/release` + `LwdEdgePrefixControl`（nonce/Timer/READY）+ `LwdRefillNotify` + `kv_cache_manager` 5 处钩子 + va `seed_prefix/skip_consumed/take_refill_done`；无 HTTP/租户链；测试最厚（host staging 22 组 + refill）。

#### （3）差异与冲突（修订后）

| 维度 | A（新） | B | 裁决 |
|---|---|---|---|
| 预留账本 | `LwdPrefixReservations`（302 行，`get_computed_blocks/truncate_groups/_pin_and_record`） | `LwdPrefixReservations`（219 行，`snapshot/truncate/_touch_and_snapshot`） | **同一设计、方法名不同**：保留 B 方法名，A 的调用点重定向（或加别名兼容） |
| 边侧驱动 | `LwdEdgePrefixControl`（173 行，`configure(hello)`） | `LwdEdgePrefixControl`（144 行，`configure(params)`） | 取 B 结构，并入 A 的 HELLO 能力位解析 |
| 消息 | wire v5：prefix + `structured_outputs` + 多云维 | v3：prefix/refill | 升 v5（只追加默认值字段） |
| 准入 | `lwd_prefix_caching` + `get_computed_blocks` | `lwd_prefix_caching` + `snapshot` | 机械收口 |
| UP 水位 | `dict[(cloud_id, engine)]` | `dict[dp_idx]` | 统一二元组键 |
| HTTP/HMAC | 死链路（无消费者） | 无 | **删除**（不迁移） |
| 计费/usage | 依赖死链路 | 无 | 删除；若业务需要，另立需求 |

#### （4）最优方案：单账本 + 单哈希域 + 单协商状态机

1. **单账本**：B `LwdPrefixReservations` 为唯一实现；把 A 的 `get_computed_blocks` 语义并入 B 的 `snapshot`（或加别名），`truncate_groups` 并入 `truncate`；A 的调用点（kv_cache_manager、cloud engine、scheduler）全部改指向 B API。
2. **单哈希域**：统一原生块哈希（云可自算生成段），**取消**上一版的 hmac 域设计；删除 `LwdPrefixHasher`/`LwdPrefixManifest`/`tenant_key_file`/`enforce_mm_abi_match` 等只服务 HTTP 探针的配置与代码。
3. **单状态机**：统一 B 的 nonce/READY/REFILL 驱动；A 的 `configure(hello)` 能力位合并进 B 的 `LwdCloudParamsNotify`/HELLO 解析；保留 A 的"READY 前不入队"修复与 refill gate。
4. **多云扩展（A 增量）**：保留 `cloud_id/engine` 路由维度、`is_refill`、`prefix_nonces`、refill done MQ、UP 水位二元组键；在 B 的 per-DP 模型上把 `dp_idx` 泛化为 `(cloud_id, engine_index)`（单云时退化为 `(0, dp)`）。
5. **消息版本**：合并为 v5 —— B 的 8 消息 + A 的 `LwdPrefixReadyNotify/LwdRefillNotify`（A 已有）+ `structured_outputs` + 多云字段；按"追加带默认值字段、双向容忍未知字段"策略。
6. **HTTP 死链路**：`engine/utils.py` 删除 `start_lwd_cloud_control_server` 调用；删除 `lwd_cloud_control.py`/`lwd_edge_client.py`/`lwd_prefix.py` 与 `lwd_coordination` 配置（或保留为 `legacy/` 目录、默认不导入）；A 的 `lwd_edge_scheduler` 中 probe 分支（`_lwd_probe_*`、`negotiate`、`placement_only`）一并删除。
7. **测试**：保留 B 的 host staging/refill 用例；迁入 A 的 `test_lwd_prefix_cache.py`/`test_lwd_edge_prefix_control.py`/`test_lwd_multi_dp_prefix.py`（按 B API 改名）；删除 A 的 HTTP/probe 测试。

| 对象 | 处置 |
|---|---|
| B `lwd_prefix_cache.py` / `lwd_edge_prefix_control.py` / `kv_cache_manager.py` 钩子 | **保留为主干**；并入 A 的多云字段与 READY-前不入队语义 |
| A `lwd_prefix_cache.py` / `lwd_edge_prefix_control.py` | **取 B 结构**，A 的差异逐项并入；不保留第二份 |
| A `lwd_cloud_control.py` / `lwd_edge_client.py` / `lwd_prefix.py` / `lwd_coordination.py` | **删除**（死链路）；测试同步删除 |
| A `lwd_notify.py` 的 prefix/structured 字段 | **并入** B `lwd_messages.py`（v5） |
| A `outputs.py` 二元组水位 | **并入** B `outputs.py` |
| A va `lwd_cloud_worker/lwd_cloud_model_runner/lwd_host_stage/pre_recv` 的 refill/多云端点 | **并入** B 的 `lwd_cloud_model/` 对应件 |

**验收**：合并后全仓只剩一套预留实现（无 `lwd_prefix_coordinator`、无第二个 `LwdPrefixReservations`）；HTTP 服务不再启动（或明确开关且默认关闭）；A 的 3 个新前缀测试与 B 的 host staging/refill 用例同时通过；wire v5 新旧端互通（新增字段有默认值）；错配/超时路径与 B 原语义一致。

#### （5）一句话结论

A 的新提交把 D1 从"两套范式择一"变成了"同构实现收口"：**取 B 的账本/状态机为唯一实现，并入 A 的多云路由与 refill 增量，删除 A 已无消费者的 HTTP/HMAC 链路，消息统一升 v5**。这比上一版方案更简单、风险更低。

### 7.4.2 D2 决策细化：数据面世界布局（rank 布局与 HCCL 组）

#### （1）融合前：A 侧

- **CLI 数字推导**（`vllm/config/parallel.py` `LwdParallelConfig`）：
  - edge ranks 全在前 `[0, E)`，dp j 占 `[j*E_per, (j+1)*E_per)`；
  - **共享边**（`E==1 && D>1`）：唯一边卡被所有云 engine 复用，不按 dp 切分；
  - 云 **instance-major** 续编：实例 i、dp j 首 rank = `E + i*D*C + j*C`；
  - `lwd_group_ranks(d)` 纯函数产出 TP/PP/DCP/PCP/DP/EP 全表，所有 rank 算出同一张表（`new_group` 全序）；
  - PP 对 = (首边首卡, 首云首卡)，prefill_only 下由 `_LWD_FULL_HEAD_TAIL` 中性化；TP 按 (cloud, engine) 细分；EP 跨 D 不跨 N。
- **Registry 模式**：`edge_ranks_layout / cloud_ranks_layout / cloud_dp_layout`（YAML 声明），`instance_ranks / instance_engine_ranks / cloud_engine_rank_base` 查询；`parallel_state.py` 直接按布局建组，绕过原生均匀 rank 网格。
- **Pre-init 一致性守卫（A 独有）**：`_validate_lwd_topology_pre_init` 在 `init_process_group` 之前用独立 TCPStore（缺省 `master_port+2`，可被 `VLLM_LWD_TOPOLOGY_CHECK_PORT` 覆盖，超时 600s）交换 axes `{world_size, cloud_dp_size, instance_parallel_size, edge_npu_count, cloud_npu_count}`，错配即 fail-fast；init 后另有 all-gather 复核 `_validate_lwd_topology_consistency`。
- **va 数据面**：`distributed/lwd_wire.py` 689 行，`_CliInstanceLayout` 合成 registry，`_load_registry` 三级回退（registry YAML > CLI > PP），`init_lwd_duplex_channels` 按 `(edge, cloud, engine, channel)` 全 mesh 建组 + echo 回声握手 + watchdog（`VLLM_ASCEND_LWD_HCCL_HANDSHAKE_TIMEOUT_S`）。

#### （2）融合前：B 侧

- `LwdParallelGroupsSpec.from_topology`：edge 每实例一份 allocation（`edge.dp[0].ranks`），云每 DP 一份；`tp_groups` 排序后建组；`dp_groups` 取各 DP 同位置 rank；`ep_groups` 每实例（边=自身 rank，云=全部 DP rank）；`channel_endpoints` 每 DP 连接一对 (edge 首 rank, cloud leader rank)；PP/DCP/PCP 用 `singleton_groups`；world_size 取 `deployment.hccl_world_size`。
- `lwd_check_parallel_config`：拒绝 PP>1 / PCP>1 / DCP>1 / elastic EP / Engram；校验 TP == 每 DP rank 数；config 期调用 `from_topology` fail-fast。
- **两处硬限制（代码内 TODO）**：非 `single_instance` 场景下 ① 边实例共享 ranks → `NotImplementedError`；② 一个 rank 持多条 DP 连接 → `NotImplementedError`。scene 四态：`single_instance`(1,1)、`edge_share`（多边实例共享同一 edge ranks/addr）、`cloud_share`（多边一云）、`lwd_cluster`（多边多云）。
- `tools/gen_lwd_topology.py` 已能生成四场景 YAML（`--edge-machines / --cloud-machines / --dp / --cloud-cards`）。
- **va 数据面**：`lwd_data_plane/lwd_channels.py` 从 spec 取 `LwdChannelEndpoints`，每连接建 UP/DOWN 两个 group + 每通道一条 NPU Stream；`lwd_warm_up_channels` 固定全局顺序预热；无 echo 握手。

#### （3）差异与冲突

| 能力/维度 | A | B | 影响 |
|---|---|---|---|
| 共享边卡（E=1 复用、多实例分时） | 支持 | 模型支持（edge_share）但 spec 未实现 | A 的共享边多实例在 B 会抛 `NotImplementedError` |
| 每 rank 多连接（mesh） | 支持 | 仅 `single_instance` 允许 | 同上，2E×2I×2C 直接被拒 |
| 拓扑来源 | registry YAML / CLI 数字 / PP 三级 | topology YAML 单源 | 迁移成本集中在 D3 |
| 运行时一致性守卫 | pre-init TCPStore + post-init all-gather | 仅 config 期 digest | B 缺"多命令错配 → 挂死"的早停 |
| PP | 支持（prefill_only 中性化） | 禁止 PP>1 | 取舍：A 下 PP 无真实收益 |
| va 通道建组 | `(edge,cloud,engine)` mesh + echo | `dp_idx` 单连接 + warmup | 结构性差异，需按连接对象扩展 |
| 冲突文件 | `parallel_state.py`（2 块）、`engine/utils.py`（2 块）、va `platform/worker`、`lwd_wire` vs `lwd_channels` | 同左 | 类型 3/4 混合 |

#### （4）优劣

| A 的优点 | A 的缺点 |
|---|---|
| 支持共享边多实例（省卡）与任意 mesh | 三套拓扑源口径易漂移（A 自己的注释都在提示 `cloud_npu_count` 口径变更） |
| pre-init 早停，避免错配挂死 | CLI 公式与代码强耦合，难维护 |
| echo 握手可定位拉起期错配 | 引入第二套握手协议，与 B 的版本化握手重复 |

| B 的优点 | B 的缺点 |
|---|---|
| 单源、pydantic forbid + digest、组表全序可测 | `edge_share` 与多连接未实现（恰好是 A 的核心场景） |
| 与上游/CI 一致，零织入 | 缺运行时早停；禁止 PP/PCP/DCP |

#### （5）最优方案：B spec 为唯一布局引擎，A 的布局能力做 spec 扩展

1. **唯一布局引擎** = B `LwdParallelGroupsSpec.from_topology`；topology YAML 为唯一事实源；A 的 `lwd_group_ranks` / registry 布局只在迁移工具中存在（D3）。
2. **解掉 B 的两处限制**（扩展 spec，而非另起炉灶）：
   - 限制①：`edge_share` 时把多个 edge 实例的 allocation 归并为一份（B 的 `_validate_ranks` 已允许共享，只差 `from_topology` 的合并与 endpoint 去重）；
   - 限制②：`LwdChannelEndpoints` 增加 `edge_id/cloud_id` 字段，允许同一 rank 持多连接；va `lwd_channels.py` 按连接对象建组（A 的 `lwd_wire.py` 已验证可行）。
3. **保留 A 的 pre-init 守卫**，但比较对象换成 `topology.digest (sha256[:16]) + role + instance_id`（比 CLI axes 更准，且与 B 的配置模型一致）；`VLLM_LWD_TOPOLOGY_CHECK_PORT` 与 600s 超时保留。
4. **PP/PCP/DCP 沿用 B 禁止**（A 的 prefill_only 已把 PP 中性化，无真实收益）；`head_tail` 模式若未来需要 PP，再按 A 的 PP 对规则做后续扩展。
5. **echo 握手不引入第二套状态机**：保留 B 的 Hello/Accept/Reject + `handshake_timeout_s`，把 A 的逐字节 echo 自检作为建组后的可选诊断（env 开关）。

| 对象 | 处置 |
|---|---|
| B `vllm/distributed/lwd/lwd_parallel_groups.py`、`lwd_check_parallel_config` | **保留为主干**；扩展 `LwdChannelEndpoints` 与 `from_topology` 的 edge_share 合并逻辑 |
| B va `lwd_data_plane/lwd_channels.py`、`lwd_parallel_groups.py` | **保留**；按连接对象建组 |
| A `parallel_state.py` 的 `LwdParallelGroups` 内联建组（+518/-29 的 LWD 块） | **删除**，改调 B spec；保留 pre-init guard 函数并改写比较对象 |
| A `LwdParallelConfig` 的布局方法（`lwd_group_ranks/cloud_engine_rank_base/instance_engine_ranks/...`） | **删除**，语义迁入 `gen_lwd_topology.py` 与 spec；`ParallelConfig.lwd_config` 可留 deprecated 只读投影 |
| A va `lwd_wire.py` 的布局合成/registry 回退 | **删除**；echo 自检作为可选诊断保留 |
| A 的 `_lwd_topology_check_port` / `_validate_lwd_topology_pre_init` | **保留**（改写为 digest 比较） |

**验收**：A 的 3 套示例拓扑经 `gen_lwd_topology.py` 生成后 `from_topology` 不抛错；组表与 A `lwd_group_ranks` 逐项一致（用 A 的单测表做 golden）；故意错配时 <600s fail-fast 而非挂死。

### 7.4.3 D3 决策细化：master_addr 与 CLI 参数体系

#### （1）融合前：A 侧

- CLI：`--edge-npu-count`、`--cloud-npu-count`（两种口径：跨实例总量 / per-engine，靠 `--cloud-dp-size` 显式与否归一）、`--cloud-dp-size`（None 哨兵）、`--instance-parallel-size`（≤16）、`--instance-parallel-start-rank`、`--role-registry`、`--edge-id`、`--cloud-id`；`EngineArgs` 归一化后塞 `LwdParallelConfig`；`VllmConfig.__post_init__` 解析 `additional_config["lwd_config"]`。
- master_addr：LWD 下 `data_parallel_address = ParallelConfig.data_parallel_master_ip`（不回退 `master_addr`）——否则 engine 级 DP 的 gloo store 会建到边侧 IP，云 engine 间永连不通。

#### （2）融合前：B 侧

- `resolve_lwd_config(additional_config, api)` → `LwdConfig(path, role, instance_id, topology, coordination, api)`；topology 从 YAML 读；顶层 `coordination` 块显式报错退役（内嵌每实例）；`LwdAPIConfig` 只有 `--api-server-rpc-port / --api-server-attach` 元数据（要求 `scene=edge_share`）。
- master_addr：同样取 `ParallelConfig.data_parallel_master_ip`，并打印日志提示跨节点云 DP 需显式 `--data-parallel-address`。

#### （3）差异与冲突

| 项 | A | B | 结论 |
|---|---|---|---|
| master_addr 语义 | 边侧 rendezvous，DP store 用 loopback 类默认 | 同左（已收敛） | **取 B**（带日志，判定条件与 B 的"key 存在即启用"一致） |
| 拓扑输入 | CLI 数字公式 | topology YAML + `gen_lwd_topology.py` | **B 为唯一入口**，A 参数降级为工具输入 |
| 参数集 | edge/cloud npu count、instance parallel、cloud dp、registry | 仅 api 元数据 | A 的数字参数做一版兼容 shim 后移除 |
| 退役字段处理 | 允许 `lwd_coordination` 顶层块 | 显式报错并给迁移指引 | 取 B |

#### （4）优劣

| A 的优点 | A 的缺点 |
|---|---|
| 一条命令推导全拓扑，部署脚本短 | `--cloud-npu-count` 双口径（A 注释自己都在提示升级陷阱） |
| 多实例按 start-rank 区分简单 | 公式与并行组实现强耦合，难以表达任意 mesh |

| B 的优点 | B 的缺点 |
|---|---|
| 声明式可审计（YAML + forbid + digest） | 部署前必须先生成/分发 YAML |
| 与拓扑校验、生成工具、文档同一事实源 | 无 A 式短命令 |

#### （5）最优方案

1. **master_addr**：取 B 的实现，删除 A 的重复分支（两版语义已收敛，保留一份即可）。
2. **CLI**：B 的 `lwd_config.path` 为唯一运行时入口；A 的数字参数只作为 `gen_lwd_topology.py` 的输入（工具层）。运行时保留一版**兼容 shim**：
   - 检测到 A 式参数 → 内存调用 `build_topology(scene=..., ...)` 生成 `LwdTopology.from_dict` + deprecation warning；一个发布周期后移除；
   - `--cloud-npu-count` 双口径在 shim 内按 `--cloud-dp-size` 是否显式给出来归一（复用 A 的现有判定，仅此一处兼容逻辑）；
   - `--role-registry` → 新增 `tools/import_lwd_registry.py`（registry YAML → topology YAML），一次性迁移；运行时不再读 registry；
   - `--edge-id / --cloud-id` → `lwd_config.instance_id`；`--instance-parallel-start-rank` → cloud instance id。
3. **前端 attach 参数不属本决策**：B 的 `LwdAPIConfig` 接口保留，A 的 `frontend_attach` 实现按能力包移植（见 §5 多实例行）。
4. **代码级**：删除 A `EngineArgs` 的 LWD 数字字段与归一化块（保留 shim）、删除 `LwdParallelConfig`（与 D2 同步）、保留 B `resolve_lwd_config/LwdConfig/LwdTopology/gen_lwd_topology.py`、保留 B 的 master_addr 分支并删除 A 的。

**验收**：A 的 3 套启动脚本（2e1c/2e2i/multi_instance）改为 topology YAML 后 world_size/rank/端口/连接数逐项等价；错配在 config 期 fail-fast；shim 路径打 deprecation warning 且行为与迁移后一致。

### 7.4.4 D4 决策细化：spec 恢复链去留

#### （1）融合前：A 侧

- `LwdDraftCache`（53 行，va `worker/lwd_cloud/lwd_draft_cache.py`）：请求短暂离批时克隆 drafts 设备快照（请求对象身份失效、单次 clone 避免逐请求 kernel），供相位切换/抢占恢复。
- va 基类 `model_runner_v1.py` 织入 `_lwd_draft_stash`（+89 行）与 `_lwd_spec_persist_enabled`（接受数持久化）；云端 runner `_lwd_restore_drafts` 恢复。
- **唯一消费者**：`LwdCloudPhaseScheduler`（`scheduler_name=prefill_first/decode_first`，可选实验臂）；A 默认是 `LwdCloudMixedScheduler`（混排，不抢占 decode）。
- **2026-11-02 更新**：A 修补了 async 调度下 `prev_req_id_to_index` 为空时的 accepted-state remap（`_prepare_inputs` 分支），并新增 `tests/ut/worker/test_lwd_accepted_state_restore.py`（87 行，覆盖 prefix-hit/returning/unknown/invalid 行不继承 row 0、Mamba prefix 恢复用 `accepted-1` 偏移）。即 A 的 persist 链仍在活跃维护，不是遗留死代码。

#### （2）融合前：B 侧

- 已明确删除。`test_lwd_base_weave_guard.py` 注释原文："The spec recovery chain (draft stash + accepted persist) was dead code in pdmix deployments (LwdCloudScheduler defers prefill instead of evicting decode; the PhaseScheduler experiment arm is the only consumer and is not used in production)."
- 保留 `_lwd_rewind_preempted_requests` 的 REWIND/SKIP（抢占后重算 / 前缀命中前跳），无 draft stash；基类白名单为空集。

#### （3）优劣

| 保留 A 链 | 丢弃 A 链 |
|---|---|
| 若启用 PhaseScheduler 抢占 decode，可保住 MTP draft 与接受数，避免恢复时重算 | 与 B 的调度策略（defer prefill，不逐出 decode）一致；少一条维护链 |
| — | 满足 B 的零织入纪律（weave-guard 白名单保持空集） |
| — | B 已判定其为死代码，且只有非生产实验臂消费 |

#### （4）最优方案

1. **主干丢弃 A 的 draft cache / persist 链**：删除 `lwd_draft_cache.py`、`_lwd_restore_drafts`、va 基类的 `_lwd_draft_stash` / `_lwd_spec_persist_enabled` 织入，回归到 B 的 REWIND/SKIP 方案。
2. **若确需 PhaseScheduler 实验臂**：以**可选子类**承载——`LwdCloudPhaseScheduler(LwdCloudScheduler)` + `LwdDraftCache` 只在该子树内使用，不织入基类；配置 `scheduler_name` 显式开启，默认关闭；测试独立，不纳入主干 CI 必过集。
3. **依赖检查**：`_lwd_spec_persist_enabled` 兼管 async 调度下 `prev_positions==-1` 的恢复语义，删除后必须回归 B 的 spec-decode 用例；A 的 rewind/SKIP 相关用例（与相位无关）保留。
4. **验收**：weave-guard `ALLOWED` 保持空集；spec-decode e2e 与 B 基线一致；A 的 `test_lwd_draft_cache` 仅在可选子树中保留；若保留 PhaseScheduler，必须一并带入 2026-11-02 的 accepted-state remap 修补（否则 async 空映射场景会回退）。

### 7.4.5 D5 决策细化：host staging 文档双版本

#### （1）两版现状

| 项 | A（858 行，中文 v7） | B（151 行，英文现状） |
|---|---|---|
| 定位 | 设计论证 + 决策推导 | 与实现/测试对应的现状说明 |
| 章节 | 问题（批把存储与调度捆死）、现状条件、关键决策（边侧组批传/云侧按请求拆）、按请求拆分存储（每请求 pinned v6 反转、存储单元、载荷布局、host 占用、设备乒乓槽）、按请求粒度调度（notify 拆请求、已收 token 上报、下发规则取小、非侵入准则、budget/KV 口径、取数指针、删除项）、生命周期与释放 | Storage and watermarks、Single-path arrival flow（hint/arrival contract）、D2H/H2D ordering、Minimal cloud preemption and recompute、Prefix caching and refill（nonce READY/fail-closed、H/H'/S/refill dispatch、prompt-only/MM guards）、completion/abort/tombstones、Full-prompt M-RoPE、DSv4 tail、Implementation map |
| 与代码一致性 | 描述 v7 设计，参数可能与 B 实现漂移 | 与代码/CI/测试一致（常量、消息、refill 语义） |
| 独有价值 | 根因分析、v5→v6 决策反转理由、下发规则推导、非侵入准则 | 实现映射、refill/prefix 语义、DSv4 tail |

#### （2）最优方案

1. **canonical = B 英文文档**（`docs/design/lwd_host_staging.md`），与实现同步维护。
2. **A 文档改名归档**：`docs/design/lwd_host_staging_design_v7_zh.md`，头部注明"历史设计档案，实现以 `lwd_host_staging.md` 为准；参数为 v7 历史值"。
3. **把 A 独有且仍有效的内容并入 canonical 的 `## Design notes`**：
   - 批把存储/调度捆死的根因与 head-of-line 分析；
   - 每请求 pinned 的 v5→v6 决策反转理由；
   - "按已收多少与预算取小"的下发规则推导；
   - 非侵入式实现准则（闸门落点）；
   - 取数指针与原生水位一致性；
   - 已删除项清单（批队首语义）。
4. **代码合并（P2）后**：canonical 补上合并后的最终语义（staging + prefix/refill 合流），A 档案冻结不再更新。
5. **合并提交处理**：两文件按"canonical + 档案"分别落盘，避免 git add/add 自动合并成混合体。
6. **验收**：canonical 引用的常量（`LWD_MAX_PENDING_CHUNKS=10`、`_LWD_EARLY_RECV_MAX_INFLIGHT=2`、`_LWD_HOST_STAGE_SLOTS=2`）与代码一致；A 档案中与实现不符的参数标注"v7 历史值"。

### 7.4.6 D6 决策细化：结构化输出（LWD prefill_only）移植（2026-11-02 A 新增）

#### （1）A 侧实现

- 背景：A 基线曾强制云侧 `skip_tokenizer_init=True`，导致 `StructuredOutputManager` 无 tokenizer，结构化输出在 LWD 下不可用。
- 恢复内容（5 文件 +95/-24，另有 picker 过滤）：
  1. `vllm/config/vllm.py`：删云侧 `skip_tokenizer_init=True` 强制；`mm_encoder_remote / skip_mm_profiling` 保留（多模态 processor 仍不建）。
  2. `scripts/lwd/split_lwd_checkpoint.py`：云侧切片补 tokenizer 条目（`tokenizer.json / tokenizer_config.json / special_tokens_map.json / added_tokens.json / vocab.json / merges.txt / vocab.* / tokenizer.model`）。
  3. `lwd_notify.py`：`LWD_WIRE_VERSION 4→5`；`LwdRequestNotify.structured_outputs: str`（StructuredOutputsParams JSON，含 `_backend`）。
  4. 边侧 `lwd_edge_scheduler.py`：删除结构化输出拒绝；组包时序列化 `sp.structured_outputs`；picker 加 `_lwd_grammar_ready` 过滤（grammar 编译中的请求不在 picker 选中集，否则步后 chunk 核验失守 → 引擎崩溃）。
  5. 云侧 `lwd_cloud_engine.py`：`_lwd_decode_structured_outputs()`；`_lwd_build_request` 重建 `SamplingParams(structured_outputs=...)`；`_lwd_promote` 在 ADD 前显式 `structured_output_manager.grammar_init(request)`（promote 绕过 IO 线程的自动触发点）。
- 设计文档：`docs/design/lwd_structured_output.md`（中文，含 5 文件明细、验证与部署约束）。

#### （2）B 侧现状

- B **明确拒绝**：`lwd_edge_scheduler.py:972` `if request.use_structured_output: raise ValueError("[LWD] prefill-only mode does not support structured output")`。
- 原生结构化输出基础设施在 B 已存在（`config/structured_outputs.py`、`EngineCore.step` 的 `get_grammar_bitmask`、ascend sampler 的 `apply_grammar_bitmask`），缺的只是 LWD 跨进程链路。

#### （3）移植方案（按 A 的方案落到 B 的结构）

| 步骤 | 落点（B） | 说明 |
|---|---|---|
| 1 | `lwd_messages.py` | 加 `structured_outputs: str = ""`，`LWD_MESSAGE_VERSION` 3→5（与 §7.4.1 的 v5 合并为一次升版） |
| 2 | B 边调度器 `_lwd_validate_request` | 删除结构化输出拒绝；其余校验保留 |
| 3 | B 边 `notify_request` | 序列化 `sp.structured_outputs`（`dataclasses.asdict` 含 `_backend` + `json.dumps`） |
| 4 | B 边 picker | 加 grammar-ready 过滤（A 的 `_lwd_grammar_ready`，只读解析 future、不迁状态） |
| 5 | B 云 `_lwd_build_request` | 反序列化并构造 `SamplingParams(structured_outputs=...)` |
| 6 | B 云 `_lwd_promote` | ADD 前 `structured_output_manager.grammar_init(request)` |
| 7 | B `config/vllm.py` / renderers | 确认 B 云侧未强制 `skip_tokenizer_init`（当前 LWD 路径未见强制）；若部署用 tokens-only/裁剪 checkpoint，需保证 tokenizer 文件存在，否则结构化请求应边侧 fail-fast |
| 8 | 测试 | 迁 A 的 wire round-trip 用例 + 端到端 `json_schema / guided_json / guided_regex` + 混批回归 + 多 DP 非 0 engine |

- **与 A 的差异**：A 的切分脚本改动在 B 不需要（B 云侧 `make_layers` 边零层、云加载完整 checkpoint，tokenizer 天然存在）；A 的 `renderers/base.py` tokenizer-None guard 需按 B 上游 renderer 重构后的位置重新评估（可能已被上游覆盖）。

#### （4）验收

- 非结构化请求零路径变化（`has_structured_output_requests` 早退、bitmask None 跳过、picker 状态比较）；
- 结构化请求：边侧前端校验/编译 → 云侧 bitmask 生效 → 输出过 schema；混批不崩；多 DP 非 0 engine 正常；
- 新旧端混部：v5 新端收旧端（缺字段 → 默认空串 = 普通请求）；旧端收新端（msgspec 容忍未知字段）。

#### （5）风险

- 云侧多加载 tokenizer：启动秒级、CPU 内存 +几十 MB（NPU HBM 零变化）；
- picker 过滤若遗漏 → "picker 选中集与原生复现不一致" → 引擎崩溃（A 已给出根因与修复，必须原样移植）；
- 若 B 部署沿用 tokens-only/裁剪 checkpoint，需显式保证 tokenizer 文件存在，否则结构化请求应在边侧 fail-fast，而不是云侧空 tokenizer 崩。

### 7.4.7 D7 决策细化：LWD 的 PP 并行清理（B 已移除，合并后不得回流）

#### （1）B 主干现状：LWD 不支持 PP，且已显式拒绝

| 证据 | 位置 | 内容 |
|---|---|---|
| config 期 fail-fast | `vllm/distributed/lwd/lwd_parallel_groups.py:190-210` | `unsupported` 列表第一项 `("pipeline parallelism", pipeline_parallel_size > 1)` → `NotImplementedError` |
| 组表无 PP | 同文件 `LwdParallelGroupsSpec` | 只产出 `tp_groups/dp_groups/ep_groups/channel_endpoints`；引擎用 `singleton_groups` 建 PP/DCP/PCP |
| 配置无模式 | `vllm/config/lwd/topology.py` | `LwdDeployment.mode` 固定为 0；无 `head_tail`/`embedding_only` |
| va 零织入 | `tests/ut/worker/test_lwd_base_weave_guard.py` | `worker.py`/`model_runner_v1.py` 的 LWD 白名单为空集 |
| 无 PP 豁免 | `vllm_ascend/platform.py` | 不存在 `_validate_pd_pp_mtp_config` 的 LWD 分支 |

**注意**：B 仍保留**原生（非 LWD）PP**：`scheduler.use_pp`、`gpu_model_runner` 的 PP broadcast、`multiproc_executor` 的 PP 路径、va 的 `_pp_receive/_pp_broadcast`、`_validate_pd_pp_mtp_config` 本体。清理只针对 LWD 分支，原生 PP 一行不动。

#### （2）A 侧需要删除的 LWD-PP 代码清单

| # | 文件 | 删除内容 | 理由 |
|---|---|---|---|
| 1 | `vllm/config/lwd.py` | `_VALID_LWD_MODES` 中的 `head_tail`/`embedding_only`；`mode` 与 `edge_head_tail_layers` 字段、`validate` 的 head/tail 分支、`validate_num_hidden_layers`、`local_layer_indices` 的 head_tail 分支、`skips_module` 的 `mode != "prefill_only"` 分支、`__repr__` 的 mode 字段 | B 只有 prefill_only 语义（边零层、云全层）；head_tail/embedding_only 依赖 PP 对 |
| 2 | `vllm/config/parallel.py` | `lwd_group_ranks` 的 `pp_groups` 构造块（`pp_paired`/`pp_edge_first`/`pp_cloud_first`）与返回表的 `"pp"` 键 | B 组表无 PP，PP 用 singleton |
| 3 | `vllm/distributed/parallel_state.py` | `_LWD_FULL_HEAD_TAIL` 全局、`_is_lwd_full_head_tail()`、`is_first_rank`/`is_last_rank` 的两处 PP 中性化、registry 分支的 PP 对建组（`pp_edge_first/pp_cloud_first/pp_groups/init_model_parallel_group("pp")`）、`_LWD_FULL_HEAD_TAIL = True` 赋值 | 同上；B 的 singleton PP 天然满足 first=last |
| 4 | `vllm/v1/worker/gpu_model_runner.py` | async 采样 PP broadcast 上的 `and not self.parallel_config.lwd_config.enable_lwd` 从句（约 4627） | PP>1 已被 config 拒绝，该从句恒真，属冗余 |
| 5 | `vllm_ascend/platform.py` | `_validate_pd_pp_mtp_config` 开头的 LWD `mode=="prefill_only"` 提前 return | PP>1 已拒绝；B 无该分支，合并时不要引入 |
| 6 | `vllm_ascend/spec_decode/llm_base_proposer.py` | `lwd_prefill_only` 对 PP 共享 embedding 守卫的豁免（约 426-430） | LWD 下 `get_pp_group().world_size` 恒为 1，豁免无意义 |
| 7 | `vllm_ascend/worker/model_runner_v1.py` | `_pp_receive_prev_sampled_token_ids_to_input_batch` / `_pp_broadcast_prev_sampled_token_ids` 调用上的 `and not ...enable_lwd` 从句（约 2395/2567） | 同 #4；且 B 基类零织入，合并时不要把这些从句带回去 |
| 8 | `vllm_ascend/examples/*/start_*.sh` | `"mode":"prefill_only"` 与 `"edge_head_tail_layers":[0,0]` 字段 | D3 的 topology/config 迁移后由新版配置替代 |
| 9 | `vllm/v1/lwd_control/__init__.py` / `serve.py` | 若有 PP/head_tail 相关校验（当前 grep 未见，合并时复核） | 防止残留 |

**保留项（不是 PP 冗余）**：
- `LwdConfig.skips_module` 的 **prefill_only** 分支（边/云各自不建无用模块）——它是权重裁剪能力，与 PP 无关；
- `lwd_weight_filter` / `split_lwd_checkpoint`（部署工具）；
- B 原生 PP 的全部代码与测试。

#### （3）验证与验收

1. **静态零残留**：在合并后的 `vllm/` 与 `vllm_ascend/` 中执行 `grep -rn "head_tail\|embedding_only\|_LWD_FULL_HEAD_TAIL\|pp_edge_first\|pp_cloud_first"`，LWD 路径命中为 0；`grep -rn "lwd" vllm_ascend/worker/worker.py vllm_ascend/worker/model_runner_v1.py` 仍为 0（weave-guard 白名单为空）。
2. **拒绝语义**：LWD 部署传 `--pipeline-parallel-size 2` 必须在 config 期 fail-fast（B 现有报错），而不是静默启动。
3. **原生 PP 不回退**：非 LWD 的 PP e2e/单测（`tests/distributed`、va PP 用例）全绿。
4. **prefill_only 功能不回退**：A 的 3 套示例（去掉 mode 字段后）能正常拉起；边零层/云全层行为与 B 一致。

#### （4）一句话结论

B 已把 LWD 的 PP 支持删干净（config 拒绝 + 组表无 PP + va 零织入）。合并时把 A 的 LWD-PP 代码（head_tail/embedding_only 模式、PP 对建组、`_LWD_FULL_HEAD_TAIL` 中性化、各处 `enable_lwd` PP 门控）作为**删除项**，只保留 A 的 prefill_only 语义与权重裁剪工具；原生 PP 代码一行不动。

### 7.5 Git 操作建议

1. 在 B 侧主干上开 feature 分支（如 `feat/lwd-cloud-reuse-multi-instance`），**用 merge 而非 rebase**（454 提交 rebase 到 4123 提交之上不可行且丢历史）；保留 merge commit 以便整体 revert 回滚。
2. 解冲突时**警惕 §6.2 六个隐藏陷阱**：尤其 modify/delete 文件不要 `git add` 复活旧结构；`config/lwd.py` 与 `lwd_control/` 弃用侧要显式删除；A 的 HTTP 探针死链路不要带进主干；A 的 LWD-PP 代码不要因"保留能力"惯性回流。
3. P0 删除清单作为合并提交的一部分（同一次 commit 里删掉弃用侧，避免半残状态被别人基于其开发）。
4. 合并后 A 分支退役：qxxxw 侧后续开发切到 datacomm 远端（或以 datacomm main 为新基线 re-pin）。
5. 两仓同步：vllm 仓合并 PR 合入后，va 仓合并 PR 基于新 vllm 提交引用（若有 submodule/版本联动则同步升）。

### 7.6 验证方案（合并完成的判据）

| 层 | 验证 |
|---|---|
| 单测 | va `tests/ut/worker/` LWD 套件（B 版 3821 行 + 并入的 A 用例）；vllm `tests/v1/lwd_control/`（B 18 文件 + A 跨进程 14 文件）；B 守卫三件套（weave/style/s1s2）必须全绿 |
| 部署回归 | A 的 3 套 examples 改适配 B 拓扑后跑通：2e1c（云复用续算）、2e2i（网关选云 + control_url 多值）、multi_instance（多实例多前端 attach） |
| 主干回归 | B 原有形态不回退：host staging + prefix refill、multi-DP、KV offload p2p；上游 CI（lwd_fast/checklist）通过 |
| 双向语义对拍 | A 的 `lwd_layer_trace` 逐层对拍工具验证集中式 ↔ 边云输出一致 |
| 协议自检 | B 版本化握手在混合部署下能拒绝旧版本对端（防混部） |

### 7.7 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| lwd_control 双栈静默并存（装配点指向残缺实现） | 高 | P0 删除清单 + `test_lwd_public_surface` 扩展断言单一套路径 |
| `config/lwd.py` 被包遮蔽、配置静默失效 | 高 | 显式删除 + import 全仓扫描 |
| va platform worker_cls 指旧路径 → ImportError | 高 | P2 挂新点 + 启动冒烟 |
| A 深侵入块（44 处 runner 引用 / +455 executor）重放遗漏 | 中 | layer trace 对拍 + 逐层性能基线对比（A 侧性能收益回归） |
| 前缀语义融合后 KV 行为变化（续算边界/误逐） | 中 | B 的 22 用例 + A 的 prefill 续算用例全跑 |
| 两套测试同名文件合并丢用例 | 低 | 逐文件 diff 清单（子报告 §6 已列 4 组） |
| 中文注释/GBK blob 与英文规范冲突 | 低 | 风格守卫豁免期 + 翻译分批 |

## 8. 一页结论

- **性质**：这不是"外部功能并入主干"，而是**同源 LWD 血统在 10 月上旬分叉后的会师**——A 走"多边多云/多实例/云复用/深侵入性能路径"，B 走"社区化重写通信层 + 多 DP + host staging + 上游 0.24 大合流"。
- **2026-11-02 新提交（`baseline_update1031`）带来的变化**：A 把前缀缓存迁移为与 B 同构的 nonce/READY/REFILL 预留设计（删除 `LwdPrefixCoordinator`，新增同名 `lwd_prefix_cache.py`/`lwd_edge_prefix_control.py`）；HTTP/HMAC 链路已无消费者（死链路）；新增结构化输出（wire v5）与权重预读跳过；va 侧补齐 host staging prefix 语义并修补 accepted-state remap。
- **冲突规模**：vllm 41 + va 20 = **61 文件**（46 真实冲突），另有 **6 个** merge-tree 不报的隐藏陷阱；最大冲突域仍是 `vllm/v1/lwd_control/`，但前缀子模块已从"两套范式"收敛为"同一设计、方法名不同"。
- **策略**：**B 骨架 + A 能力包**，P0（架构决策 + 删除/移植清单）→ P1（语义裁决）→ P2（结构重放）→ P3（机械收尾），先 vllm 后 vllm-ascend，merge commit 保留历史可回滚（454 提交）。
- **memc 无需合并动作**（基线自带，B 侧反而更新）。
- **前缀协调已修订为单实现收口**（§7.4.1）：以 B 的 `LwdPrefixReservations` + `LwdEdgePrefixControl` 为唯一账本/状态机，并入 A 的多云路由（cloud_id/engine、is_refill、refill MQ、UP 水位二元组键）；**删除** A 已无消费者的 HTTP/HMAC 链路；消息统一升 v5。
- **LWD 的 PP 并行已从 B 移除**（§7.4.7）：合并后删除 A 的 `head_tail`/`embedding_only` 模式、PP 对建组、`_LWD_FULL_HEAD_TAIL` 中性化与各处 `enable_lwd` PP 门控；只保留 prefill_only 语义与权重裁剪工具，**原生 PP 代码不动**。
- **其余决策**（§7.4.2–§7.4.6）：
  - D2 世界布局：B `LwdParallelGroupsSpec.from_topology` 唯一布局引擎，A 的共享边/多连接能力做成 spec 扩展（解掉 B 两处 `NotImplementedError`），A 的 pre-init 拓扑守卫改为比较 topology digest；
  - D3 参数体系：master_addr 取 B，CLI 以 topology YAML 为唯一源，A 的数字参数降级为 `gen_lwd_topology.py` 输入 + 一版兼容 shim，registry 一次性导入迁移；
  - D4 spec 恢复链：主干丢弃（与 B defer-prefill 一致、B 已判定为死代码）；A 在 11-02 修补了 accepted-state remap，PhaseScheduler 实验臂若保留需一并带入；
  - D5 host staging 文档：B 英文文档为 canonical，A 中文 v7 改名归档，独有根因/决策推导并入 canonical 的 Design notes；
  - **D6 结构化输出（新增）**：按 A 的 5 文件 +95/-24 方案移植进 B（消息字段 + 云侧 grammar_init + 边侧 picker 过滤），与 D1 的 v5 升版合并一次完成。
