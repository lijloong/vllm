# LWD 网关单流协调设计

> 日期：2026-11-20
> 状态：实现候选，待代码评审
> 适用仓库：`vllm`；ZMQ、HCCL 和 `vllm-ascend` 数据面不在本次改造范围内

## 1. 决策与目标

HTTP 协调面以《边-网关-云的交互逻辑》为验收合同。每次用户推理最多增加一次经过平台网关的 `POST /v1/chat/completions`，启动控制信息、保活和最终 usage 共用一条 SSE 响应。

本设计完全替换此前的“候选 `/v1/lwd/probe` + 赢家 `/v1/lwd/activate`”两段 HTTP 协议，不保留新旧客户端调用分支。

关键决策：

1. 保留每云 DP 一个 HTTP 监听、`LwdConnectionId(edge_id, cloud_id, dp_idx)` 和现有 ZMQ/HCCL 数据面。
2. 保留逐块及尾部 HMAC manifest、文本 v1、图片 v2、MM-ABI、真实 prefix hit 和真实 usage。
3. Edge 不再向全部候选 DP 并发 HTTP probe。每个请求只进入一个网关地址；共享网关地址时，由网关按 `model` 和路由配置选择实际上游。
4. Edge 必须收到并校验启动 content，才能发送 `RequestNotify` 和准入推理。HTTP 200 或响应头本身不代表可以准入。
5. 文档要求的字段位于 OpenAI 兼容外层；LWD 扩展的 Cloud/DP、alignment 和 nonce 位于启动 content 的控制 JSON 内，不增加私有响应顶层字段。

## 2. 单请求交互

```text
Edge
  → POST /v1/chat/completions
  → Gateway 按 model/路由配置选择 Cloud DP
  → Cloud 内部只读 peek + 注册 usage generation
  → SSE startup content
  → SSE empty delta
  → Edge 校验结果、绑定 ConnectionId、发送 RequestNotify、准入推理
  → SSE comment heartbeat（推理期间每 10 秒）
  → SSE 真实 usage 或 usage-unavailable
  → data: [DONE]
```

请求体只允许：

```text
model、messages、stream、stream_options
```

其中 `stream=true` 和 `stream_options.include_usage=true` 与用户到 Edge 的流式选择无关。真实 prompt、采样参数和媒体内容不经过网关。

摘要前缀按块形态区分：完整块使用 `ecb1:`，不足一个完整块的尾块使用 `ect1:`。尾块至多一个且只能位于 messages 末尾；两者均为 base64url 编码的 32 字节摘要，摘要数量必须与 `block_size`、`prompt_tokens` 匹配。

请求头使用合同规定的 `X-Edge-Cloud-*`、可选 `Authorization` 和 `X-Mse-Consumer`。不再发送 `X-Lwd-Probe-Nonce`，也不要求 Edge 预先填写 Cloud/DP 请求头。

## 3. 启动控制信息与路由绑定

Cloud 在首个 `choices[0].delta.content` 中返回 JSON 字符串，包含：

- 合同基础字段：`type/protocol/request_id/instance_id/block_size/hit_blocks/hit_tokens`。
- `cloud_id`：多 Cloud 可见时必填；单 Cloud 时可省略并推断。
- `engine_index`：该 Cloud 多 DP 可见时必填；单 DP 时可省略并推断。
- `alignment_tokens`：可选，缺省为 0。
- `prefix_nonce`：当前 LWD 代际机制必填，用于 RequestNotify、C2E 和 usage 的代际隔离。
- 图片 v2 的 `mm_abi`：仅图片请求必填，文本请求不得携带。

Edge 支持网关将 content 拆成多个 chunk，拼接并完整解析后才校验。解析器忽略非 JSON 插件内容，并能在后续 JSON 起点重新同步，同时保留嵌套对象的正常分片。以下情况均不得准入：

- 启动超时、流提前结束或 content 超过大小限制。
- 协议、request ID、MM-ABI、nonce 或数值字段不合法。
- 返回的 `(cloud_id, engine_index)` 不属于该逻辑边的可见连接。

校验成功后，Edge 以启动消息中的 Cloud/DP 构造 `LwdConnectionId`，而不是按 cloud ID 猜测 DP。后续 RequestNotify、abort、C2E、DOWN 和 usage 均固定到该连接。

响应头同时携带合同要求的协议、请求 ID、实例、block size、命中信息以及 `Cache-Control: no-cache`、`X-Accel-Buffering: no`。握手只依赖 content，不依赖网关是否透传这些响应头；全部自定义响应头被剥离时仍必须完成握手和 DP 绑定。

## 4. Cloud 内部职责

每个 DP listener 仍共享唯一的 `LwdCloudControlBridge` 和 event queue：

```text
listener(dp_idx)
  → bridge.activate(canonical_request_id, dp_idx)
  → bridge.probe(dp_idx, manifest, reply_key)
  → command_queues[dp_idx]
```

`activate` 和 `probe` 是 Cloud 进程内部步骤，不是两个 HTTP API：

1. usage generation 在响应头之前注册，避免快速请求先完成、usage 无处投递。
2. probe 仍为只读 peek，不建立正式 reservation。
3. 正式 reservation 仍只由后续 RequestNotify 在对应 EngineCore 建立。
4. 同 canonical request ID 的新 activation supersede 旧 generation；旧 nonce 事件直接丢弃，同一 nonce 但来自错误 DP 的 usage 立即按不可用收流，不等待全局超时。
5. listener bind 失败仍使整个 Cloud 启动失败，不能只启动部分 DP 后报告 ready。

旧 `/v1/lwd/probe` 和 `/v1/lwd/activate` 路由及 Edge 调用被删除。

## 5. SSE 顺序与 usage

正常流严格为：

```text
data: <startup chat.completion.chunk>
data: <empty-delta chat.completion.chunk>
: edge-cloud-usage-pending
...
data: <choices=[] 的真实 usage chunk>
data: [DONE]
```

控制帧不包含 `usage`，空 delta 的 `finish_reason=null`。usage 后直接发送 `[DONE]`，不额外发送 stop/length choice 帧。

usage 继续使用现有调度账本：

- `prompt_tokens` 来自实际请求。
- `completion_tokens` 来自实际接受的输出。
- `prompt_tokens_details.cached_tokens` 来自真实 prefix hit，包含在 prompt tokens 中。
- 启动文本、空 delta、心跳和摘要文本不参与 Token 计数。
- abort 已有接受 Token 时照常发布真实 usage；零接受 Token 时发布内部无用量事件，对外为 `usage-unavailable + [DONE]`，不得构造零 usage。

RequestNotify 内层等待使用 `probe_timeout + PROBE_HEADROOM_S`（默认 30 秒），真实 usage 总等待仍受 `usage_wait_timeout`（默认 600 秒）约束。Cloud peek 和 Edge 启动分别使用 `probe_timeout - PROBE_HEADROOM_S`（默认 20 秒）和 `probe_timeout`（默认 25 秒）。RequestNotify 等待超时、推理超时、abort 后无结算结果或其他无法取得真实用量的情况发送：

```text
: edge-cloud-usage-unavailable
data: [DONE]
```

不得发送零用量，也不得按可见文本长度估算。usage 等待异常同样按不可用收流。Edge 在启动后继续后台收流；已建立流的 abort 不通过主动断开连接代替上述正常结束。

## 6. Edge 生命周期与失败语义

`LwdEdgeProbeManager` 每个请求只提交一个 activation job：

- runtime 负责异步 HTTP、启动 deadline 和完成通知，不读取 Scheduler 账户。
- engine thread 收到有效 completion 后才执行 `_lwd_admit_and_notify()`。
- RequestNotify 成功后才 `mark_admitted()`，移除未准入代际但保留后台 drain。
- 启动失败、提交失败或结果不合法时 fail-closed，不回退成无握手推理。
- 启动前 abort 可以取消等待；启动后 abort、同 ID 重提或路由校验失败均不得主动断开已建立的 drain，由 Cloud 返回真实 usage 或 unavailable。
- 同 request ID 重提使用内部 generation 丢弃旧 completion；Cloud nonce 用于数据面和 usage 代际匹配。

coordination 完全关闭时，才走不涉及 HTTP activation 的原有路径。

## 7. 拓扑与部署

继续使用 DP 的 `coord_port/coord_url`：

- 直连部署可由 `dp.addr/coord_port` 派生地址。
- NAT、DNS 或平台网关使用显式 `coord_url`。
- 多个 DP 可共享同一个网关 `coord_url`；网关根据 `model` 和路由配置选择上游，Edge 根据启动 content 识别实际 Cloud/DP。
- 同一物理地址上的 `coord_port` 冲突、与 `ctrl_port/data_port` 冲突及监听启动失败仍按现有规则 fail-fast。
- 不增加第二套 gateway 配置，不修改 ZMQ、HCCL、host staging、reservation/pin 或 preemption 的职责。

## 8. 验收矩阵

### 8.1 HTTP 与客户端

- 请求路径、方法、body 白名单、固定 stream/include_usage 和合同请求头。
- 首个 content、第二个空 delta、10 秒 comment 心跳、真实 usage 和 DONE 的顺序。
- usage 尚未产生时，启动帧和空 delta 已可被客户端读取。
- content 跨 chunk 拼接；HTTP 200 后无启动 content 不得准入。
- request ID、协议、MM-ABI、nonce 和 Cloud/DP 错误均拒绝。
- usage-unavailable 后只有 DONE，整条流没有 usage 字段或零用量。
- 旧 `/v1/lwd/probe`、`/v1/lwd/activate` 不再可用。

### 8.2 Edge 与 Cloud 状态

- 每请求一次 HTTP activation、一次 RequestNotify、一条 usage stream。
- 网关返回不同合法 DP 时，Edge 绑定启动消息中的实际 DP，并断言应答 DP、RequestNotify/运行 DP 和 usage 发布 DP 相同。
- prefix 开关只改变 chain/hit 语义，不改变单请求交互。
- usage 必须匹配 canonical request ID、DP 和 nonce；旧 generation 事件不得完成新请求。
- 启动前 abort 可取消等待；启动后 abort 不主动断开 drain。
- 多 DP listener 仍只访问自己的 command queue，bind 失败使整云启动失败。

### 8.3 系统验收

真实 Higress 环境仍需验证：

- model 路由、控制头与 JSON 透传、启动 content 分片和禁用缓冲。
- 长推理期间网关不因 idle timeout 断流。
- usage 与独立数据面账本一致，unavailable 不被平台记成零消耗。
- abort、重提、多 DP、多边多云和图片 v2 的端到端行为。

单元测试不能替代上述真实网关与 NPU 验收。
