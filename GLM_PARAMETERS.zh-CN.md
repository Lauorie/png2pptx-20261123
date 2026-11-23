# GLM 运行参数

本说明对应交接包的 `scripts/prepare_forward.py` 完整流程，按 2026-11-19 的实际代码整理。
准备入口会覆盖部分源配置，因此不能直接把根目录 `config.json` 中的并发数当作最终运行参数。

## 当前使用方式

当前交接流程通过兼容 OpenAI Chat Completions 的外部 API 调用 GLM，CPU worker 不加载 GLM 权重。
SenseNova 与 SAM 使用各自的 GPU 服务，部署方法见 [README](README.md)。

完整命令如下，先替换 API 地址、凭据路径以及两个服务地址：

```bash
cd /mnt/shared-storage-user/intern7shared/yinjinhui/png2pptx_forward_agent_clean_20261119

python scripts/prepare_forward.py \
  --input-root inputs/my_pngs \
  --run-dir runs/juli_first \
  --api-base-url "$API_BASE_URL" \
  --api-model glm-5.3-flash \
  --api-key-file /protected/api_key_file \
  --api-concurrency 48 \
  --case-workers 2 \
  --text-workers 2 \
  --critic-profile full \
  --sam-url http://SAM_POD_IP:18103 \
  --image-edit-url http://SENSENOVA_POD_IP:8091/v1

bash scripts/submit_ppt_api_cpu.sh runs/juli_first
```

`--prepare` 阶段登记原图、冻结代码，并检查 SAM 身份；真正的 GLM 请求在 worker 运行时发起。
上面的示例不额外传入 `--force-reasoning-effort`，保留各阶段已有的推理配置。

## 命令行参数和最终默认值

| 参数 | 完整流程默认值 / 要求 | 作用 |
| --- | --- | --- |
| `--input-root` | 必填 | PPTBench 格式或含 `catalog.json` 的输入目录 |
| `--run-dir` | 必填，必须是新目录 | 代码快照、请求记录和产物目录 |
| `--api-base-url` | 包装入口要求显式提供 | 上游 API 基址，例如以 `/v1` 结尾；程序追加 `/chat/completions` |
| `--api-model` | `glm-5.3-flash` | 发往上游的模型标识；需与提供方实际支持的名称一致 |
| `--api-key-file` | 必填，可重复 | 每行一把 key；文件权限须为 `0600` |
| `--api-concurrency` | `48` | 本任务 GLM 上游请求并发上限 |
| `--case-workers` | `2` | 当前 fresh-full 流程的图形样本并发；不改变箭头分支的样本数设置 |
| `--text-workers` | `2` | 同时运行的文字样本数 |
| `--critic-profile` | `full` | `full` 开启 geometry / style / pattern critic；`lean` 关闭 geometry 和 pattern，保留 style |
| `--force-reasoning-effort` | 不传 | 可选 `low/medium/high`；属于显式覆盖选项，常规交接示例不启用 |
| `--api-budget-config` | 不传 | 可选的持久化共享请求预算配置文件 |
| `--ids` | 全部登记样本 | 指定需要处理的样本 ID |
| `--sam-url` | 包装入口要求显式提供 | SAM 服务基址；准备时核验 epoch 和 checkpoint 哈希 |
| `--image-edit-url` | 不传则关闭生成式图片编辑 | SenseNova 的 `/v1` 基址 |
| `--image-edit-model` | `SenseNova-U1.5-8B-MoT` | 图片编辑服务的模型名 |
| `--image-edit-concurrency` | `2` | SenseNova 图片编辑并发，与 GLM 并发分开 |

参数定义见 [run_ppt.py](run_ppt.py)。包装入口未直接声明的参数会透传给它。
完整参数帮助：

```bash
python run_ppt.py --help
```

## Thinking、输出预算和采样

API relay 默认发送 `thinking={"type":"enabled"}`，并使用 SSE 流式响应。
**没有全局强制推理强度，不等于每个阶段都省略 `reasoning_effort`。**
原代码中图形观察通常为 `high`，几何 critic 默认使用 `low`，部分样式判断也使用 `low`；
文字准备阶段保留参考实现的 `high` 配置及局部覆盖。geometry-first 箭头主配置开启 thinking，
未指定统一的 `reasoning_effort`。本次整理没有统一改写这些阶段参数。

下表是完整流程的主要预算；局部观察、critic、补漏和修订可能使用更小的单次上限：

| 分支 | 主要 token 上限 | 生效方式与来源 |
| --- | ---: | --- |
| 图形观察、拟合和样式 | `100000` | `run_shapes.prepare()` 设置 `glm_max_tokens=100000`；`ForwardRuntime.call()` 取阶段预算与此上限的较小值 |
| 图形几何 critic | 至多 `16384` | critic 内部对 `check_max_tokens` 再取较小值；部分辅助检查为 `8192` |
| geometry-first 箭头 | `113000` | `third_party/arrow_geometry/config.json`；样式等子阶段仍可设置各自预算 |
| 文字全局调用配置 | `131072` | 冻结文字后端的 `global_max_tokens` |
| 文字局部 / 两阶段配置 | `100000` | 冻结文字后端的 `texts.max_tokens`；局部调用继续沿用阶段规则 |

图形请求通常使用 `temperature=0` 和 `seed=42`；箭头主配置为 `temperature=0.1`。
这些是代码请求参数，上游是否接受及如何处理由服务端决定。
`max_tokens` 是输出预算，不是模型总上下文长度，也不代表每个请求会生成这么多 token。

主 CLI 没有统一的 `--max-tokens` 参数。需要调整预算时，先修改对应准备逻辑或后端配置，
然后重新准备一个新任务；不要直接修改已冻结任务内的配置，否则文件哈希校验会失败。

## 并发如何分配

默认完整流程同时运行图形、文字和箭头分支：

- 图形样本并发 `2`，通过 `--case-workers` 设置。
- 文字样本并发 `2`，通过 `--text-workers` 设置；文字后端参考配置另有每样本 API 并发 `4`。
- geometry-first 箭头准备时设置样本并发 `2`；内部 `task_workers=8`。这两个设置不能直接用 `--case-workers` 一起调整。
- `--api-concurrency=48` 是上游请求配额，不是同时处理 48 张 PNG。

一至两把 key 时，各分支共用总请求配额。至少三把 key 且总并发不小于 6 时，
完整流程为 shape / arrow / text 分配独立 relay 和请求名额；默认总并发 48 分为各 16。
三路端口分别为 `18188`、`18189`、`18190`，均监听 worker 的 loopback。
分支自身的样本数和线程数仍可能使实际并发低于分配上限。

具体分配以任务 `settings.json` 的 `api_relays`、`api_global_concurrency`、
`glm_concurrency` 及 `fresh_arrows/config.json` 为准。
传入的多个 key 文件会按文件中的非空行去重并分配，不能仅以文件数量判断 key 数量。

## 超时和重试

| 配置位置 | 当前主要值 | 含义 |
| --- | ---: | --- |
| API worker / relay | `3600` 秒 | `api_request_timeout_seconds` 和 `glm_request_timeout_seconds` 的准备默认值 |
| 图形阶段配置 | `1200` 秒 | 阶段可以进一步缩短请求等待；不能把 relay 的 3600 秒理解为所有请求的统一时限 |
| geometry-first 箭头主配置 | `2400` 秒 | 请求超时及 stream idle timeout；部分子阶段可覆盖 |
| geometry-first 箭头请求 | `request_max_attempts=2` | 主配置中的请求尝试数；同时保留原有格式 / 传输重试规则 |

其他阶段的重试由各自的 schema、格式和传输规则控制；没有统一的“全流程重试 N 次”。
relay 要求收到完整的 SSE 结束信息，截断或不完整流不会作为正常完成结果使用。

## 准备后去哪里查实际参数

每个任务会生成以下记录：

| 文件 | 用途 |
| --- | --- |
| `runs/<name>/settings.json` | 实际模型、上游 URL、relay 配额、图形 / 文字并发和 worker 参数 |
| `runs/<name>/glm_profiles.json` | GLM 配置概要；不包含所有子阶段的实际覆盖值 |
| `runs/<name>/shape/pipeline_config.json` | 图形阶段的配置 |
| `runs/<name>/backends/text/config.json` | 实际冻结的文字配置 |
| `runs/<name>/fresh_arrows/config.json` | 当前实际执行的 geometry-first 箭头配置 |
| 各阶段请求目录中的 `request.json`、`transport.json`、`response_*.json` | 单次请求的实际预算、推理配置、计时和响应证据 |

`runs/<name>/arrows/config.json` 属于兼容旧后端的配置；本交接版默认执行
`fresh_arrows/`，判断箭头参数时优先看后者。
预算概要、模板值与单次调用不一致时，沿实际执行分支核对请求记录和代码覆盖规则。
凭据位于任务的受保护 `private/` 目录，不应随参数记录对外分享。

## 代码中保留的本地 GLM GPU 参数

下面是 `full_recovery/prepare.py` 内保留的本地双服务配置，供追溯原实现：

| 项目 | 服务 A | 服务 B |
| --- | ---: | ---: |
| Tensor parallel | `4` | `4` |
| Data parallel | `1` | `1` |
| `max_model_len` | `262144` | `262144` |
| `max_num_seqs` | `24` | `24` |
| GPU memory utilization | `0.90` | `0.72` |

该配置原本配合 8 GPU、同机 SAM 等服务使用。当前 `--api` 配置会移除这些本地服务字段，
API 模式的上下文容量由上游服务管理，不能据此认定外部 API 提供 262144 上下文。
本交接包未复制本地 GLM 权重，也未把本地 GLM GPU 模式作为已验收的一键部署入口；
上述表格是代码中保留的配置记录，本次没有重新启动本地 GLM 进行验证。

## 参数来源

- [入口默认值](run_ppt.py)、[API 准备与并发分配](full_recovery/prepare_api.py)。
- [图形准备](run_shapes.py)、[图形请求预算裁限](experiment/shape_sam_guided/runtime.py)。
- [几何 critic](src/shape_integration/fitting/critic.py)、[推理覆盖配置](full_recovery/inference_profile.py)。
- [箭头准备](full_recovery/fresh_arrow.py)、[箭头模板](third_party/arrow_geometry/config.json)。
- [文字冻结与本地服务历史参数](full_recovery/prepare.py)、[文字模板](backends/reference/shape_text_glm_geometry_20261019/config.json)。
- [上游 SSE relay](full_recovery/api_gateway.py)。
