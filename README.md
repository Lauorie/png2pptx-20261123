# PNG → 可编辑 PPTX：正向 Agent 交接版

交接给 juli，整理日期：2026-11-19。代码来源为原目录
`/mnt/shared-storage-user/intern7shared/kouyiqun/png2pptx_geometry`。
集群外部署见 [部署说明](DEPLOYMENT.zh-CN.md)。
完整流程入口是 `run_ppt.py`，推荐通过下面的 `scripts/prepare_forward.py` 使用。

GLM 的模型名、thinking、分阶段 token 上限、并发、超时及本地服务历史配置见
[GLM 运行参数说明](GLM_PARAMETERS.zh-CN.md)。

输入原始 PNG，分别恢复图形、箭头和文字，再统一处理重复对象、遮挡、图片素材与层级，
输出可编辑 PPTX。SenseNova 用于需要编辑或补全的图片素材，SAM3 提供定位、轮廓和 mask 证据；
图形、箭头、文字的多模态推理使用外部兼容 API。

本目录是可继续开发的代码与模型副本。主流程、文字后端、箭头后端、SAM3 推理源码、
LightLLM / LightX2V、渲染器、字体及实际权重均已复制。
历史批次产物、数据集、预测缓存、密钥、运行中的锁文件没有作为运行输入打包。
少量方法来源清单保留在 `backends/reference/arrow_method/`，仅供代码来源核对。

## 目录

| 路径 | 内容 |
| --- | --- |
| `run_ppt.py`、`full_recovery/` | 完整流程准备、调度、图片编辑与 PPT 合成 |
| `run_shapes.py`、`src/`、`png2pptx/` | 图形定位、拟合、样式及原生 PPT 导出 |
| `backends/reference/` | 完整入口需要冻结的文字实现、旧箭头方法代码及 SAM 配置来源 |
| `third_party/arrow_geometry/` | 原目录已指定的 geometry-first 箭头实现；新任务从原图重新推理 |
| `weights/sensenova/SenseNova-U1.5-8B-MoT/` | SenseNova 模型、8 个 safetensors 分片、索引和 tokenizer |
| `weights/sam3_best_epoch7/` | 微调 SAM3 best epoch 7 的 `best.pt` 和 `best.json` |
| `deploy/sam/` | SAM HTTP 服务、SAM3 推理源码、类别表与部署配置模板 |
| `deploy/sensenova/` | SenseNova 服务、LightLLM / LightX2V 源码及定制 Python runtime overlay |
| `.runtime/` | 两套实际渲染器、glibc 兼容库、字体和辅助 Python 包 |
| `scripts/` | 任务准备、输入登记、CPU rjob 提交和离线校验 |
| `tests/` | 随主流程保留的回归检查 |
| `provenance/` | 来源、复制哈希、打包改动和验证记录 |

## 权重与运行环境

权重是独立文件副本，未用软链接指向原权重。合计 **45,152,405,448 字节，约 42.05 GiB**。
逐文件源路径、大小和 SHA-256 见 [权重清单](provenance/weights_manifest.json)。

- **SenseNova-U1.5-8B-MoT**：约 32.66 GiB，保持原有模型文件。
- **SAM3 best（epoch 7）**：约 9.39 GiB；这是微调版本。checkpoint SHA-256：
  `a9dc4a2a9752165feac412f961c6fa6e017f60fb8f178621d9caba93aa0276ed`。
  checkpoint 按原文件复制，没有删减训练状态或重新导出。

当前交接路径采用“外部多模态 API + CPU Agent + 独立 SAM / SenseNova GPU 服务”。
未打包本地 GLM 大模型权重。API 模型名默认沿用 `glm-5.3-flash`，可传 `--api-model` 指定；
模型提示词、阶段预算和原有 thinking 配置保留，打包没有额外指定 reasoning effort。

**基础 Conda 环境、CUDA 和容器镜像仍按集群既有资产挂载，未整份复制。**
SAM 的基础环境来自 `gpfs1/dongdaozhu/conda/envs/sam3`；
CPU 基础环境来自 `gpfs1/kouyiqun/conda/envs/SVG`；
SenseNova 基础环境来自 `gpfs1/yinjinhui/H-conda/anaconda3/envs/openrlhf`。
可通过部署脚本的环境变量换成有访问权限的等价环境。
SenseNova 原服务中相对于基础环境新增或覆盖的包已复制到 `runtime_overlay/`，
重建脚本会连接基础包并恢复覆盖内容，不需要旧实验目录。

## 1. 准备 CPU Python 环境

在 Linux 上使用 Python 3.12，与本次验证的锁定依赖保持一致。示例：

```bash
cd /path/to/png2pptx_forward_agent_clean_20261119
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu-validated.txt
```

这里的锁定版本来自本次 Python 3.12 离线验证环境；原来的宽松依赖清单也保留在
`requirements.txt`。pip 环境用于准备任务和离线检查。
集群 CPU worker 仍通过挂载 `/opt/shape-cpu` 启动，并使用随包提供的 LibreOffice、
兼容库和字体。跨机器自建环境需要系统 fontconfig；依赖安装成功本身不代表原生渲染已通过。

## 2. 部署 SAM3 best epoch 7

准备新的部署目录，然后提交单卡任务：

```bash
python deploy/prepare.py sam --run-dir runtime_runs/sam_juli
bash deploy/submit.sh runtime_runs/sam_juli --dry-run
bash deploy/submit.sh runtime_runs/sam_juli
```

复用的资源配置：`ailab-llmmultimodal / llmmultimodal_gpu_pool`，1 GPU、
8 CPU、49152 MiB，端口 `18103`；模型代码要求单卡可见及 BF16 支持。
如需换基础环境挂载，设置 `SAM_ENV_MOUNT=gpfs://gpfs1/你的路径`。

加载完成后查看 `runtime_runs/sam_juli/endpoint.json`，并检查：

```bash
curl --noproxy '*' http://SAM_POD_IP:18103/health
```

应确认 `warmup_passed=true`、`checkpoint_hash_verified=true` 及 `checkpoint.epoch=7`。
分割接口为 `POST /v1/segment`，请求中的图片和输出目录必须对服务所在 worker 可见：

```json
{
  "image": "/absolute/shared/path/example.png",
  "output": "/absolute/shared/path/new_sam_output",
  "categories": ["line", "block_arrow"],
  "threshold": 0.5
}
```

输出目录每次使用新路径。服务支持类别表中的全部类别；
上面的类别仅为请求示例，Agent 会按自己的阶段配置选择类别和阈值。

## 3. 部署 SenseNova

先用原服务相同的 CPython 3.12 基础环境重建 overlay，输出放在共享目录：

```bash
python deploy/sensenova/setup_runtime.py \
  --base-python /mnt/shared-storage-user/yinjinhui/H-conda/anaconda3/envs/openrlhf/bin/python \
  --output "$PWD/.runtime/sensenova"
python deploy/prepare.py sensenova --run-dir runtime_runs/sensenova_juli
bash deploy/submit.sh runtime_runs/sensenova_juli --dry-run
bash deploy/submit.sh runtime_runs/sensenova_juli
```

更换基础环境时，重建 overlay 使用对应的 `--base-python`，
部署准备增加 `--base-environment /新环境路径`；
提交前设置 `SENSENOVA_BASE_ENV_MOUNT=gpfs://gpfs1/对应环境路径`。
运行时目录变更用 `--sensenova-python /共享路径/bin/python` 指定。

复用的配置为 2 GPU、16 CPU、131072 MiB、CUDA 12.8，端口 `8091`：
理解 TP=1、图像生成 GPU=1、CFG parallel size=1、PyTorch SDPA、
`max_req_total_len=65536`、`max_total_token_num=69632`。
GPU 任务统一使用原已验证的容器镜像 digest，见 `deploy/submit.sh`。
这是原运行配置，未宣称为最小硬件需求。

从 rjob 状态中取得当前 Pod IP，检查：

```bash
curl --noproxy '*' http://SENSENOVA_POD_IP:8091/health
curl --noproxy '*' http://SENSENOVA_POD_IP:8091/v1/models
```

模型 ID 为 `SenseNova-U1.5-8B-MoT`。
图片编辑使用 `POST /v1/images/edits`；Agent 传入的基址是 `http://SENSENOVA_POD_IP:8091/v1`。
请求格式由 `full_recovery/image_edit.py` 对接。
健康检查、模型列表与真正返回可解码图片分别验证；本次交接没有新做 GPU 推理验收。

两个服务都保留请求优先的功率保活：
无推理请求且实测功率连续低于 enforced power limit 的 40% 达 600 秒后，
才启动以 60% 功率为目标的空闲负载；有请求立即让出并重新观察。
健康 GET 不算推理请求，不修改 GPU 硬件功率上限。

## 4. 准备输入并运行完整 Agent

支持 PPTBench 的 `taskset.txt + tasks/<id>/reference.png`，也支持 `catalog.json`。
普通 PNG 文件夹可先登记，原图原地引用：

```bash
python scripts/make_input_catalog.py --images /absolute/shared/my_pngs --output inputs/my_pngs
```

准备 API key 文件，一行一把 key，权限 `0600`。该文件由接手人自行提供，包内没有密钥。
使用实际服务地址准备一个全新任务：

```bash
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

`--api-key-file` 可重复传入多份文件；`--ids id1 id2` 可选择子集。
不传 `--image-edit-url` 会关闭生成式图片编辑。
准备时会检查 SAM 的实际健康与模型身份；`--prepare` 只冻结代码、登记源图，不调用恢复模型。
CPU worker 使用 `ailab-intern10 / intern10_proxy`，默认 16 CPU / 65536 MiB / 0 GPU。
换 CPU 环境挂载可设置 `CPU_ENV_MOUNT=gpfs://gpfs1/对应环境路径`。

原图数据和权重应位于 worker 可见的共享盘。任务目录必须是新目录；
不要给运行中的任务启动第二个 writer，也不要在冻结之后修改任务内代码或配置。
搬迁本交接包后重新准备任务与服务；已准备任务含绝对路径，不直接搬迁续跑。

## 5. 产物与排查

- `runs/juli_first/delivery/<id>/combined.pptx`：最终 PPTX。
- `runs/juli_first/delivery/<id>/preview.png`：实际原生渲染。
- `runs/juli_first/shape/`、`fresh_arrows/`、`texts/`：各分支状态、请求与中间证据。
- `source_manifest.json`、`provenance.json`、`glm_profiles.json`：本次输入与实现来源。
- `status.json`、`progress.json`、`logs/`：作业状态、逐图进度和失败原因。

提交后从 `jobs/job.id` 取得 metadata job ID，使用
`rjob get --namespace <namespace> <metadata-id>`、`rjob events` 和日志检查排队、加载及实际进度。
`created_rjob_name`、`Running`、健康检查和最终视觉正确各自代表不同检查层级。
模型仍可能留下 unresolved；以实际原图 / PPT 渲染和阶段证据判断质量。

## 6. 离线检查与交接记录

```bash
python scripts/verify_bundle.py
python scripts/verify_bundle.py --weights
python scripts/check_prepare.py
python scripts/check_render.py
```

第一项核对非权重内容哈希及所有文件大小；第二项会完整读取约 42 GiB 权重。
第三项使用临时 PNG、临时占位凭据和本地模拟 SAM health，
真实执行完整任务准备，并禁止读取原代码 / 实验目录；不调用真实模型。
第四项用一个临时原生 PPT 核验两套渲染器和字体，不代表重建质量验收。
具体交接验证结果见 [验证记录](provenance/VALIDATION.md)。

[原文件来源](provenance/source_files.json)、
[打包适配记录](provenance/adaptations.json) 和
[最终内容清单](provenance/bundle_manifest.json)
用于追踪来源与交接后的修改。
只调整了打包路径、显式凭据入口和部署配置；原多阶段算法与预算未在整理时重写。
