# 部署说明（面向后端）

本文说明怎样在原集群之外部署这套 PNG → 可编辑 PPTX 系统。内容来自 2026-11-23 至 11-25 在两台租用 GPU 机器上的实测：第一台用于摸索，第二台是全新实例，严格按本文从 GitHub 拉代码、从零搭环境、启动服务并跑通转换（记录见第 6 节）。凡是没有实测的部分都单独标出。

原作者的 `README.md` 描述的是原集群（rjob、gpfs 挂载）上的用法，本文不依赖那些设施。

## 1. 系统组成

一次转换由一个 CPU worker 进程驱动，它调用三个常驻服务：

| 组件 | 作用 | 资源 | 接口 |
| --- | --- | --- | --- |
| CPU worker | 调度图形、文字、箭头三个分支，合成 PPTX | CPU；实测机器上单任务无 GPU 占用 | 命令行，见第 4 节 |
| 多模态大模型 | 三个分支的识别与推理 | 外部 API | OpenAI 兼容的 `/chat/completions` |
| SAM3 服务 | 定位、轮廓、mask | 1 张 GPU，建议 16 GB 显存（实测整卡最高 12.4 GB，见 5.3） | HTTP，端口 18103 |
| 图片编辑服务 | 图片素材的编辑与补全 | 外部 API（推荐），或自建 SenseNova（1 张 GPU，约 62 GB 显存） | HTTP，`/v1/images/edits` |

图片编辑有两种接法，见 5.2。用外部图片模型时，整套系统只需要一张 16 GB 显存的 GPU 跑 SAM3，`weights/sensenova`、`runtime_overlay/` 和 SenseNova 环境都不用准备。自建 SenseNova 时，两个 GPU 服务可以放在同一张 80 GB 的卡上（实测为 A800 80GB，共占用约 74 GB）。

三条必须遵守的约束：

- **worker 和 SAM3 必须共享文件系统，且路径一致**。SAM3 的请求里传的是图片和输出目录的绝对路径，不是图片内容。
- **一个容器同一时间只跑一个 worker 进程**。worker 会改写 `/usr/bin/libreoffice`，并在 `/tmp/shape-recovery-lo` 下放置渲染器，因此需要 root，且多个 worker 会互相覆盖。要并行就把多张图放进同一个任务（见第 4 节），或者开多个容器。
- **自建 SenseNova 时不要用 RTX PRO 6000 Blackwell**。SenseNova 在这张卡上预热时会触发 CUDA 非法访存，之后 GPU 需要宿主机复位才能恢复。

## 2. 需要准备的文件

代码仓库之外，还有三类不进 git 的大文件，都要放到仓库目录下的同名位置：

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `weights/` | SenseNova 和 SAM3 权重 | 约 42 GiB |
| `.runtime/` 下的 `arrow_renderer`、`portable_lo`、`renderer_support`、`shape_layers_py311`、`text_font_bundle_20261021` | 两套 LibreOffice、兼容库、字体、辅助 Python 包 | 约 1.2 GB，约 1.7 万个文件 |
| `deploy/sensenova/runtime_overlay/` | SenseNova 运行时的 Python 包覆盖层 | 约 3.2 GB，约 2.2 万个文件 |

`weights/` 可以是指向别处的软链接。

**打包传输请用 tar，不要用 zip**。zip 会丢掉 `.runtime/` 里的软链接和可执行权限，渲染器随后无法启动。

文件放好后用清单核对：`provenance/bundle_manifest.json` 记录了每个文件的 SHA-256，`provenance/weights_manifest.json` 记录了权重。注意 `scripts/verify_bundle.py` 会对本分支改过的文件报哈希不符（清单是原作者打包时生成的），改动列表见第 8 节。

`.runtime/sensenova/` 不需要传输，它由第 5.3 节的搭建脚本在本机生成，里面有指向本机环境的绝对路径。

## 3. CPU worker 环境

### 3.1 依赖

依赖锁在 `deploy/standalone/requirements.worker.txt`，Python 3.12，系统为 Ubuntu 22.04。这是目前唯一跑通过端到端的组合。

仓库里另外两份版本记录**不能用于 worker**：

- `requirements.cpu-validated.txt` 锁的 numpy 2.5.5 不支持二维 `np.cross`，合成步骤会崩溃。
- `provenance/original_local_cpu_versions.json` 记录的 OpenCV 5.0.0.93 与代码不兼容：`full_recovery/source_marks.py` 按 OpenCV 4 的返回形状写，OpenCV 5 下部分图片会在合成时报 `IndexError`。

原集群真实的 worker 环境清单没有随包提供。当前锁里的 `opencv-python-headless==4.10.0.84` 是验证时选定的版本，不是原作者确认过的版本。

### 3.2 用镜像

```bash
docker build -f deploy/standalone/Dockerfile.worker -t png2pptx-worker .
```

**这个 Dockerfile 没有实际构建过**（验证机器上没有 Docker）。它装的包和版本与实测环境一致，但镜像里渲染器能否正常工作需要用第 6 节的检查确认。

镜像里只有 Python 环境。运行时把仓库目录（含 `weights/`、`.runtime/`）和任务目录挂进去，挂载点要和 SAM3 服务看到的路径相同。

### 3.3 不用镜像

```bash
python3.12 -m venv /opt/shape-cpu
/opt/shape-cpu/bin/pip install -r deploy/standalone/requirements.worker.txt
apt-get install -y fontconfig libnss3 libcairo2 ca-certificates
```

这四个系统包在两台验证机器的镜像里都已预装，所以这份清单是按渲染器的动态库依赖推出来的，没有在缺少它们的系统上验证过是否完整。

环境可以建在别的位置，通过环境变量 `PNG2PPTX_CPU_ENV` 指定前缀，默认值是 `/opt/shape-cpu`。

环境的 `bin` 目录必须在 `PATH` 上：渲染器包装脚本的首行是 `#!/usr/bin/env python3`，找不到 `python3` 时 worker 会在启动约 1 分钟后以退出码 127 失败。`run_job.sh` 已经处理了这一点。

## 4. 运行一次转换

```bash
export PNG2PPTX_API_BASE_URL=https://<网关>/api/v1
export PNG2PPTX_API_KEY_FILE=/path/to/api_key        # 权限 0600
export PNG2PPTX_SAM_URL=http://127.0.0.1:18103
export PNG2PPTX_IMAGE_EDIT_URL=http://127.0.0.1:18300/v1   # 图片转接层；自建 SenseNova 时为 http://127.0.0.1:8091/v1
# 模型、推理强度、输出上限、并行数都有默认值（gpt-6-luna、medium、32000、6），见下表

bash deploy/standalone/run_job.sh /data/jobs/123/images /data/jobs/123/run
```

- 第一个参数是放 PNG 的目录，目录里每张 PNG 都会被转换，文件名（去掉扩展名）作为结果的名字。图片是链接进任务的，任务结束前不能移动或删除。
- 第二个参数是任务目录，必须是不存在的新路径。脚本还会在旁边建一个 `<任务目录>.input`。
- 脚本退出码为 0 表示全部成功。

结果在 `<任务目录>/delivery/`：

| 文件 | 内容 |
| --- | --- |
| `<名字>/combined.pptx` | 可编辑的 PPTX |
| `<名字>/preview.png` | PPTX 的渲染图 |
| `<名字>/comparison.png` | 原图与渲染图的对比 |
| `<名字>/validation.json` | 这张图的校验记录 |
| `editable_pptx.zip`、`comparison.pdf` | 全部结果的打包 |
| `validation.json` | 整个任务的汇总 |

进度和最终状态写在 `<任务目录>/status.json` 的 `state` 字段：运行中为 `recovering_full_ppt` 等，结束后为 `completed` 或 `failed`。各阶段日志在 `<任务目录>/logs/`。

**失败后重试**：环境类的失败（例如依赖版本问题）修好后，在原任务目录上重新启动 worker 即可，已完成的分支会被复用。**模型回复本身有问题导致的失败不适用**：出错的回复缓存在任务目录里，重跑会原样复用并立刻再次失败；要重新请求模型，需先移走对应的缓存目录（文字分支是 `<任务目录>/texts/cases/<图片名>/`）。实测一次合成阶段失败后的重跑只用了 2 分半，没有重新调用大模型。命令是 `run_job.sh` 里的最后一步：

```bash
cd <任务目录> && "$PNG2PPTX_CPU_ENV/bin/python" -B -u code/full_recovery/api_worker.py --run-dir <任务目录>
```

### 可选环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PNG2PPTX_API_MODEL` | `openai/gpt-6-luna` | 模型名 |
| `PNG2PPTX_REASONING_EFFORT` | `medium` | `low`、`medium`、`high`，或 `none` 表示沿用原配置 |
| `PNG2PPTX_ARROW_REASONING_EFFORT` | 不设，即与上一项相同 | 只作用于箭头分支的推理强度。`Sage-S2.1-Flash` 只接受 `low` 和 `high` |
| `PNG2PPTX_ARROW_MAX_TOKENS` | 32000 | 箭头分支单次请求的输出上限；设为空串则用后端自带的 113000 |
| `PNG2PPTX_API_MAX_TOKENS` | 32000 | 所有请求的输出上限，避免个别请求推理过长；设为空串则用各后端自带的值 |
| `PNG2PPTX_CASE_WORKERS` | 6 | 图形分支同时处理几张图。六张图的任务从 2 改为 6 后总用时从 40 分钟降到 20 分钟 |
| `PNG2PPTX_TEXT_WORKERS` | 6 | 文字分支同时处理几张图 |
| `PNG2PPTX_API_IDLE_TIMEOUT_SECONDS` | 300 | 模型流多久没有输出就放弃。`gpt-6-luna` 推理时长时间不出字，**不要**调到 60 |
| `PNG2PPTX_API_HEDGE_AFTER_SECONDS` | 不设，即关闭 | 请求超过这么久没答完就再问一次，取先答完的 |
| `PNG2PPTX_CPU_ENV` | `/opt/shape-cpu` | worker 的 Python 环境前缀 |

## 5. 常驻服务

### 5.1 大模型

worker 需要一个 OpenAI 兼容的 `/chat/completions` 接口，支持图片输入、thinking 和 `reasoning_effort`，单次输出上限要能设到 113000 token。

当前选定的模型是 `openai/gpt-6-luna`（`run_job.sh` 的默认值），它和此前用过的 `Sage-S2.1-Flash` 一样，在验证用的网关上直接支持 `/chat/completions`，把 `PNG2PPTX_API_BASE_URL` 指向网关即可，不需要任何转接。

此前验证用的 `z-ai/glm-5.3-flash` 在同一网关上只能通过 Anthropic 风格的 `/messages` 调用，为此仓库里带了转接层 `chat_to_messages_adapter/`（`start_adapter.sh`，端口 18200）。它只在模型不支持 `/chat/completions` 时才需要；它不保存密钥，只把请求里的 `Authorization` 原样转发，不记录请求内容。

**推理强度的默认值是 `medium`**。用 `gpt-6-luna` 时 `medium` 比 `high` 快一倍多，画面没有变差（第 6 节最后一小节）。不要用 `none`：原配置在箭头阶段不设推理强度，用 GLM 时表现为不封顶，大量调用思考到撞上输出上限后重试，五张图 162 分钟、箭头阶段输出 3510 万 token；强制 `high` 后是 58 分钟、690 万 token，质量大体相当。

### 5.2 图片编辑

worker 以 JSON 调用 `<PNG2PPTX_IMAGE_EDIT_URL>/images/edits`，请求不带密钥。调用量很小：六张图的任务一共只调用了 3 次。有两种接法。

**用外部图片模型（推荐）**。仓库里的 `images_edits_adapter/` 把这个接口转成对图片输出模型的 `/chat/completions` 调用，当前选定的模型是 `google/gemini-3.1-flash-image`：

```bash
PNG2PPTX_IMAGE_UPSTREAM_BASE_URL=https://<网关>/api/v1 \
PNG2PPTX_IMAGE_MODEL=google/gemini-3.1-flash-image \
PNG2PPTX_IMAGE_KEY_FILE=/path/to/api_key \
PNG2PPTX_IMAGE_USAGE_LOG=/data/png2pptx-logs/image_usage.jsonl \
  bash deploy/standalone/start_image_adapter.sh        # 端口 18300，前台运行
```

- 这个转接层**持有密钥**（从密钥文件读取，因为 worker 的请求不带密钥），只监听本机，不要暴露到外网。
- 它不记录提示词和图片，每次调用往用量日志里追加一行 token 数，供第 7 节的成本统计使用。
- 上游偶发失败或没有返回图片时会重试一次。
- 请求里的模型名会被忽略，一律使用 `PNG2PPTX_IMAGE_MODEL`。

**自建 SenseNova**。按 5.3 搭环境并启动，把 `PNG2PPTX_IMAGE_EDIT_URL` 指向 `http://127.0.0.1:8091/v1`。需要 62 GB 显存、约 42 GiB 权重和 3.2 GB 的 `runtime_overlay/`。

### 5.3 SAM3 和自建 SenseNova

只用外部图片模型时，下面只需要 SAM3 的部分：搭建环境时加上 `PNG2PPTX_SKIP_SENSENOVA=1`，脚本只建 SAM3 环境；启动时只需要 `start_sam.sh`。

原 `README.md` 第 2、3 节描述的是原集群的做法（挂载现成的 conda 环境、用 `deploy/submit.sh` 提交），集群外不适用。下面的脚本替代它们，都在 `deploy/standalone/` 下。

**前提**：CPython 3.12；CUDA 工具包装在 `/usr/local/cuda`（SenseNova 启动时会检查 `nvcc`），或用 `CUDA_HOME` 指定；能访问 PyPI 和 `download.pytorch.org`。

**搭建环境**（一次性，实测约 15 分钟，主要是下载）：

```bash
PNG2PPTX_BASE_PYTHON=/path/to/python3.12 bash deploy/standalone/setup_gpu_envs.sh /data/png2pptx-envs
```

它建三样东西：

| 位置 | 内容 | 大小 |
| --- | --- | --- |
| `<环境目录>/sam3` | SAM3 环境，依赖锁 `requirements.sam3.txt`（torch 2.12.1+cu130） | 约 5 GB |
| `<环境目录>/openrlhf` | SenseNova 基础环境，依赖锁 `requirements.sensenova-base.txt`（torch 2.9.1+cu128） | 约 11 GB |
| `<仓库>/.runtime/sensenova` | SenseNova 运行时，由基础环境加 `runtime_overlay/` 组合而成 | 约 3 GB |

脚本可以重复执行，中途失败后重跑会从失败的那一步重做。两点说明：

- SenseNova 基础环境是**不解析依赖**安装的。原环境记录的版本组合本身不满足包之间声明的依赖（例如 `vllm 0.16.0` 要求 `transformers<5`，实际运行的是 5.2.0），按常规方式安装会直接报冲突。
- `.runtime/sensenova` 里是指向 `<环境目录>/openrlhf` 的链接，所以这个目录建好后不能移动。

**启动服务**（三个脚本都在前台运行，交给 systemd、supervisor 或容器去托管）：

```bash
# SAM3，端口 18103，约 1 分钟就绪
PNG2PPTX_SAM_PYTHON=/data/png2pptx-envs/sam3/bin/python \
  bash deploy/standalone/start_sam.sh /data/png2pptx-runs/sam

# SenseNova，端口 8091，约 5 分钟就绪
PNG2PPTX_SENSENOVA_BASE_ENV=/data/png2pptx-envs/openrlhf \
  bash deploy/standalone/start_sensenova.sh /data/png2pptx-runs/sensenova

# 转接层，端口 18200，仅在需要时启动（见 5.1）
PNG2PPTX_UPSTREAM_BASE_URL=https://<网关>/api/v1 \
  bash deploy/standalone/start_adapter.sh
```

每个脚本的参数是一个**不存在的新目录**，用来放这次运行的配置和日志；重启服务时换一个目录，或先删掉旧的。

启动脚本相对原配置做了三处调整：

- **SenseNova 改为单卡**：原配置把理解和生成分在 2 张卡上，脚本改成 `--x2i_server_deploy_mode colocate`。
- **关掉空闲保活**：原配置在服务空闲时会主动制造 GPU 负载（原集群用来避免资源被回收），脚本把 `idle_resume_seconds` 调到极大值关掉了它。
- **SenseNova 启动前先清理**：结束上一次残留的进程，释放无人挂接的 SysV 共享内存段。

关于最后一点：SenseNova 运行时占用约 54 GiB 的 SysV 共享内存，容器内存上限为 120 GiB 时够用。但它异常退出或启动失败后，会残留子进程和那段共享内存。**在残留没清掉的情况下再启动，新进程会因为内存超限被系统杀掉，表现为加载到一半卡住、日志里没有任何报错**。`start_sensenova.sh` 每次启动前都会清理，所以重启请始终通过这个脚本。

**SAM3 的显存**（2026-11-27 起）。PyTorch 默认把服务过的最大一次请求所占的显存留作缓存。在 32 GB 的卡上，未加限制时服务在 20 秒内从 12 GB 涨到 28.8 GB，之后两个半小时、约 7000 次请求一直停在那里，直到顶到上限才自己回落，全程没有报错。真实需要的远没有这么多，只是看不出来。现在服务在每次请求结束后（仍持有推理锁时）检查，闲置缓存超过上限就归还显卡，并在 `/health` 里报告实际用量：

| `/health` 字段 | 含义 | 21 张图一次任务的实测 |
| --- | --- | --- |
| `gpu_memory.in_use_mib` | 模型常驻占用 | 3473 |
| `gpu_memory.reserved_mib` | 进程当前持有 | 3580 到 5504 |
| `gpu_memory.request_peak_mib` | 上一次请求的峰值 | 约 3940 |
| `gpu_peak_mib` | 启动以来单次请求的最高峰值 | 8412 |

同一次任务里 `nvidia-smi` 看到的整卡占用平均 7.1 GB、最高 12.4 GB（比进程持有的数字多出约 2 GB，是 PyTorch 统计之外的固定开销）。SAM 单次请求耗时没有变化（中位 0.32 秒，平均 0.41 秒，与改动前相同）。

`runs/<目录>/config.json` 里有两个可选项，改后重启服务生效：

| 配置项 | 默认 | 作用 |
| --- | --- | --- |
| `gpu_cache_limit_mib` | 2048 | 闲置缓存超过这个值就归还 |
| `gpu_memory_limit_mib` | 不设 | 进程显存硬上限；超过时请求直接报错，而不是把整张卡占满 |

硬上限默认不开。只在 21 张测试图上量过峰值（8.4 GB），更大或元素更多的图可能更高；要设的话建议不低于 12000，并留意 `gpu_peak_mib`。

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:18103/health     # 应有 "phase": "ready"、"warmup_passed": true、"checkpoint_hash_verified": true
curl --noproxy '*' http://127.0.0.1:8091/health      # 返回 200
curl --noproxy '*' http://127.0.0.1:8091/v1/models   # 模型 ID 为 SenseNova-U1.5-8B-MoT
```

两个 GPU 环境没有做成镜像。

## 6. 验收

按顺序做，前两项不需要 GPU 和大模型：

```bash
export PATH=$PNG2PPTX_CPU_ENV/bin:$PATH
python scripts/check_prepare.py    # 任务准备，输出 "passed": true
python scripts/check_render.py     # 两套渲染器，输出里没有失败项
```

然后用 `run_job.sh` 跑真实图片。**只有跑完整流程才算验收**：验证过程中有两次环境问题都是在最后的合成阶段才暴露的，前面的自检查不出来。

### 演练记录（2026-11-25，全新实例）

在一台没有任何历史环境的 A800 80GB 实例上，按本文从头做了一遍，结果如下（GLM，强制 high）：

| 步骤 | 结果 | 耗时 |
| --- | --- | --- |
| 从 GitHub 克隆本分支，放入第 2 节的三类大文件 | 通过 | |
| 建 CPU 环境，`check_prepare.py`、`check_render.py` | 通过 | 几分钟 |
| `setup_gpu_envs.sh` 从零建两个 GPU 环境 | 通过 | 约 15 分钟 |
| 启动 SAM3、SenseNova、转接层，健康检查 | 通过 | SenseNova 约 5 分钟 |
| `run_job.sh` 转换一张简单流程图 | 导出 1 张 | 2 分 17 秒 |
| `run_job.sh` 转换六张图（五张顶会论文图加一张带三维曲面的示意图，同一任务） | 导出 6 张，0 失败 | 108 分钟 |

六张图的任务共调用大模型 1909 次，没有失败和限流，同时在途的请求最多 48 个。各张图完成的时间点（从任务启动算起）依次是第 26、41、47、51、52、108 分钟：前五张集中在前 52 分钟，最后一张带三维曲面的图独占了后面近一个小时，时间主要花在箭头分支上。此前在第一台机器上，五张论文图单独一个任务是 58 分钟，带三维曲面的那张单独一个任务是 51 分钟，两者相加与这次接近，说明**把多张图放进同一个任务并不会明显缩短总耗时**。

演练中发现的问题都已修进本分支，列在第 8 节。

### 换用 Sage 和 Gemini 后的验证（2026-11-25）

大模型换成 `Sage-S2.1-Flash`（直连 `/chat/completions`）、图片编辑换成 `google/gemini-3.1-flash-image`（经图片转接层）后，在同一台机器上重跑，SenseNova 不参与：

| 任务 | 结果 | 耗时 | 对比 GLM |
| --- | --- | --- | --- |
| 简单流程图 1 张 | 导出 1 张 | 6 分 36 秒 | 2 分 17 秒 |
| 六张图同一任务 | **失败**，见下 | 第 6 分钟失败 | 108 分钟，6 张 |
| 去掉 `acl23_xray` 后的五张图 | 导出 5 张，0 失败 | 244 分钟 | 同样五张约 108 分钟 |

- **Sage 在 `acl23_xray` 上稳定失败（已修复，见下一小节）**。这张图有 77 段文字和一个多段上下标公式，Sage 的文字清单 JSON 在公式处漏写键名（`{", y", "script": ...}` 少了 `"text":`），四次请求全部如此。文字分支只重试一次，失败后整个任务失败，同任务的其他图也拿不到结果。GLM 在这张图上没有这个问题。
- **Sage 慢约一倍**，个别请求单次长达 20 到 33 分钟。五张图的 1361 次调用里有 13 次撞输出上限、6 次失败，流程自行重试后没有影响结果。
- **还原质量与 GLM 大体相当**（看过三张图的对比）：各有细节出入，没有系统性的优劣。
- Gemini 图片编辑在五张图的任务里被调用 3 次，全部成功，每次约 9 秒。

### 文字清单容错和箭头分支提速（2026-11-25，`acl23_xray` 一张图）

- **文字清单容错**：`full_recovery/text_json_repair.py` 在 Sage 漏写 `"text":` 键名时把键名补回去，值不变；其他格式错误照旧失败。加上后这张图的文字分支正常完成（补回 12 处）。
- **箭头分支为什么慢**：箭头分支对每个候选区域单独问一次模型。全程 `high` 时，五张图任务的输出 token 约有三分之二花在箭头分支；一次请求思考了 57,824 个 token、19 分钟，回答只是“这是容器边框，不是箭头”。
- **箭头分支单独设推理强度**：`PNG2PPTX_ARROW_REASONING_EFFORT=low` 只降低箭头分支的强度，图形和文字分支仍是 `high`。实现方式是箭头分支走本地网关的 `/arrow/v1` 路由，网关只对这条路由改写强度和输出上限（`full_recovery/arrow_budget.py`）。
- **箭头回答的格式修复**：低强度下 Sage 有约四分之一的箭头回答把节点编号写错（不带引号，或只写坐标不写编号），原后端会整条丢弃。`full_recovery/arrow_reply_repair.py` 只恢复编号、不改坐标，准备任务时装入该任务自己的箭头代码副本。

| `acl23_xray`，Sage | 用时 | 输出 token | 费用 | 箭头几何任务失败 | 画面 |
| --- | --- | --- | --- | --- | --- |
| 全程 `high` | 167 分钟 | 163 万 | $0.51 | 1 / 52 | 箭头基本齐全 |
| 箭头 `low`，无格式修复 | 37 分钟 | 59 万 | $0.33 | 12 / 45 | 少十来条箭头 |
| 箭头 `low`，有格式修复 | 39 分钟 | 60 万 | $0.32 | 0 / 53 | 箭头基本齐全，与全程 `high` 相当 |

只在这一张图上对比过，其余图片未验证；每行各跑一次，同一配置两次运行之间本身也有出入。Sage 不接受 `medium`（网关返回上游 400）。

### 换用 gpt-6-luna 和提速（2026-11-26，六张图同一任务）

下表最后一行的配置（`openai/gpt-6-luna`、`medium`、两个输出上限 32000、并行数 6）现在是 `run_job.sh` 的默认值。模型价格按输入 $0.1、输出 $0.5 每百万 token 计。

| 配置 | 总用时 | 导出 | 模型费用 |
| --- | --- | --- | --- |
| `high`，一次 2 张 | 83.7 分钟 | 6（带曲面的那张丢了底图） | $3.11 |
| `medium`，一次 2 张 | 39.6 分钟 | 6 | $1.98 |
| `medium`，六张并行 | 20.1 分钟 | 6 | $2.03 |

最后一行各张图导出的时间点为第 5.5、8.9、12.6、13.2、14.8、19.8 分钟；最慢的是带三维曲面的示意图，它单独跑约 19 分钟。1877 次请求没有失败。每种配置只完整跑成一到两次。

这一轮的改动：

- **箭头分支的队列宽度**原来固定为 8，现在随网关并发数放宽（并发 48 时为 24）。
- **箭头定位复核**：模型漏写 `label` 时用同一条目的 `reason` 补上，不再整条作废重做；复核失败后拆成的两半改为同时进行。
- **`medium` 强度**：文字分支原先不接受这个取值，已放开。`medium` 比 `high` 快一倍多，六张图的画面没有变差，其中两张更好。
- **带透明背景描述的位图素材**：给图片模型的指令里明确要求白底，否则图片模型会画出棋盘格，素材被整块丢弃。
- **位图底图上的小图标**：没有指定遮挡对象的纹理底图改为插在它盖住的图形之下；被底图盖住的封闭路径也计入前景。
- **上游短时故障**：5xx 和网络错误的重试由 3 次（间隔 1、2 秒）改为 5 次（间隔 2、4、8、16 秒）。
- **模型服务方拒绝的请求**：网关把服务方的 4xx 包成 `502 upstream error 4xx`，重试没有用。现在识别后不重试，以 422 返回；图形分支遇到被拒绝的复核时保留已拟合的图形并标为未复核，不再让整张图失败。实测遇到过的一例是一张白底洋红色圆角矩形的示意图被服务方稳定拒绝（403），原因不明；这条降级路径只有单元测试，还没有在真实任务里触发过。

仍然存在的问题：同一配置两次运行的结果不完全一样，例如同一张图的面板边框三次出了三种样子，曲面示意图上的小图标有一次只还原了 5 个中的 3 个。

### 自动评分和三项通用机制（2026-11-27，21 张图同一任务）

评分工具 `scripts/score_delivery.py RUN_DIR [RUN_DIR ...]`：不调用模型，把 `delivery/<图>/preview.png` 和原图逐像素比较，输出 recall（原图轮廓找得到的比例）、precision（还原图轮廓有依据的比例）、colour、defects（局部明显不对的小块占比）和总分。它不识字，文字认错基本不扣分；对箭头虚实不敏感。

合成阶段新增三项机制，都不调用模型，同一输入每次结果相同：

| 机制 | 代码 | 关闭开关（任务 `settings.json`） |
| --- | --- | --- |
| 箭头颜色、线宽、虚实从原图像素测量。箭头分支的样式阶段本来是关闭的，此前所有箭头都是灰色 2px 实线 | `full_recovery/stroke_measure.py`、`arrow_stroke_style.py`、`arrow_restyle.py` | `arrow_stroke_from_pixels: false` |
| 背景角色的位图插到最后一个完全盖住它的不透明图形之上，不再被面板或黑框盖住 | `full_recovery/texture_layer.py` | 无 |
| 合成后与原图比较，原图有内容而幻灯片 6 像素内没有任何内容的区域，用原图裁块补成图片（名称 `RESIDUAL n`，不可编辑） | `full_recovery/residual.py` | `source_residual_pictures: false` |

21 张图（旧 6 张加 15 张新图）同一任务，默认配置：70 分钟，21/21 导出，$8.70。与改动前各自的基线相比，平均总分 0.893 → 0.912，最低 0.628 → 0.700。机制实际触发的图提升明显（一张漏掉整栏小字的图 0.667 → 0.966）；另有 8 张下降 0.01–0.07，与同一配置重复运行的波动幅度（0.02–0.06）相当，逐张查看是图标、文字等模型波动，不是这三项改动引起的。单次运行不足以分辨 0.03 以内的差别。

用时随图片数近似线性增长：图形分支在配色前要对每条线做 SAM 小块分割（每张图 170–720 次请求，每次 0.5–1.6 秒），全部串行走同一个 SAM 服务。

### 只重放合成阶段，比较一项改动的净效果（2026-11-27）

同一配置重复跑完整任务，单张图的分数会波动 0.02–0.06，小于这个幅度的改动用两次完整任务分辨不出来。合成阶段的改动可以这样比较：

```bash
cd png2pptx
python -m full_recovery.replay_composition ../jobs/<已完成的任务> ../jobs/<新目录> [--set 开关=值 ...]
python -B -u ../jobs/<新目录>/code/full_recovery/api_worker.py --run-dir ../jobs/<新目录>
python -m full_recovery.replay_composition ../jobs/<新目录> --report    # 复用了多少请求，新问了多少
python scripts/score_delivery.py ../jobs/<新目录>
```

它绑定已完成任务的三个分支产物（只读），用仓库当前的合成代码重新合成；内容完全相同的模型请求直接取那次任务记录的回复，不同的请求才会真的去问模型。需要大模型网关的上游和图片编辑服务可用，同一台机器同时只能跑一个。范围只到合成阶段：图形、文字、箭头三个分支内部的改动不在其中。

在 21 张图上的结果：6 分钟一轮；同设置重放 359 个请求复用 340 个，逐图分数与原任务最大相差 0.003。用它量三项机制中两项的净效果（关掉该项重放，与全开的重放相比）：

| 关掉的机制 | 平均分 | 最低分 | 变好超过 0.01 的图 | 变差超过 0.01 的图 |
| --- | --- | --- | --- | --- |
| 原图裁块兜底 | 0.891 → 0.912 | 0.614 → 0.697 | 5 | 0 |
| 箭头像素测量 | 0.906 → 0.912 | 0.645 → 0.697 | 3 | 0 |

关掉箭头像素测量时有 88 个请求内容变了（箭头外观进了后续请求的图片），这一轮带有模型波动，0.01 以内的差别不作数。

### 稳定性修复和用时分析（2026-11-27，RTX 5090 32GB，21 张图）

这一轮换了机器（A800 80GB → RTX 5090 32GB，路径布局不变），模型和默认设置同上一节。

**修掉的四个问题**

| 问题 | 表现 | 处理 |
| --- | --- | --- |
| 文字被误加粗 | 文字属性识别两次：整图一次，再把每个小块放大后重读，以第二次为准。放大约 8 倍后笔画看着都粗，111 个文字被改成加粗，其中只有 17% 的像素笔画粗细像粗体 | 第二次识别给出的单一粗细与第一次矛盾时，保留第一次的；第二次在同一标签内区分出粗细不同的片段时照旧采纳（`full_recovery/text_weight.py`） |
| 配色回退后线宽为 0 | 配色修正被部分采纳时按字段回退。给原本没有描边的部件新加描边、而评审只对线宽有异议时，线宽退回旧值 0，描边却留着。以往任务里出现过 33 次，画成极细的线；落在虚线上时除以 0，整张图失败 | 描边的有无一旦被认可，颜色、线宽、线型跟着保留；线宽为 0 一律按无描边写 |
| 重复元素同名 | 模型把两排相同的小方块起了同样的名字，校验拒绝整份回复，连续两次后整张图失败 | 重名的元素自动加序号区分，位置完全相同的才算重复（`src/shape_integration/reply_ids.py`） |
| SAM 显存不可控 | 见 5.3 | 见 5.3 |

加粗修复的效果，用三次 21 张任务核对：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 第二次识别改掉加粗的文字数 | 137 | 0 到 6 |
| 最终加粗与像素笔画粗细一致的比例 | 0.835 | 0.911、0.919 |
| 两次运行同一文字加粗不一致的比例 | 11.1% 到 13.6% | 3.9% 到 5.8% |
| 两次识别都判加粗（或都判不加粗）却被改动的 | — | 0 |

像素笔画粗细的量法：文字框内笔画的平均宽度除以字号，在两次识别结论一致的 646 个文字上，单一阈值能分对 91%。它只用来裁决对错，没有进入流程。

**三次 21 张任务**

| 任务 | 成功 | 用时 | 费用 | 均值 | 最低 |
| --- | --- | --- | --- | --- | --- |
| 上一节的任务（A800） | 21/21 | 70.2 分钟 | $8.70 | 0.912 | 0.700 |
| 含加粗修复 | 19/21，另两张即上表第二、三行的失败，修复后重跑为 0.910、0.978 | 65.8 分钟 | $8.18 | 0.898 | 0.711 |
| 含全部修复 | 21/21 | 77.5 分钟 | $8.95 | 0.899 | 0.617 |

最后一行的分数是用最终代码重放合成得到的；那次任务实际运行时带着一版后来撤回的兜底改动（见第 7 节），当时为 0.904。

均值在 0.90 到 0.91 之间波动，没有随修复上升：这几项修复针对的是失败和文字属性，评分不识字，也不为失败的图计分。同一张图在不同任务里的分数差别不小，复杂图尤其明显（`image10` 三次为 0.839、0.937、0.905；`image3` 为 0.889、0.883、0.768）。差别来自图形和位图分支每次识别到的内容不同，例如某次漏掉整块底板或整张照片。

**每张图的时间花在哪里**

数据来自原始记录而不是文件时间：网关的逐条用量记录（按 token 用量与请求目录一一对应，7313 条中 7311 条对上）、SAM 客户端逐次记录与服务端日志（图形分支 4981 次成功、55 次"忙"，两边完全一致）。每张图各区间之和与实际导出时刻的差在 0.1 分钟以内。

| 每张图平均（分钟） | A800 | RTX 5090 |
| --- | --- | --- |
| 排队等开工 | 23.2 | 18.6 |
| 关键分支里有模型请求在途 | 9.4 | 8.6 |
| 有 SAM 请求在途 | 1.5 | 1.3 |
| 两者都不在途（本机计算） | 3.6 | 2.5 |
| 合成 | 2.4 | 2.4 |
| 导出时刻 | 40.1 | 33.3 |

- 决定一张图何时完成的分支：图形 13 张、箭头 6 张、文字 0 张。
- **SAM 不是瓶颈**，换显卡对用时没有可见影响。
- 排队占了一半以上。同时处理的图片数被限制住了：图形和文字分支最多 6 张，**箭头分支最多 2 张**，最后一张图的箭头到第 51 分钟才开工。网关允许 48 个并发，平均只有 28 个在途。
- 一张图自己要多久，由图形分支里首尾相接的几个环节决定（定位 → 轮廓拟合 → SAM 取样 → 配色 → 合成），机器空闲也不会更快。四张图各单独跑一次：

  | 图 | 元素数 | 单跑 | 在 21 张任务里自身用时 |
  | --- | --- | --- | --- |
  | `image15` | 57 | 8.8 分钟 | 9.5 |
  | `image7` | 169 | 16.8 | 15.2 |
  | `image3` | 142 | 19.5 | 25.3 |
  | `image10` | 498 | 31.6 | 31.7 |

  这四张同批跑用了 40.1 分钟，`image10` 那次遇到 26 个超过 60 秒的模型请求（最慢 263 秒）。同一张图的用时在不同次之间差 10% 到 25%。
- 用时与元素总数（线条、图形、文字块、位图）的关系：约 8.8 分钟固定开销，每 100 个元素再加约 5 分钟，能解释 68% 的差别，留一法误差约 4 分钟。图片像素大小几乎无关。

**还能快多少（估算，没有改）**

| 假设 | 21 张总用时 |
| --- | --- |
| 现状 | 66 到 78 分钟 |
| 放开同时处理的图片数，让 48 个并发一直满载 | 约 38 到 42 分钟（全部请求耗时合计 30.8 小时，除以 48） |
| 并发上限再翻倍 | 约 32 到 35 分钟，此时受限于最慢那张图自身的用时 |
| 单张图内部，同一环节里互不依赖的对象同时发请求 | 图形分支的模型环节可再减约四成，是按请求目录分组得到的粗估 |

前两行取决于上游能否长时间承受满载并发，这一点没有验证过。

分析用的脚本没有入库，在演练机的 `rehearsal/` 下：`join_usage.py`、`timeline.py`、`branch_spans.py`、`chain_floor.py`、`sam_account.py`。

## 7. 已知限制

接入前请让产品方知情：

- **线条样式不还原**：线条和箭头的颜色、粗细、虚实都不会被还原，输出一律是细实线。这是流程本身的限制，与模型无关。
- **复杂插图会丢内容**：叠在位图上的小图标、手绘风格的路径容易丢失。
- **耗时长且波动大**：用 GLM 时单张 2 到 50 分钟；用 Sage 时单张 7 分钟到 3 小时以上，箭头分支设为 `low` 后实测一张复杂图从 167 分钟降到 39 分钟（见第 6 节）。接口必须做成异步（提交、查询、取结果）。
- **一张图失败会连带同任务的其他图**：上线后建议一张图一个任务。已知的 Sage 格式错误有容错（见第 6 节），但只覆盖见过的几种写法，遇到新的写法仍会失败，失败的图只能换模型重跑。
- **成本**（Sage 输入 $0.05、输出 $0.2 每百万 token；Gemini 图片模型输入 $0.5、输出 $3.0 每百万 token）：

  | 任务 | 大模型调用 | 输入 token | 输出 token | 大模型费用 | 图片编辑 | 合计 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 简单流程图 1 张 | 55 次 | 31 万 | 2 万 | $0.019 | 0 次 | $0.02 |
  | 五张复杂图 | 1361 次 | 827 万 | 385 万 | $1.184 | 3 次，$0.012 | $1.20，平均 $0.24/张 |

  图片编辑每次约 $0.004，可以忽略。失败的那次六张图任务另外花了 $0.20。任务结束后用下面的命令统计，图片用量取自图片转接层的用量日志，按任务的起止时间截取，所以同一时段只有一个任务在跑时才准确：

  ```bash
  python scripts/estimate_cost.py <任务目录> --image-usage-log <用量日志> \
    --chat-input-price 0.05 --chat-output-price 0.2 --image-input-price 0.5 --image-output-price 3.0
  ```
- **结果每次不完全一样**：同一张图重跑，评分可以差 0.05 到 0.1。图形分支偶尔漏掉浅色底块或整块底板，位图分支偶尔漏掉整张照片（21 张里的 `image13` 三次中出现两次）。原图裁块兜底能补回其中一部分：缺失处在幻灯片上还留着少量碎片时，按现在的规则不补。试过改成按缺失物自身轮廓判定，三次任务的均值各高 0.005 到 0.012，但会把行尾位置略偏的文字当成缺失一起补上，叠出重影，所以没有采用；要用，先得把"缺失"和"位置偏了"分开。
- **文字**：认错字、同一文字两次运行颜色不一致（约 15%）仍然存在；更早的"逐行核对"步骤每次还会改动十几到三十个文字的加粗。评分不识字，这些在分数上看不出来。
- **批量任务里靠后的图要排队**：见第 6 节的用时分析，21 张图的任务里每张图平均有一半以上的时间在等名额。
- **并发**：只验证过单个 worker 内最多六张图的任务，没有验证过多个 worker 同时访问同一组 GPU 服务。
- **结果需要人工看**：任务状态为 `completed` 只表示流程走完并导出了文件，各分支日志里常见 `needs_review`、`completed_with_unresolved` 这类标记，还原质量要看 `comparison.png`。

## 8. 本分支相对原包的改动

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `full_recovery/prepare.py` | 图形阶段的类别表改从任务目录内的清单读取 | 原来读一个只存在于原集群的数据集目录 |
| `full_recovery/prepare.py`、`full_recovery/api_worker.py` | CPU 环境路径可用 `PNG2PPTX_CPU_ENV` 指定；复制渲染器启动脚本时按该路径改写首行 | 原来写死 `/opt/shape-cpu` |
| `full_recovery/api_worker.py`、`inplace_resume.py`、`split_coordinator.py`，`scripts/arrow_renderer_wrapper.py`、`fresh_sam_worker.py` | 写 `/proc/self/oom_score_adj` 失败时忽略 | 多数容器禁止写这个文件，原代码会直接崩溃 |
| `scripts/check_prepare.py` | 增加对上面两项路径的断言 | |
| `chat_to_messages_adapter/`、`tests/test_chat_to_messages_*.py` | 新增 | 见 5.1 |
| `images_edits_adapter/`、`tests/test_images_edits_adapter.py`、`scripts/estimate_cost.py` | 新增 | 见 5.2 和第 7 节 |
| `full_recovery/text_json_repair.py`、`full_recovery/text_stage.py`、`tests/test_text_json_repair.py` | 新增容错并在文字阶段启用 | 见第 6 节 |
| `full_recovery/arrow_budget.py`、`api_gateway.py`、`prepare_api.py`、`run_ppt.py`、`tests/test_arrow_budget.py` | 箭头分支可单独设推理强度和输出上限；箭头分支改走网关的 `/arrow/v1` 路由 | 见第 6 节。不设置时请求内容与原来相同 |
| `full_recovery/arrow_reply_repair.py`、`full_recovery/fresh_arrow.py`、`tests/test_arrow_reply_repair.py` | 准备任务时给箭头后端的解析函数加一层节点编号修复 | 见第 6 节。始终启用 |
| `full_recovery/arrow_workers.py`、`texture_layer.py`、`foreground_candidates.py`、`transport_retry.py`、`request_limits.py`、`request_hedge.py`、`text_parallel.py` 及对应测试 | 新增 | 见第 6 节最后一小节 |
| `full_recovery/api_gateway.py`、`image_edit.py`、`image_assets.py`、`fresh_arrow.py`，`experiment/shape_sam_guided/request_io.py`，`src/shape_integration/fitting/control.py`、`local.py` | 重试、拒绝识别、白底指令、素材层级 | 同上 |
| `third_party/arrow_geometry/.../localization/code/local_review.py`、`review_label_policy.py` | 复核答案补 `label`，失败拆分并行 | 同上 |
| `deploy/standalone/` | 新增 | 本文用到的脚本和清单 |
| `.gitignore` | `weights/` 改为 `/weights` | 原规则把 SenseNova 源码里 19 个名为 `weights` 的代码目录（99 个 `.py` 文件）也忽略了，仓库因此缺代码，现已补入 |
| `backends/reference/arrow_sam_ckpt_20261015/runs/` | 两个配置文件强制纳入 git | 准备任务时必须读取，原来被 `runs/` 规则忽略 |
| `deploy/sensenova/setup_runtime.py` | 容忍基础环境里几个无功能的缺失条目；不链接被覆盖层替换的包的版本元数据；生成 `hypercorn` 启动命令 | 原脚本只适用于原集群的现成环境，对按清单重建的环境会报错或生成无法启动的运行时 |

`.runtime/arrow_renderer/soffice` 也需要同样的 `oom_score_adj` 容错，但 `.runtime/` 不进 git。拿到原始 `.runtime/` 后，把这个文件开头的

```python
Path('/proc/self/oom_score_adj').write_text('0')
```

改成

```python
try: Path('/proc/self/oom_score_adj').write_text('0')
except OSError: pass
```

没有改到的地方：`scripts/ppt_worker.py`、`scripts/shape_worker.py`、`experiment/` 下的脚本里仍然写着 `/opt/shape-cpu` 和原集群路径，它们属于原集群的本地模型流程，本文的 API 流程不经过。
