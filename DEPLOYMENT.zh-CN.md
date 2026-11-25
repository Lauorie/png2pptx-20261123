# 部署说明（面向后端）

本文说明怎样在原集群之外部署这套 PNG → 可编辑 PPTX 系统。内容来自 2026-11-23 至 11-25 在两台租用 GPU 机器上的实测：第一台用于摸索，第二台是全新实例，严格按本文从 GitHub 拉代码、从零搭环境、启动服务并跑通转换（记录见第 6 节）。凡是没有实测的部分都单独标出。

原作者的 `README.md` 描述的是原集群（rjob、gpfs 挂载）上的用法，本文不依赖那些设施。

## 1. 系统组成

一次转换由一个 CPU worker 进程驱动，它调用三个常驻服务：

| 组件 | 作用 | 资源 | 接口 |
| --- | --- | --- | --- |
| CPU worker | 调度图形、文字、箭头三个分支，合成 PPTX | CPU；实测机器上单任务无 GPU 占用 | 命令行，见第 4 节 |
| 多模态大模型 | 三个分支的识别与推理 | 外部 API | OpenAI 兼容的 `/chat/completions` |
| SAM3 服务 | 定位、轮廓、mask | 1 张 GPU，约 10 GB 显存 | HTTP，端口 18103 |
| SenseNova 服务 | 图片素材的编辑与补全 | 1 张 GPU，约 62 GB 显存 | HTTP，端口 8091 |

两个 GPU 服务可以放在同一张 80 GB 的卡上（实测为 A800 80GB，共占用约 74 GB）。

三条必须遵守的约束：

- **worker 和 SAM3 必须共享文件系统，且路径一致**。SAM3 的请求里传的是图片和输出目录的绝对路径，不是图片内容。
- **一个容器同一时间只跑一个 worker 进程**。worker 会改写 `/usr/bin/libreoffice`，并在 `/tmp/shape-recovery-lo` 下放置渲染器，因此需要 root，且多个 worker 会互相覆盖。要并行就把多张图放进同一个任务（见第 4 节），或者开多个容器。
- **不要用 RTX PRO 6000 Blackwell**。SenseNova 在这张卡上预热时会触发 CUDA 非法访存，之后 GPU 需要宿主机复位才能恢复。

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

`.runtime/sensenova/` 不需要传输，它由第 5.2 节的搭建脚本在本机生成，里面有指向本机环境的绝对路径。

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
export PNG2PPTX_API_BASE_URL=http://127.0.0.1:18200/v1
export PNG2PPTX_API_KEY_FILE=/path/to/api_key        # 权限 0600
export PNG2PPTX_SAM_URL=http://127.0.0.1:18103
export PNG2PPTX_IMAGE_EDIT_URL=http://127.0.0.1:8091/v1

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

**失败后重试**：在原任务目录上重新启动 worker 即可，已完成的分支会被复用。实测一次合成阶段失败后的重跑只用了 2 分半，没有重新调用大模型。命令是 `run_job.sh` 里的最后一步：

```bash
cd <任务目录> && "$PNG2PPTX_CPU_ENV/bin/python" -B -u code/full_recovery/api_worker.py --run-dir <任务目录>
```

### 可选环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PNG2PPTX_API_MODEL` | `z-ai/glm-5.3-flash` | 模型名 |
| `PNG2PPTX_REASONING_EFFORT` | `high` | `low`、`medium`、`high`，或 `none` 表示沿用原配置 |
| `PNG2PPTX_CPU_ENV` | `/opt/shape-cpu` | worker 的 Python 环境前缀 |

## 5. 三个常驻服务

### 5.1 大模型

worker 需要一个 OpenAI 兼容的 `/chat/completions` 接口，支持图片输入、thinking 和 `reasoning_effort`，单次输出上限要能设到 113000 token。

验证时用的是一个第三方网关。它上面的 GLM 只能通过 Anthropic 风格的 `/messages` 调用，所以仓库里带了一个转接层 `chat_to_messages_adapter/`，把 `/chat/completions` 转成 `/messages`：

```bash
python -m chat_to_messages_adapter --upstream-base-url https://<网关>/api/v1 --port 18200
```

转接层不保存密钥，只把请求里的 `Authorization` 原样转发，不记录请求内容。**如果生产环境的接口本身支持 `/chat/completions`，就不需要转接层**，直接把 `PNG2PPTX_API_BASE_URL` 指过去。

**推理强度请保持 `high`**。原配置在箭头阶段不设推理强度，在验证用的网关上表现为不封顶：大量调用思考到撞上输出上限后重试。五张图的对比如下。

| | 原配置 | 强制 high |
| --- | --- | --- |
| 总耗时 | 162 分钟 | 58 分钟 |
| 箭头阶段输出 token | 3510 万 | 690 万 |
| 撞上限的调用 | 104 次 | 7 次 |

两种配置的输出质量大体相当。这个现象只在该网关上验证过，换接口后需要重新确认。

### 5.2 SAM3 和 SenseNova

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

## 7. 已知限制

接入前请让产品方知情：

- **线条样式不还原**：线条和箭头的颜色、粗细、虚实都不会被还原，输出一律是细实线。这是流程本身的限制，与模型无关。
- **复杂插图会丢内容**：叠在位图上的小图标、手绘风格的路径容易丢失。
- **耗时长且波动大**：单张 2 到 50 分钟，接口必须做成异步（提交、查询、取结果）。
- **成本**：箭头阶段每张图约 140 万到 170 万输出 token，其他阶段没有统计。
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
