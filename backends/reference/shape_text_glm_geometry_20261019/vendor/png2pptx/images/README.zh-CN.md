# 图片提取阶段

入口：`python run_images.py` 或 `python -m png2pptx.images`。当前默认流程：全图初检与版面搜索规划 → 程序补齐未覆盖区域 → 原图局部搜索/边界细化 → 未看清区域的子区域检查与整图查漏 → **逐对象独立边界复核** → 扩充每个目标的 bbox → GLM 看干净局部图写编辑 prompt → SenseNova 编辑 → 本地透明化。不调用 RMBG。

模型为 Boyue 接口的 `glm-5.3-flash`，与现有几何实验使用的 GLM-5.3 视觉模型一致，`reasoning_effort=low`。检测单次输出上限 8192 tokens；逐目标写 prompt 上限 1536 tokens；查漏修复最多 2 轮，提前完整则停止。预算节省来自短输出、复用完整候选和精确请求缓存，不再通过删除局部检测来省预算。该接口不允许关闭 thinking；同服务的 `glm-5.3` 拒绝图片，`bailian/glm-5.3` 未实际看到图片，所以使用已验证的视觉接口名，可通过 `--model` 覆盖。

**提示词统一维护在 [prompts.py](prompts.py)**：分类与召回规则、全图检测、局部补检、整图查漏、独立边界复核、GLM 编写编辑指令、SenseNova 固定背景说明、错误纠正指令都在这一个文件。文件开头列出实际调用顺序。`coverage.py`、`boundary.py`、`planning.py`、`clients.py` 只调用对应模板并传入运行数据，不再在各自文件追加 prompt 文本。

实际逐对象编辑指令由 `planning.py` 调用 GLM 看局部像素后编写，包含具体保留项、删除项和完整编辑 prompt，不能使用通用模板代替这次视觉理解。定位框仅由检测阶段负责，写 prompt 时不再重新估计或覆盖 bbox。GLM 判断背景类型与目标是否完整可见；若返回 `needs_context`，最多再扩大一次上下文（每侧至少目标长边的 45%），两次输入/响应分别保留；仍不明确就停止该目标。所有照片场景、图标底牌、浅色细线/分离附件都要求具体识别。

模型生成的编辑 prompt 禁止夹带 x/y 等像素坐标，避免把 2048 的观察坐标误用于 1024 的编辑画布；检测到此类指令会要求模型纠正，不直接发送给 SenseNova。

GLM 按子图、栏目或留白规划少量较大的搜索区域，独立于已有候选；程序按像素覆盖补齐缺口，碎片过多时合并成包围区域检查，缺口不会直接丢弃。局部区域增加 15% 上下文，实际视野必须包含分配目标的 bbox。局部输入直接从原图裁剪，再放到长边 2048，保存原生裁剪和局部→全图→原图变换。

局部响应包含 `search_complete` 和 `zoom_regions`：清楚区域完成检查，未看清区域建立子任务；`needs_context` 为目标建立更宽视野。父区域只有所有子任务完成才完成。没有有效放大提案的未完成区域沿长边二分，适用所有输入。`needs_context` 结合目标 bbox 与上次实际视野向四周扩展 50%；按原图裁剪范围去重，连续请求不会重复发送同一视野。到达整图边界仍不能解决时保留 unresolved。

默认 `--coverage-rounds 2`，允许两层后续搜索/修复，`--max-local-calls 16` 限制每张图局部调用数量（不含单次 schema 纠正重试）。未完成区域和未修正目标都阻止 `model_review_complete`；预算用完时保留候选、子任务及问题。重复目标归属在修改候选前统一验证。空间搜索完成只表示这些区域已按协议检查，不代表图片召回率已经视觉验证。

独立边界复核在 [boundary.py](boundary.py)：全部候选各自从原始分辨率取聚焦局部，每侧增加目标长边 50% 的上下文，至少 8 个原图像素，再放到长边 2048。每次发送两张同尺度图片：干净原图和当前 bbox 叠加图。GLM 仅检查四边、细小附件、自带底牌及外部标注，不修改分类、不增加或删除目标。输出 `accept`、`revise`、`needs_context` 或 `unresolved`；修改框后重画框，再独立复核。原始检测坐标与全部提案保留，源图裁剪框同步更新，避免历史 `source_box_half_open` 覆盖新定位。

默认 `--boundary-review-rounds 2` 表示每个目标最多两次视觉调用（含修改后验证或扩大上下文），`--max-boundary-calls 64` 限制单图总调用，`--boundary-max-tokens 1536` 限制单次输出；schema 纠正最多另试一次。先让每个目标获得首次复核，再处理后续请求。`needs_context` 扩大上次实际视野，已经到全图时不重复发送。预算为零或耗尽、末次修改尚未验证、上下文仍不够均保留 `unresolved`。搜索完成和边界复核分别记录；未完成边界复核的目标保留在清单和定位可视化中，暂不调用编辑服务，在提取报告中显示占位。`accept` 只是模型判断，不是视觉精度真值。

分类按独立对象的可见表达方式与作用判断：由明确图元及其位置、连接、大小、颜色、数值关系组成的结构图和基础数据图交给几何重建，元素多、重复、彩色、小尺寸、普通重叠均不是归为 image 的理由。具象 pictogram、logo、复杂曲线徽标和装饰符号保留整体图片；不能把普通结构图改称 icon 来绕过几何分类。照片、纹理型科学面板、具象插画和轮廓/遮挡确实复杂的三维组合归为 image；简单透视实体和边界清楚的平面层叠仍可重建为 shape。全图不清楚时局部检查，查漏和局部复查使用相同规则，同时检查漏检和误检。不可分离的图像组合整体保留，分开的缩略图分别定位，图表外框、文字和外部连线不并入目标。

写 prompt 时同时提供检测阶段的保留/删除范围及边界备注，区分图标自身底牌和图表的大容器背景。自动管线拒绝带有 `edit_feedback` 或 `previous_edit_path` 的人工反馈输入。历史逐 case 反馈重试只属于诊断实验，不能混入自动评测结果。

统一评测使用冻结代码与统一参数、仅提供原始图片、不导入历史候选、不选择特定目标，也不人工替换失败结果。固定的 schema 纠正重试和模型请求扩大上下文适用于全部目标；预算用尽即保留失败。报告分别显示检测未完成、检测无目标、覆盖未解决和提取失败。

```bash
# 从原始 PNG 开始；包含局部补检与查漏
python run_images.py --image figure.png --output runs/image_extract/example

# 已有候选，跳过搜索检测；仍执行独立边界复核
python run_images.py --image figure.png --candidates previous/inventory.json \
  --output runs/image_extract/reuse

# 复用本项目此前的 74 个检测结果；只导入匹配原图的目标
python run_images.py \
  --image tests/test_image_recovery/runs/fresh_12images_20260920_2032/images/052/original.png \
  --legacy-assets tests/test_image_recovery/runs/fresh_12images_20260920_2032/recovery/assets.json \
  --output runs/image_extract/052

# 把图片阶段接到几何 scene 上；原有 objects/points/paths/issues 保持原样
python run_images.py --scene path/to/geometry/scene.json --scene-root path/to/geometry_run \
  --output runs/image_extract/merged

# 全部 12 张原图、74 个已有目标；包含原图/恢复图交错 PPT 与显式失败占位
python -m png2pptx.images.batch \
  --source tests/test_image_recovery/runs/fresh_12images_20260920_2032/images \
  --legacy-run tests/test_image_recovery/runs/fresh_12images_20260920_2032 \
  --output runs/image_extract/all_images --workers 3
```

检测保留自然照片、科学栅格图、插画、复杂图标、装饰五类。全图、检测局部图和写 prompt 的局部图长边均为 2048；检测模型可规划无已知目标的搜索区域，无固定候选数量上限。每个目标按长边每侧扩充 15%，至少 8 个原图像素，通过 `--context-padding` 调整。局部源像素直接取自原始分辨率，使用 `frames.py` 的 half-pixel 变换。SenseNova 编辑画布单独使用长边 1024、16 倍数补边，完整变换在 `assets/<id>/transform.json`。

透明化是本地后处理，不能称为 SenseNova 原生 alpha：程序选取远离源目标颜色的品红/绿/青色，让编辑器只把需要删除的部分涂为该颜色。再从输出边缘估计实际背景色，清除同色区域（包括封闭孔洞），窄边缘做 alpha 混合与去色。没有白色背景的自动兜底，避免误删白色目标。若编辑器忽略指定背景色，记录 `background_key_not_found`，保留原始结果并标记 `needs_review`。矩形照片/科学面板内部始终不透明，仅裁掉外部画布。

实际背景色允许同色系偏移（例如亮品红被画成偏红粉色），但会与原目标像素再次比较；若与目标颜色冲突则拒绝自动透明化，保留输出并标记问题。此保护不能替代视觉检查，细线重绘、底色误保留和边缘残色仍可能发生。

SenseNova 默认 30 steps、seed 42、text guidance 4、image guidance 3。`--image-guidance` 可调；服务字段已实测可用，调高不等于像素保真保证。生成模型仍可能改变线条、人物、毛发或科学噪声，结果默认 `extracted_unverified`，不会自动视为验收通过。

输出：

- `inventory_before_boundary_review.json`：独立复核前的完整候选；`inventory.json` 保存复核后的整图框与逐目标复核状态。
- `boundary_review/state.json`：独立复核预算、决定、未解决 ID 及调用证据。`boundary_review/<id>/round_NN/` 保留原生裁剪、干净输入、当前框叠加图、修框提案叠加图、修改前后实际裁剪、变换和原始复核结果；对应真实请求在 `calls/boundary_review/`。
- `detection/state.json`：初始候选、局部发现、被排除项、查漏响应、未解决问题。`model_review_complete` 只表示模型覆盖审核，不等于人工验证召回率。
- `detection/search_coverage.json`：区域父子关系、实际视野、尺度、搜索状态、未完成任务和预算。分别记录规划覆盖、实际查看覆盖、完成检查覆盖。
- `detection/local_*/NNN/`：`source_native.png` 为原始分辨率裁剪，`input_2048.png` 为实际局部输入；`frame.json` 和 `region.json` 保存原图裁剪范围、坐标变换与区域身份。
- `image_stage.json`：图片资产独立状态；`scene.json` 使用 `png2pptx.scene.1`，增加顶层 `image_assets`，避免侵入 shape 对象管理。
- `assets/<id>/`：原生裁剪、扩充上下文、`writer_input_2048.png`、`prompt_plan.json`（模型生成）、`prompt.txt`（实际发送）、编辑输入、未修改的 `edit_raw.png`、`rgba.png`、透明预览、变换、后处理记录。
- `calls/`：真实请求、响应、SSE、usage、失败记录和缓存身份；API key 不落盘。
- `recovered.pptx/svg/png`：原位置放置的独立图片对象；合并时先图片再几何。PPT 支持移动/缩放图片，图片内部不是原生形状。`write_pptx()` 原来的几何计数语义保留。

`--detect-only` 完成搜索检测及独立边界复核、不编辑；`--only I0001 I0003` 仅限制提取目标，搜索与边界复核仍检查完整候选，正式全量不要使用；`--resume` 要求代码、参数、原图及候选身份不变。VLM 请求出错时最多一次纠正重试，原始错误、两次请求均保留；SenseNova 不自动重试。缓存仅复用精确匹配的成功结果。查漏预算用尽仍有问题则 `unresolved`，末轮发现的候选不会丢弃。三个 VLM 阶段的输出预算分别由 `--max-tokens`、`--boundary-max-tokens`、`--prompt-max-tokens` 控制，输入图片另计；`status.json` 分别汇总检测、边界复核、写 prompt 的调用数和实际 usage。

批量报告逐图显示 `生成数/候选数`，失败或未执行的目标有红色占位框，避免把抽样或失败误看成检测结果。机器清单位于 `asset_manifest.json`、`page_manifest.json`、`VALIDATION.json`，逐目标对照图包含原图局部、SenseNova 原始输出和 alpha。使用 `python -m png2pptx.images.report --batch <batch_run> --output <review_dir>` 可离线重导出，不调用模型。

此阶段只准备融合接口，没有根据 bbox 重叠自动删除几何对象。若旧几何管线已经重建了同一个图标，合并导出会同时保留两份，应由后续统一路由处理。`accepted_only=True` 仅导出人工标记 `accepted_by_user` 的图片资产。
