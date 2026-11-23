# 轻量文字恢复

当前 GLM 主入口 `run_global_ocr_review.py` 使用单轮低分 CROP：全局识别与必要公式 LaTeX 恢复 → 原生基线渲染 → CPU 分数低于 80 的文本按邻近关系分组（每组最多 6 个，4 组并发）→ 一次联合修正内容、颜色、字体样式、旋转及 bbox → 原生重渲染及不降分回退。高分对象只作为只读上下文，不再先对全体文本做 CROP，也不再追加几何专用 CROP。全局提出的普通文本局部补查、多行重请求延后到该轮；公式结构恢复仍独立保留。格式修复重试可发生，但碰撞处理只做 CPU 拒绝，不追加模型轮次。

每个低分 CROP 提供相同坐标/尺度的干净原图、bbox/ID 高亮图、当前 PPT 渲染图。高亮图红框为低分目标、灰框为只读邻居，不覆盖干净原图；提示框是待校正预测，不是正确边界。模型还获得逐目标 CPU 分数。内容、样式和坐标可以联合修正，CPU 不再贴边收框。默认不接受 CPU 降分，并保护高分邻居不被改动遮挡。缓存及请求记录位于 `low_score_crop/`，接受/回退记录在 `cpu_guard/`。已识别的低分对象才能进入该轮；全局漏检且没有对象的文字不会被该分数筛选自动发现。

一行一个 bbox；一次全图 GLM 请求同时识别内容、框、旋转和基础样式。字号完全由后台计算。数学标签通过原生文本 run 表达上下标。普通图不增加逐行 critic/refine；只有模型提出密集区域时，才并行放大这些区域补查一轮。

## 入口

在实验根目录 `experiments/shape_pipeline_integration_20260922` 执行：

```bash
/mnt/shared-storage-user/yinjinhui/H-conda/anaconda3/envs/internsvg/bin/python -B -u run_texts.py \
  --run-name text_pipeline_full_13_v1 \
  --from-run runs/arrow_critic_evidence_first_workbench_v1/final_13
```

`--from-run` 读取已完成的 shape/arrow 实验，复制冻结 PPT 后追加文字。省略它可生成 text-only PPT。`--ids 015 090` 可限定图片。再次执行同一命令时，成功模型响应按请求身份复用；配置、代码和输入不匹配时要求新目录。超时响应不会当作完成结果复用。

完整流程使用 `run.py --complete --run-name <新名字>`，启用拟合、样式、箭头、图片插入和文字恢复。文字阶段读取 `image_composition/combined.pptx`，图片区域会进入 IMG 标注；`--from-fitting` 的重新检查流程同样支持。也可按需使用 `--with-texts`、`--with-images` 等独立开关。默认 `texts_enabled=false`。

## 规则

- 全图长边 2048；局部倍率以全局输入坐标为基准，3 倍、长边上限 2048。`frame.json` 保存像素中心约定、裁剪范围、精确倍率与偏移。
- Times New Roman / Arial / Calibri / Courier New 四选一，默认 Times New Roman；等宽代码或打字机风格使用 Courier New。PPT 保留标准字体名，`fc-match` 记录本机预览的替代字体。原生度量表通过同一 LibreOffice 环境渲染，跨机器可能存在细小度量差异。
- 同一位置的长文本标签可合并下划线/下标转写差异，必须高度重叠并保留完整字符与大小写；不同位置、数值或公式不做模糊合并。合并保留来源与被替换 ID。
- GLM 的单行、公式及多行逐行 bbox 直接采用模型坐标，取消 CPU 原图像素找边界和字形模板缩放后的二次收框；没有有效模型框时保留上一个框。字体度量只按该框计算字号和原生文本框位置，最终 CPU 渲染评分仍可拒绝候选。`glyph_registration_enabled` 默认关闭；`--ocr-template-validation` 仅适用于单独的 OCR 后端。独立 OCR 模式保留其模板校验，但不再调用原图像素收框。
- 文本框填充不再由 LLM 输出或修改，旧回复中的填充字段也忽略。CPU 固定文本、颜色、字体及几何，对白底/透明两种实际原生渲染计算同一区域的 RGB `1-MSE`，选较高分，同分透明。仅内部保留 `background_fill` 导出原生文本框；不生成遮挡图片。有非文本底层 PPT 时自动比较，互不重叠的文本批量渲染，避免逐字调用渲染器。纯文本白底预览中两者无法区分，记录原因并保留透明，不根据原图臆造底层。`run_global_ocr_review.py --base-pptx /path/to/nontext.pptx` 可在最终组合时启用。审计在 `background_selection/selection.json`；失败时透明回退，无额外 API 请求。字体度量仍禁用填充。
- 用字体文件测量候选大小，然后每图生成一张离线文字度量表，通过 LibreOffice 实际渲染统一校准字形宽高、基线和旋转边界。按 bbox 约束等比求字号，默认容纳比例 0.98；不横向强行拉伸文字。若原生度量失败，明确记录 Pillow 近似 fallback，并标记未解决状态。
- 每行一个可编辑文本框。原生 PPT 文本框的内部排版尺寸由独立字体度量和基线偏移计算，与原图像素收框不同，不修改输入 `bbox_2d`；斜体等负向伸出的笔画通过必要的正内边距容纳。关闭自动缩放和换行。排版时补偿旋转中心，保持文字显示位置与字号。每行一个主颜色，可加粗/斜体；上下标使用原生 run 的 baseline，由 Office 自动缩小一次，避免同时缩小字号导致重复缩小。旋转单位为顺时针度。
- 保存已识别 ID；局部通过 `replaces` 修正既有行。仅在内容相同且 IoU 至少 0.65 时自动去重，不合并不同位置的重复标签。保护可读完整行，防止被局部不可读结果替换；保存合并历史。
- 已保留为图片的 IMG 资产以独立标注图提供给模型，用于避免重复覆盖图片内部文字；自然图像内容保持栅格。模型看到干净原图，标注不作为原始文字。
- 不可读内容和复杂二维公式的排版限制明确留存；当前不实现完整公式编辑器、字距恢复、逐字符多色或逐行视觉优化。

## 输出

`cases/<id>/texts/` 下：

| 文件 | 用途 |
| --- | --- |
| `scene.json` | 原图坐标、内容、runs、规则字号、字体替代和不确定项 |
| `recognition.json` | 合并后的识别结果，未读出的行也保留 |
| `overlay.png` | 原图上的逐行 bbox 与 T ID |
| `text.pptx`, `native.png`, `native.pdf` | 可编辑文字及真实 PPT 渲染 |
| `combined.pptx`, `combined/native.png` | 冻结形状/图片/箭头加原生文字 |
| `combined_comparison.png` | 原图与最终组合渲染对照 |
| `detection/global/`, `detection/regions/` | 实际请求、响应、reasoning、框变换 |
| `typography/metrics/` | 批量原生文字度量表及测量值 |
| `summary.json`, `export.json` | 执行状态、导出数量和哈希 |

运行根目录保存代码快照、配置、输入/底图哈希、用量、逐图报告和 `final_audit.json`。审计会重新打开组合 PPT，验证冻结底图对象 XML 保持不变及文字框数量。执行完成、bbox 数量和结构检查通过都不代表人工识别准确率。

组合结果继承底图已包含的对象。完整流程先插入当前同源图片，再识别文字；`retained_images.json` 只记录 PPT 中实际存在的图片。历史 `text_pipeline_full_13_v2` 的底图缺少图片，可使用 `run_complete.py --from-run runs/text_pipeline_full_13_v2 --run-name <新名字>` 离线补入，原有文字保持不变，可能与图片内文字重复，候选会保存在组合结果中。

## 代码来源

逐行 OCR 规则参考 `ref/native_qwen/native_qwen/element_protocol.py` 的 text task；规则字号思路参考 `method/locate_hier_agent_v1/png_agent/typography.py`。本模块独立实现坐标、基础样式、原生上下标、度量和导出，复用当前共享 Boyue transport 与 PPT 渲染器。历史 GLM 文字实验没有接入 PPT，不把其结果称为本轮新识别。
