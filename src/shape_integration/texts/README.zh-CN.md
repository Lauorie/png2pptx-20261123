# 轻量文字恢复

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
- Times New Roman / Arial / Calibri 三选一，默认 Times New Roman，PPT 保留这三个标准字体名。`fc-match` 解析本机预览的替代字体，保存字体文件、实际 family 和 fallback 标记；原生度量表通过同一 LibreOffice 字体替代环境渲染。跨机器打开时会使用当地可用字体，可能存在细小度量差异。
- 用字体文件测量候选大小，然后每图生成一张离线文字度量表，通过 LibreOffice 实际渲染统一校准字形宽高、基线和旋转边界。按 bbox 约束等比求字号，默认容纳比例 0.98；不横向强行拉伸文字。若原生度量失败，明确记录 Pillow 近似 fallback，并标记未解决状态。
- 每行一个可编辑文本框，按实际字形范围与字体基线偏移收紧，只保留微小取整余量；斜体等负向伸出的笔画通过必要的正内边距容纳。关闭自动缩放和换行。收紧框时补偿旋转中心，保持文字显示位置与字号。每行一个主颜色，可加粗/斜体；上下标使用原生 run 的 baseline，由 Office 自动缩小一次，避免同时缩小字号导致重复缩小。旋转单位为顺时针度。
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
