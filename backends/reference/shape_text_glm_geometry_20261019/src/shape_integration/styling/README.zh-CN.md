# 固定几何的样式恢复

输入是与原图哈希匹配、已经结束的 fitting scene。读取每个对象最终选中 revision 的全部成功 part，包括几何状态为 unresolved/rejected 的对象。样式阶段不移动点、不改路径、不增加部件、不回写定位 bbox；所有样式及显示顺序另存。

## 使用

沿用实验工作区的 Python 环境；样式模块使用 Pillow、NumPy、SciPy、OpenCV、Shapely、python-pptx、CairoSVG 和 PyMuPDF，实际 PPT 渲染还需要系统 `libreoffice`。本轮验证使用现有 `internsvg` 环境，未修改系统依赖。

在实验工作区根目录执行：

```bash
# 冻结已有几何，只跑样式
python -B run_styling.py --run-name my_styles --from-fitting runs/fitting_3d_cues_7v_gpt_timeout1200_cap131072_v1
# 无 API 的规则与导出检查（仍用 LibreOffice 实际渲染 PPT）
python -B run_styling.py --run-name my_rules --from-fitting runs/fitting_3d_cues_7v_gpt_timeout1200_cap131072_v1 --rules-only
# 子集试点；JSON 为 case ID -> 非空 object ID 列表，只影响实验取样
python -B run_styling.py --run-name my_pilot --from-fitting runs/fitting_3d_cues_7v_gpt_timeout1200_cap131072_v1 --selection selection.json
# 新定位/拟合完成后接样式
python -B run.py --run-name my_full --with-styling
```

`--ids` 可选择图片；`--config` 可指定明确的实验配置。只有 `run_styling.py --from-fitting` 是纯样式入口，`run.py --from-fitting` 会重新检查/修正几何。每次实验使用独立目录；来源、代码或配置改变必须换运行名。缓存不会静默重发已中断的模型请求。

## 流程与接口

1. 从最终路径的真实边界（包含三次曲线极值）裁剪原图，保留少量 padding。clean 图提供颜色，另一张图只标本次目标轮廓/短 ID；不把全局 bbox、几何点表或邻居清单注入 prompt。
2. 遮挡关系只在不同 `object_id` 的完整组件之间建立，以组件全部拟合区域的并集判断包含/部分重叠。包含者在下，被包含者在上；部分重叠先按连通组分开，再分批由模型看完整组件的联合 crop 判断。每个组件只有一个 C ID 和统一颜色的外轮廓，不提供内部 part ID。内部层序保持原拟合顺序；拓扑冲突、跨组交错、证据不足保留未决。
3. 规则估计颜色、线宽和线型，生成颜色/填充候选；仅依据已确认且能应用的跨组件顺序，排除前景实际填充区域或描边带。空心框不遮住整个内部。同组件内部关系不生成排除区域；未判定、超时和循环冲突也不触发双向屏蔽。整组件 Top 15 颜色统计保留自身所有可见颜色。image bbox 仍只作为保守的不可信采样区域，不当作精确透明度 mask。
4. 组件级分组只考虑独立的单 part 对象。规则按几何类型、比例和尺度提出重复候选，模型看 CLEAN 确认哪些外观能共享完整样式；复合对象明确排除。不确定、非重复或分类失败的组件走普通管线。确认的重复组件共享一份样式，选择、修正及 critic 都查看组内全部实例。
5. 复合对象整体一次进入模型，select、critic、refine 均不按 part 拆批；某个 part 需要修正时仍提供整个对象的全部目标。独立非重复对象沿用规则筛选疑难对象的流程。规则初值、reasons 和线宽/段长候选均不进入请求。原图目标区域像素按 RGB 精确计数，重叠区域只计一次；颜色候选按像素数降序取前 15 个，每次请求只给一份公共色表，计数和范围另存 color_candidates.json。渐变候选绑定对应面。候选颜色/渐变仅供参考，select 首轮即可返回任何符合协议且有原图证据的颜色或双色渐变，后端不再校验候选成员关系。重点恢复填充与描边；虚线需估计交替实段/空段组成的重复周期，线宽和段长在主 CLEAN crop 中自由估计。
6. 导出原生 PPT，LibreOffice → 矢量 PDF。critic/refine 按主 CLEAN 的尺寸、位置、倍率与亚像素补边，从 PDF 直接渲染局部图（direct_pdf_crop.1）。全图 PNG 保持原图尺寸用于总览。critic 对比同范围原图与实际 PPT 渲染，并通过每个目标的 `current_style` 接收该渲染实际使用的填充、描边、线宽、线型和段长参数；检查对应边界上的虚线段长、间隙和重复密度。通过项给结论，问题项给图像证据与修改建议；参数本身不是视觉正确的证据。
7. 候选与 critic 问题共用最多一次自由修正。refine 的每个目标通过 `previous_style` 收到上一轮最终用于渲染的样式参数；线宽和虚线段长换算到当前主 CLEAN crop 像素，其他样式值保留，按图像证据调整并返回完整样式。随后渲染并复查受影响的完整对象或组件样式组。预算耗尽仍保留最后可用样式和状态。

### 组件并行与实时产物

调度协议 `component_pipeline.1`：完成遮挡、规则和组件分组后，各完整对象/共享样式组独立推进 `select → critic → refine → critic`。组件之间不再等待同一阶段全部结束；同一组件仍按依赖顺序执行，共享组作为一个完整单元提交。模型任务只有一层共享线程池，不在池内提交任务后阻塞等待，因此 worker 较少时也不会死锁。全局 API 名额、渲染和像素分析各自沿用原有上限。

每次提交模型调用时冻结上下文；critic 的 PDF、`current_style` 与随后 refine 的 `previous_style` 来自同一个快照，包含应用最低线宽后的最终值；未显式指定的虚线段长使用导出器同一默认周期补齐，再换算为当前主 CLEAN crop 像素。每个组件结果返回后，协调线程立即保存 `styles.json` 和 `progress.json`；组件尚未完成时使用 `running_*` 状态。如果其他组件后来的样式变化进入已检查 crop（也包括跨入 crop 的描边），旧接受结论标为 `pending_context_recheck`。全部样式稳定后，仅对这些单元做一次最终复查，不追加 refine。因而每组件最多一次自由修正；critic 首查一次、自身样式变化后一次、异步邻居又改变已检查区域时额外一次。没有变化的区域不重查。

渲染在独立线程运行，按 `styling.live_render_interval_seconds` 合并新样式，默认 **5 秒**；这是有变化时两次渲染启动的最小间隔，并非保证五秒内完成。最后一批不等待间隔。整图 `snapshots/snapshot_<sequence>_<hash>/` 保存不可变的样式、原生 PPT、PDF、PNG、比较图与哈希，失败快照保存 `error.json`。全部输出通过完整性校验后才原子切换 `current` 软链接；公开的 `styled.pptx`、`native.png`、`comparison.png` 等都经该指针引用最近完整渲染，渲染失败继续保留上一版。已打开的 PowerPoint 文件需要重新打开/重新加载才能显示新内容。

`styles.json` 可领先于渲染；`progress.json` 分别列出当前 `sequence`、`rendered_sequence` 和每单元阶段，`live.json` 指出当前 PPT 所属快照。运行总览 Markdown 每 5 秒刷新已出现的比较图。`scheduler.json` 保存组件断点，恢复时保留原请求上下文和 PDF，不把已完成调用换成新图重做；源文件/代码/配置改变仍需新目录。整个运行没有引入按 case 或对象 ID 特判。

`styling.pixel_calibration_enabled` 控制模型结果之后的像素线宽/虚线段长校准，默认 `false`，当前 `config.json` 也显式关闭。关闭时完全跳过该估计器，不测量、不覆盖样式；每轮 `calibration.json` 记录 `enabled=false`、`reason=disabled_by_config`，报告与 summary 显示开关状态。设为 `true` 可恢复原校准流程。该开关不影响初始规则候选生成，也不影响必要的 crop → 原图像素换算。

复合对象整体看图恢复；其内部可有多种样式，仍逐 part 返回并保留不同面色。组件级样式组由模型明确确认后共享一个输出，后台以原图单位应用到成员，贯穿恢复、修正和复查。几何分组不改变，复合对象不参与组件级分组。几何类型/尺度相似仅为候选条件；矛盾通过 issue/uncertain 暴露，不声称分组已被证明正确。

`styling.component_grouping_enabled` 默认 true，关闭后全部组件走普通流程。`color_candidate_limit` 默认 15，最大 15。旧 selection_batch_parts/review_batch_parts 配置读取时丢弃，不能重新开启复合对象拆批。

基础样式：纯色/无填充、描边色/无描边、原图像素线宽、solid/dash/dot/dash_dot 及交替实段/空段长度。

模型协议 `shape_integration.style.10` 的 `width_px` 与 `dash_pattern_px` 使用**本次主 CLEAN crop 像素**。后台按 `原图长度 = crop 长度 / source_to_crop_scale` 转换一次，再进入状态、可选像素校准和导出。规则估计原本就是原图像素，不再转换。原始响应、缓存、格式纠正始终保留模型单位；refine 使用本次 crop 自己的倍率。`validation.json` 同时保存 `model_valid`、换算后的 `valid` 和 `length_conversion`，最终 `styles.json` 声明 `length_unit=original_source_pixels`。

`styling.minimum_stroke_width_px` 默认 `0.8`，单位是**原图像素**；设为 `0` 可关闭。每轮导出前，已有描边的线宽取该值与恢复值的较大者，无描边部件仍保持无描边。规则、模型和 refine 的输出都执行此下限；原始响应保留，各轮 `stroke_width_floor.json` 记录实际调整，refine 的 `previous_style` 使用应用下限后实际渲染的参数。此规则与关闭的像素校准器独立。

select/refine/critic 的每张输入图底部附加相同的 MIN 参考条，其线宽为 `0.8 × source_to_crop_scale` 个局部像素（阈值可配置），支持小数像素覆盖。主图不缩放、不覆盖；`frame.json` 的 `crop_display_size` 是主图尺寸，`display_size` 包含参考条，`stroke_reference` 记录线宽和位置。模型只接收换算后的局部最低线宽。prompt 明确参考条是下限而非目标线宽，放大后的浅色抗锯齿边缘也应纳入估计，存疑时适度偏粗以保证全图可见；critic 同样遵守下限。遮挡关系与组件分组请求不添加参考条。

clean 与实际渲染使用相同等比变换，`frame.json` 保存实际倍率、亚像素补边偏移和像素中心映射。prompt 不注入原图倍率；移除独立 3× 边缘图，模型视觉输入只有同尺度的原图、目标指认图和实际渲染。critic 与 issue 的长度意见要求定性描述，避免下一轮裁剪改变后沿用错误的像素建议。

局部 PDF 渲染先将原 PDF 页映射到 crop 大小的矢量画布，再一次栅格化，避免生成巨大的全页高分辨率位图。PyMuPDF 操作在进程内加锁，模型调用仍共享原有并发调度。`frame.json` 的 `rendered_source` 保存 PDF 路径、哈希、页尺寸和显示映射；`job.json` 记录同一来源。初次 critic 和 refine 使用修正前 PDF，复查使用修正后 PDF。缺失或无效 PDF 暴露为输入错误，不回退到放大 PNG。几何与 direct_pdf_crop.1 渲染链保持不变。

简单渐变绑定单个已有闭合面：两个不透明 RGB 端点和 `angle_deg`（原图中 0 向右、90 向下）。跨度以面自身的填充参考框为准，不依赖裁剪 padding 或遮挡后的可见范围。SVG 写原生 linearGradient，PPT 写原生 a:gradFill，不拆色带、不贴位图。

默认模型为当前配置的 Boyue glm-5.3-flash，thinking 开启；API 全局 128、样式任务 worker 128、单请求 1200 秒、最大输出 131072。对象内与图片间请求共享同一个 Runtime；CPU 像素分析默认 4 路、PPT 渲染 2 路。格式纠正额外最多一次，仅处理无效/缺失行；超时、截断响应不提交样式、不重试该请求。

## 产物与状态

每图 `cases/<id>/styling/`：

- `scene_source.json`、`source.json`、`identity.json`：输入几何、原图和配置身份。
- `occlusion.json`：协议 `shape_integration.component_occlusion.1`，`scope=components`；关系端点是 object ID，另存 `component_parts` / `part_owners` 映射、实际采用的顺序、未决及冲突。内部 part 顺序不变，不改变原 scene；旧 part 级实验保持冻结。
- `component_groups/groups.json`：组件候选、模型确认的重复组及失败回退；`style_units.json` 记录完整对象/共享样式组调度单元。组任务及 trace 在 `component_groups/S*/revision_*/`，最终 styles.json 保存 style_groups 与逐成员引用。
- `rules.json`：像素候选、证据和疑难原因；`timings.json` 保存准备阶段及逐组件任务耗时，任务之间可以重叠，不能把它们的 wall_seconds 相加当作总耗时。
- `objects/<id>/revision_*/`：输入图、变换、短 ID 映射、请求/原始响应/reasoning、校验与逐行重试。
- `objects/<id>/revision_00/`、可选 `revision_01/`：组件每轮样式与校准；共享组的对应目录位于 `component_groups/S*/`。`context_recheck/` 保存必要的最终邻居影响复查。
- `snapshots/`：完整场景的不可变渲染快照及几何不变检查。`current` 指向最近完整快照；`live.json` 给出版本与样式哈希。
- `styles.json`：运行中持续保存逐 part 样式、来源、几何状态、样式状态和 critic 引用；每个 part 的 selected_style_revision 指向自身组件 revision，总体 selected_revision 指向整图渲染快照。
- `progress.json`、`scheduler.json`：逐组件实时阶段、渲染进度及可恢复断点。
- `styled.svg`、`styled.pptx`：可编辑结果；`styled.png` 是 SVG 预览，`native.pdf` 是实际 PPT 导出的矢量 PDF，`native.png` 是原图像素尺寸的整页预览。每轮 PDF 纳入导出哈希，最终 PDF 随选中 revision 一起保留。
- `comparison.png`、`REPORT.zh-CN.md`：原图/实际 PPT 对照和逐部件状态。Markdown 使用相对图片链接。

只导出已有 shape，缺失文字、图片和几何不能当作样式错误。`critic_accept` 是模型结论；`rules_only` 没经过模型复核；`unresolved` 及 `critic_uncertain/revise` 均保留可用结果。协议正确、几何未变、成功渲染都不等于视觉准确率。

## 实现位置与边界

`geometry.py` 负责几何裁剪/参考框；`occlusion.py` 负责关系与可见采样区域；`rules.py`/`calibration.py` 迁移旧像素方法（来源见 SOURCE.json）；`gradient.py` 提供通用渐变候选；`views.py` 准备原图证据；`grouping.py` 确认组件级重复组；`contracts.py`/`model.py` 实现短协议和逐行纠错；`controller.py` 管理单图依赖；`export.py` 原生导出与真实渲染；`runner.py` 冻结输入并共享调度。

不支持径向/多色标/透明渐变、纹理、渐变描边、任意分边样式或拆解现有分组表达交错遮挡。无效/多子路径几何的可见区域推断有明确诊断限制；已有几何错误可能阻碍取色。文字没有精确 mask，image bbox 也不能替代像素分割。PPT 已通过 LibreOffice 渲染路径，尚未人工验证 PowerPoint UI 编辑。

不得按图片名、对象 ID、语义类别或目标坐标硬编码样式。试点清单属于测试数据，不参与恢复算法判断。
