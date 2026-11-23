# 细线箭头恢复

主线可用 `run.py --with-arrows` 在 shape 几何/可选样式之后执行。复用已完成 shape 样式时，使用独立入口：

```bash
python -B run_arrows.py --run-name arrow_full_example \
  --from-styling runs/style_fill_stroke_context_full_13_v1
```

沿用当前配置的模型、全局并发和请求超时。可用 `--ids` 选图，`--config` 使用新实验配置。改变代码/配置必须新建 run，历史文件不覆盖。

只修改样式请求时，可复用已结束箭头实验的定位和原图像素几何：

```bash
python -B run_arrows.py --run-name arrow_style_replay_example \
  --from-arrows runs/arrow_pipeline_full_13_v2
```

该入口整体重跑箭身、头型、实际渲染与样式复核；不调用定位或几何拟合。保留无几何/排除对象的状态，冻结来源哈希和原有 shape PPT，输出仍位于新 run 的 `cases/<id>/arrows/`。

附加 `--retry-missing-geometry` 时，仅对来源中没有可用几何且未被排除的对象重新拟合，从整图上下文开始；已有几何保持不变。这适合裁剪不足导致的失败恢复。旧 record 单独保存于对象的 `source_record.json`，来源 scene 哈希保存在 manifest 中。普通拟合在 `needs_context` 时也最多切换一次整图，避免围绕偏小原框增加少量 padding 后仍截断路线。

整图只负责找回完整路线。取得暂定点后，后台按其范围重新裁剪、放大，再进行局部拟合和几何 critic/refine，避免在整图上直接接受偏粗的端点。上一轮点表会转换到新 crop，旧视图的不确定性文字坐标不继续注入；原始输出保留。每轮 `input/target.json` 保存实际指认范围及原始候选框。局部几何 critic 最多修正次数仍由 `geometry_refines` 控制。

`controller.py`：全图箭头/粗区域发现 → 分区补漏 → 跨区全图补查 → 对象整体几何/critic/refine → 箭身样式 → 原生头部枚举 → 实际 PPT 样式 critic/refine。对象之间并发，单对象内部串行。点表 ID 显式表达共享节点；一条路径支持多个 line/curve 段，曲线使用线上采样点插值为开放三次 Bézier。

`geometry.py` 保存局部像素中心变换；模型数值长度只使用当前 crop 像素，最终 scene 长度为原图像素。`export.py` 使用原生 connector 或开放 freeform，头部是原生线属性。critics/refine 的几何/样式文本与输入图对应同一 revision。

样式阶段仅传路径 ID、头部有无、图像尺寸和相应样式参数，不传整张点表或几何不确定性文字。头型选择/样式 critic 附真实 PPT 渲染的枚举形状图（`assets/head_types.png`）；它不提供目标线宽、颜色或尺寸候选。原生示意 PPT 和哈希一并保存。

头型选择前的实际渲染首轮只有箭身，避免预置头型干扰；refine 使用上一轮头部配合新箭身，`shaft_style` 对应图中的完整参数。上一轮渲染可提供枚举尺寸参照，仍以 CLEAN 判断类型。原生头部大小不是线宽的通用固定倍数。

每图产物在 `cases/<id>/arrows/`：

- `detection/`：全图、区域和补查请求，框的来源及可视化。
- `objects/A*/geometry/revision_*/`：原图 crop、框、点/线图、几何请求及复核。
- `objects/A*/style/revision_*/`：箭身、头部、真实渲染、critic 请求及样式参数。
- `scene.json`、`progress.json`：保留未决状态及可用结果。`arrows.pptx`、`combined.pptx` 随对象结果定期发布；最终 `native.png` / `comparison.png` 使用实际 PPT 渲染。
- `duplicates.json`：仅高度重合的完整箭头网络去重记录。暂不与 shape 去重。

输出通过、候选数、结构校验都不是视觉准确率；需对比源图、实际渲染及 trace。设计和约束见 [ARROW_INTEGRATION.zh-CN.md](../../../plans/ARROW_INTEGRATION.zh-CN.md)。

当前代码分别记录 `geometry_attempt`（最后尝试）与 `geometry_revision` / `geometry_source_view`（最后有效几何），后续失败不会把有效结果标成失败尝试的产物。已启动的历史快照不回写；复核报告通过实际几何文件比对补充来源。

实验入口与结果：

- [v2 全部 13 图端到端实验](../../../runs/arrow_pipeline_full_13_v2/REPORT.zh-CN.md)。
- [v3 冻结已有几何、全量恢复样式与补试缺失几何](../../../runs/arrow_pipeline_recovery_full_13_v3/REPORT.zh-CN.md)。
- [v4 定向补拟合、组合输出 13 图](../../../runs/arrow_context_local_refine_v4/REPORT.zh-CN.md)：只对 v2 无几何的 12 个候选重新调用拟合，其余复用 v3，不能当作全部阶段重新运行。
- [实验分析](../../../reports/arrow_integration_20260927/REPORT.zh-CN.md) 与 [视觉及 trace 复核](../../../reports/arrow_integration_20260927/REVIEW.zh-CN.md)。
