# 图片与完整管线

`composition.py` 将当前同源图片资产插入 PPT，位于文字／公式下方。保留原有形状、箭头、文字的 XML 和相对顺序，不重新拟合或识别。图片为独立栅格 picture，可在 PPT 中移动、缩放或替换；其内部像素不是原生矢量。

## 两种入口

在实验根目录执行：

```bash
# 已有结果离线合成：不加载 API 凭据，不调用模型，不覆盖来源目录
python -B -u run_complete.py --run-name my_complete_composition \
  --from-run runs/text_pipeline_full_13_v2

# 新完整运行：定位、拟合、样式、箭头、图片插入、文字恢复
python -B -u run.py --complete --run-name my_complete_pipeline

# 复用当前定位，再运行后续完整阶段
python -B -u run.py --complete --run-name my_complete_from_localization \
  --from-localization runs/localization_image_no_reasoning_effort_50k_c64_v1
```

离线入口支持 `--ids`、`--workers`、`--render-workers`，以及显式的 `--image-source <运行目录>`。源文件、源 PPT、图片清单与来源链均做哈希核对；新运行保存代码快照。相同名字只用于同输入、同代码恢复，修改后使用新名字。

完整主入口和 `--from-fitting` 重检入口都在箭头之后、文字之前插图。也可使用独立开关 `--with-images`。文字模块优先读取含图片底图，将实际 picture 位置作为 IMG 标注送入 OCR，提示图片内部文字已被保留。

## 图片与图层契约

- 沿 manifest 的 localization/geometry/style/base/arrow/recognition/composition 来源查找清单；源图哈希、尺寸必须匹配，缺少清单报错，不能当作没有图片。
- 优先精确使用 `source_box_half_open`，裁切原图像素，不插值重建。图片放置使用相同的 half-open 坐标。
- 已导出的 RGBA 必须通过源图、裁剪范围、文件哈希和放置范围验证。否则使用原图裁剪。`background=alpha` 但没有遮罩时保留矩形原像素，写入 `alpha_unavailable_original_rectangle_preserved`，不冒称已透明分割。
- 每个图片对象名为 `IMG Ixxxx`，插在最早的非空文字／公式对象之前。图片以下原有图形的相对顺序保持不变；带文字的 group 整体保留在图片上方。合成前后及重新打开 PPT 后核对全部原对象 XML。
- 重复插入时检查同名图片的内容哈希与位置；完全相同则复用，冲突报错。边界未决、alpha 回退、上游未决状态均保留。

## 输出与视觉限制

离线结果位于 `runs/<name>/cases/<id>/complete/`：`combined.pptx`、LibreOffice 实际渲染的 `native.png/.pdf`、`comparison.png`、`scene.json`、`assets.json`、逐图片 crop/record，以及 `text_image_overlaps.json`。

新完整运行的图片中间产物位于 `cases/<id>/image_composition/`，最终结果位于 `cases/<id>/texts/combined.pptx` 和 `texts/combined/native.png`。

离线复用历史 OCR 时保留其全部文字；文字 bbox 超过 90% 位于图片框内仅作为重复候选记录，不据此删除公式。新运行的 IMG 排除提示尚需真实 OCR 验证。矩形图片可能遮盖底层图形；有争议的边界、图片内重复文字以及已有形状／箭头／文字误差仍需查看实际画面。导出完成和数量正确不表示视觉准确。
