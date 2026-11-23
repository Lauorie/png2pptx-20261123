# 固定几何的样式恢复

输入为与原图哈希匹配、已经结束的 fitting scene。只修改填充、描边和显示顺序；路径、点、分组成员和定位 bbox 保持不变。

## 使用

沿用原有 Python/LibreOffice 环境。完整数据、数值拟合、模型调用与渲染应在 rjob 中执行。默认同时处理两张图，像素与渲染各至多两个任务，样式模型 worker 为 16；该限额不改变定位/拟合阶段配置。

```bash
# 最新 SAM 缓存已由相同原图与固定几何生成时
python -B run_styling.py --run-name my_styles --from-fitting runs/my_fitting \
  --sam-evidence /path/to/latest_sam_cache
# 不提供 SAM 缓存：相同数值流程使用 CLEAN 像素
python -B run_styling.py --run-name clean_styles --from-fitting runs/my_fitting
# 小范围试点；rules-only 不调用模型，但仍有真实 PPT 渲染
python -B run_styling.py --run-name pilot --from-fitting runs/my_fitting --ids ds005 --rules-only
```

`--selection` 为 case ID → object ID 列表；`--config` 指定配置。相同运行名冻结来源、代码、配置及缓存身份，变更后必须使用新运行名。

## GLM 上下文与生成预算

样式阶段的 `max_tokens` 默认 **32768**，包含思考与最终答案。项目配置和省略该字段时的默认值一致；select、critic、refine 及关系判断均使用此预算。

`experiment/style_sam_native/submit.sh` 为新任务生成的 GLM 服务配置默认使用 **131072** 总上下文，可通过 `GLM_MAX_MODEL_LEN` 覆盖，例如 `GLM_MAX_MODEL_LEN=131072 bash experiment/style_sam_native/submit.sh /path/to/new_run`。输入中的文本和图片 token 与生成预算共同占用上下文；增大服务上下文不会自动增大 `max_tokens`。

配置调整在新启动的任务中生效，已有任务的冻结配置不变。启动参数检查不等于 GPU 服务已验证支持该长度；需要查看实际服务的 `/v1/models`、启动日志与请求结果。通过中继或外部端点调用时，总上下文取决于对应后端配置。

## 精简流程

1. 读取全部成功 part 的固定路径。跨组件包含/遮挡先由几何与原图判定；证据不足保持未知，不按面积猜前后关系。
2. 规则提供填充/描边的初始语义证据，GLM select 提议填充、描边存在性、颜色和渐变；明显不受原图支持的描边色，由稳定的局部线芯校正。重复组先合并原图颜色证据，统一取色，再逐实例测量宽度。复合对象整体看图；确认重复的独立组件只共享这些语义属性。
3. 每个 part 独立进行原生 crop 数值拟合。沿固定路径采样法向剖面，在上采样 CLEAN 与 SAM mask 中联合拟合线宽，再换算回原图像素。SAM 填充体、其它目标碎片、阴影和不可见剖面不当成真实描边厚度；没有可用 mask 时使用 CLEAN。
4. CLEAN 的连续实段/空段决定虚线种类和重复周期。交叉线、遮挡与缺失 mask 视为未知，不制造虚线间隙。目前不拟合虚线相位。
5. 实心小点用内部颜色覆盖与独立边环判断，取消重复同色描边，保留真正空心点。同一组件内的填充背景放在可见内部部件后面；几何保持不变。
6. 全部样式完成后渲染一次原生 PPT。critic 检查与原图同范围、同倍率的 PDF crop，并指出受影响的语义或数值字段。最多一次 refine；颜色改变重新拟合，纯线宽/线形问题直接进入数值拟合，省去不能修改数值的颜色请求。线宽修订只给 increase/decrease，选择原图已有的宽度模式；不按固定倍数缩放，也不执行与要求相反的变化。
7. 若样式变化，完整渲染第二次，并复查所有 crop 内可见该变化的组件，避免邻近组件继续使用旧画面上的通过结论。不再追加修正。

合并流程启用 `conditional_pattern_critic`：只对数值未决线形或 critic 与测量有冲突的线宽，使用首张实际 PPT 的局部 PDF crop 做一次小观测，输出线形类别及粗细方向，生成预算至多 8192。观察不确定时保留未决；它不修改几何、颜色或文字，不增加整图渲染次数。原实验默认不启用此选项。合并流程的常规样式请求预算为 100000，保持其他分支各自的配置。

SAM 的缺失不能产生虚线间隙。目标细线 mask 与 RGB 线芯匹配后才提供正向支持；填充体 mask 不作描边。SAM 提议的周期还须同时得到原图实段和空段支持。可见性按目标轮廓判断，不再因宽采样带碰到邻居就排除整条剖面。未经确认的粗规则线形不会进入最终样式。

GLM 的 select/refine 不输出线宽、段长和间隙；critic 提供定性的原图/渲染差异。重复组件不共享代表对象的数值宽度。删除“优先略微加粗”的目标；`minimum_stroke_width_px` 默认 **0**，避免统一下限再次放大细线。

颜色恢复与宽度测量分开：填充边界用内外两侧背景的颜色混合解释抗锯齿；GLM 把浅色边缘选成描边色时，可用原图持续出现的深色线芯测量宽度，不按颜色对比度放大导出线宽。原始语义颜色、测量颜色和诊断都会保留。

## SAM 缓存与本轮验证

缓存格式为 `ROOT/<case>/line_evidence.json`，包含原生 `tile.png`、每个 part 独占的目标 mask、tile 原图范围和倍率。缓存必须来自相同原图与几何。显式指定但缺失的 case 缓存会报错；只有不提供缓存时才整体使用 CLEAN。

本轮沿用原 SAM3 环境、代码、阈值，只使用最新 `sam3_training_11classes_8gpu_bs32_ep10_min10_20261015/checkpoints/best.pt`（epoch 10）。重新完成同输入的 8795 个最新权重原生 crop 推理；复用裁剪图，没有训练。96 px 的原图 tile 上采样到 1008 px，拟合保留这个分辨率；不先缩回整图再测宽。借鉴箭头实验的局部 mask 参数拟合，但保持现有 shape 几何，不重新联合拟合箭头头部。

SVG 和原生 PPT 的圆端帽规则分别校验；近零长度的原生 PPT dot 会在 LibreOffice 中变粗，因此使用经实际渲染验证的 dot 参数。模型接受、协议通过或几何不变均不是人工视觉真值。

本轮报告与 27 图对照：`runs/style_sam_native_20261016_v1/delivery/REPORT.zh-CN.md`。原始纯 GLM 与上一版 SAM 代码、trace、失败试点和导出均保留在 runs 中。

```bash
# 在集群的同一运行环境中执行
python -B experiment/style_sam_native/tests/test_native.py --src src
python -B experiment/style_sam_native/tests/test_mainline.py
```

## 输出与边界

`styles.json` 保留逐 part 语义、数值测量、历史与 critic 状态；`native_fits/` 保存拟合身份、SAM/RGB 权重、线芯色、偏移与宽度分布；`context_recheck.json` 说明哪些组件因邻近变化复查。`snapshots/` 为完整不可变渲染，`current` 仅指向通过哈希校验的快照；`styled.pptx`、`native.pdf`、`native.png` 与 `comparison.png` 指向最终结果。

旧 `pixel_calibration_enabled` 参数为兼容旧配置保留，新流程统一使用 native fitter，不再叠加旧校准器。旧的实时组件异步渲染已由至多两次完整渲染替代。

低分辨率、文字遮挡、浅色描边、几何偏移和不支持的渐变仍可能不准。缺失文本、图片或箭头需要前序几何阶段处理。保留未决结果；不按 case ID 或对象名称硬编码恢复规则。

## 已有 GLM 样式的精修

`--semantic-seed ROOT` 复用同一固定几何的已完成样式，读取 `ROOT/cases/<id>/styling`（也接受直接指向 cases）。该模式保留原填充、描边色和渐变，只在测量可靠时更新线宽/重复周期，并按源图证据去掉实心点的冗余描边。不可靠的宽度或线型保留原值；仅进行一次完整渲染，不调用样式模型。输入语义结果与几何必须完全匹配且已经结束。

```bash
python -B run_styling.py --run-name repaired_styles --from-fitting runs/my_fitting \
  --semantic-seed runs/my_previous_styles --sam-evidence /path/to/epoch10_cache
```

完整语义重判与保留语义精修有独立对照；采用哪种结果应结合实际对照图与报告，不能把宽度专项指标当成整体视觉质量。
