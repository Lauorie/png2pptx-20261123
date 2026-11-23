# 基线 task_0236 / task_0201 重复箭头诊断

日期：2026-10-24。只读核查原始图片、冻结代码、实际请求、各阶段 scene 与最终 PPT 渲染；没有模型调用，没有改动基线代码或产物。

用户提供的 `experiments/pptbench125_critic_min25_20261021_v1` 与工作区的 `references/pptbench125_run` 指向同一个原始 run。本报告的 A 编号以该 run 最终 `scene.json` 为准。

**结论：这两个案例的主要重复在初次恢复就已经存在。0236 主要是同一排箭头被多个组重复认领，并且每个整排请求还估错数量；0201 是相同虚线被不同组、不同目标重复恢复。局部补漏没有制造这里指出的重复。**

## 1. task_0236：用户截图中的 AIMC CORE—ADC 横排

原图这一横排有 **18 支向下的小箭头**。最终相同区域却有 **68 条路径**，下面五组的路径全部已存在于 `stages/initial/scene.json`，早于 continuation 和全局补漏。

| 最终标记 | 实际来源 | 路径数 | 恢复范围 |
|---|---|---:|---|
| A1 | G001:T1 | 7 | 横排左侧部分 |
| A38 | G034:T1 | 31 | 几乎整排 |
| A85 | G082:T1 | 27 | 几乎整排，与 G034 重叠 |
| A51 | G049:T50 | 2 | 横排中部两条 |
| A25 | G022:T1 | 1 | 横排最右侧 |
| 合计 | 5 个独立恢复任务 | **68** | 对照原图 **18** 支 |

18 的计数经局部放大核对，并用原图 y=665、667、669、671、673 的箭头头部横截面交叉核验，各截面均为 18 个独立黑色区间。像素计数仅是离线诊断，不写入重建算法。68 是该区域导出的路径数，不声称每条都能精确对应某一根原图箭头，因为两组自身的间距和数量也错了。

![原图、初次恢复、最终实例与原生 PPT](0236_cluster.jpg)

![逐组拆开查看](0236_bottom_by_group.jpg)

### 错误从哪里开始

1. **候选本来只指向局部几支，身份输出却把任务扩成整排。** G034 的 G34 框只覆盖中部一小段；G082 的 G89 框只覆盖另一小段。实际输入都有 CLEAN、候选框、整图，以及橙色主证据/浅青色邻组证据。不是没有给 bbox，也不是没有给 mask。
2. G034 身份输出 `network`，定位描述是“从 AIMC CORE 底部到 ADC 顶部的密集向下箭头排”；G082 也输出同一整排的 `network` 身份。两组得到相互重叠的任务，但没有统一它们是否代表同一个对象族。
3. 后续几何分别收到一个独立 R1。G034 的首个几何回复直接输出 31 个 path；G082 的首个回复输出 27 个。原图 18 支被各自估错，再叠加到一起。不是编译器把一条几何复制了多次。
4. 两个 geometry critic 都返回 `accept`。它们只收到当前目标的 CLEAN 与 CURRENT，没有另一组的已恢复路径和归属表，既没发现本组数量不对，也不能验证组间重复。
5. 汇总只检查路径 ID 是否唯一。`G034:T1/p1` 与 `G082:T1/p1` 是不同 ID，即使描画相同原图区域，也会同时发布。后续范围审核明确禁止去重，样式阶段固定几何，因此重复一直留到 PPT。

![两组实际身份输入：小框/局部橙色证据被解释成整排](0236_identity_inputs.jpg)

实际 trace：

- G034：[身份回复](../../references/pptbench125_run/cases/task_0236/groups/G034/identity/result.json)、[几何 prompt](../../references/pptbench125_run/cases/task_0236/groups/G034/geometry/F001/request/prompt.json)、[31 路径回复](../../references/pptbench125_run/cases/task_0236/groups/G034/geometry/F001/request/result.json)、[critic accept](../../references/pptbench125_run/cases/task_0236/groups/G034/geometry_critic/eff815fe215b/revision_00/answer.json)。
- G082：[身份回复](../../references/pptbench125_run/cases/task_0236/groups/G082/identity/result.json)、[27 路径回复](../../references/pptbench125_run/cases/task_0236/groups/G082/geometry/F001/request/result.json)、[critic accept](../../references/pptbench125_run/cases/task_0236/groups/G082/geometry_critic/2a14398bd026/revision_00/answer.json)。
- 两个几何请求和两个 critic 的实际 `call_000` 均为 `transport_complete / sse_done=true / finish_reason=stop`。不是超时残片或重试结果被重复追加。

### A96–A102 是否也是重复

这七条来自 `continuation2`，位于左侧 DAC → AIMC CORE 竖排中 A2 下方；A2 自身恢复上方十条，continuation2 接着补后面七条，其中一条替换已有 `G038:T1`。**它们不是下方横排 68 条重复的来源。** 这批线的端点和头部样式仍有误差，不能因为来源合理就说最终视觉完全正确。

另一个 continuation 也尝试修改 G038，代码记录了同 ID 写冲突，并用更新后的已有路径重新请求；没有把两份冲突结果直接同时加进去。[冲突记录](../../references/pptbench125_run/cases/task_0236/continuation/parallel_summary.json)。

阶段数量：初次恢复 174 → continuation 181 → 范围审核 179。全局补漏超时，未产生 missing 局部补漏产物。[覆盖状态](../../references/pptbench125_run/cases/task_0236/coverage_status.json)。关闭补漏不能消除初次恢复中已经存在的横排重复。

## 2. task_0201：create sandbox 被恢复了三次

原图两条橙色虚线分别是 initialize 与 create sandbox。G003 已恢复了两条；G013 又认领这两条，并把两份几何都画在右侧 create sandbox 附近。

| 最终标记 | 身份 | 实际几何全图坐标，约值 | 结果 |
|---|---|---|---|
| A5 / G003:T3 | initialize | (328.8,214.6) → (594.0,165.3) | 上方长虚线 |
| A6 / G003:T4 | create sandbox | (469.1,213.3) → (597.1,182.1) | 下方较短虚线 |
| A16 / G013:T1 | 名义为 initialize | (470.0,216.2) → (601.0,180.1) | 实际贴近 A6；身份与几何不一致 |
| A17 / G013:T2 | create sandbox | (470.2,215.9) → (600.9,182.7) | 再次重复 A6 |

![0201 原图和重复虚线](0201_cluster.jpg)

G013 用同一个候选 G16 给 T1/T2 定位，两个几何请求分别独立执行，未获得另一目标的归属信息。G013 的 CLEAN 全图范围为 `[392,148,679,248]`，真正 initialize 起点 x≈329 已在 crop 外。虽然身份请求提了向左扩上下文的 followup，初次几何仍把 initialize 错接到 x≈470，两个 critic 都 accept。

**这里既有跨组重复，也有错误起点造成的同组重复。** 只比较最终曲线距离而不看身份、原图端点和所属目标，会把两类问题混在一起。

更直接的证据是后续局部补漏回复自己说明：initialize 已有 E7，create sandbox 已有 **E8/E28/E29**；[existing_map](../../references/pptbench125_run/cases/task_0201/missing/missing1_7061b843e2/local/existing_map.json) 将后三者映射到 `G003:T4 / G013:T1 / G013:T2`，恰好就是 A6/A16/A17。它返回 complete、无新增无替换，因为任务要求补缺失，没有承担已有路径的删重。[实际回复](../../references/pptbench125_run/cases/task_0201/missing/missing1_7061b843e2/local/request/result.json)。

初次恢复、continuation、范围审核以及最终结果均为 41 条路径，最终 ID 集合与初次恢复一致。[补漏提交记录](../../references/pptbench125_run/cases/task_0201/coverage_changes.json) 为 `added=[] / replaced=[]`。两个局部补漏区域均未改动结果。

注意：41 是 drawable path 数，不是 41 支完整箭头。该旧版也会把部分头翼、虚线短段拆成 path；这些要单独判断，不能一概按完整箭头重复统计。

## 3. 共同的流程缺口与后续方向

当前跨组信息流主要传的是 **像素证据**。实际 prompt 明确说浅青 mask 是未验证的位置提示，不代表已完成覆盖或对象归属。模型能看到相邻墨迹，不等于知道“这整排/这条虚线已由哪个任务负责”。

- 身份阶段允许将局部候选补全为完整对象，这对残缺 mask 有用；但多个组都补成同一个对象族时，需要统一目标身份。
- 单目标几何/critic 有利于减少单请求负担，但如果缺少简短的相邻目标归属信息，各请求仍可能重复恢复。
- [汇总代码](../../references/pptbench125_run/code/scene_snapshot.py) 只验证 ID 唯一；[范围审核](../../references/pptbench125_run/code/scope_review.py) 明确写了 `do not ... deduplicate overlapping arrows`，它不是漏执行的去重阶段。

因此后续应优先验证：身份结果在几何前对明显重叠目标做局部归属核对；同一密集对象族只由一个任务恢复，其余候选引用该目标；几何合并后针对少量重叠冲突核对原图，区分合法近邻、共享段和重复。密集排的 critic 还应检查可见成员数量和对应关系，不能只修端点/平滑性。以上是待验证方向，本次没有改 prompt、加规则或启动实验。

不能把框变成强制几何裁剪边界，也不能把“18”之类个例数量写入算法。要保留补全残缺对象的能力，同时明确任务之间的归属。

[机器可核查证据与哈希](evidence.json) · [离线采集脚本](collect_evidence.py) · [整图概览](overview.jpg)。
