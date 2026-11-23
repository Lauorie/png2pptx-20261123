# 几何后补漏：全局 GLM 与 SAM 单候选核验

用户已确认同时保留全局 GLM 漏区发现；后续仍逐框核验。

完整顺序：定位/整理 → 初次几何/critic → 全图与分块 SAM 扫描、SAM 候选过滤 → 全局 GLM 漏区提名 → 单 bbox 核验 → 新增几何/critic → 原有与新增联合去重 → 样式/critic/导出。

- SAM 内 bbox IoU ≥ 0.9 合并；单个已有定位框覆盖 SAM 候选面积 ≥ 0.9 时过滤。坐标使用原图浮点精度，原始证据完整保留。
- 全局 GLM 看原图和实际灰底蓝线图。其提名不经过旧定位 bbox 覆盖过滤，允许发现旧框内部遗漏。
- 两路候选近乎相同的框合并来源，其余每框独立并发核验，绝不按普通重叠关系打包。
- 单框请求：CLEAN、CURRENT（真实几何）、CANDIDATE（一个橙框及对应 SAM 青色像素）。既有几何不显示 bbox；输入数字统一一位小数。
- 返回 covered/not_arrow/missing/needs_context/uncertain；只提出新增范围，不修改既有几何。上下文不足允许扩大一次，仍不明确则记录待处理。
- 新增走现有普通线/块状箭头两条几何分支，追加到联合去重之前。保留当前默认思考、1080–1536 自适应尺寸与样式约束。
- SAM 扫描失败仍允许初次几何及全局漏区检查继续，问题记录进入最终状态；GLM 全局失败也不阻止已有 SAM 候选核验。

入口位于 `runners/full_pipeline/prepare.py`，把参考运行的原图和参数与当前代码冻结到新目录；不复制旧 GLM/SAM/几何结果。执行命令分成离线准备与付费运行两步：

```bash
python -B code/checkpoint/runners/full_pipeline/prepare.py --working . --reference-run <旧完整运行目录> --run-dir <新目录> --cases task_0027 task_0008 task_0056 task_0077
python -B <新目录>/code/run_full125.py --run-dir <新目录>
```

第二条才启动服务请求。每次新实验先说明目录与验证目标。冻结 run 与旧 v046 均不覆盖。

本次同时保留既有 `LINE_PARTS_COMPAT_20261030.md` 所描述的分段协议改动；坐标输入精度修复见 `COORDINATE_INPUT_PRECISION_20261030.md`。灰底投影补齐 shape_parts，曲线、轮廓、宽度与虚线均只做一次坐标变换。

离线回归：40 项通过（新补漏、输入精度、块状箭头分流、旧 bbox 管线）；不代表新模型补漏召回率已验证。
