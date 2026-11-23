# SAM新增候选独立范围审核

新阶段接在review_additions.run_case的新增框整理之后、最终几何交接之前。仅判断新增候选是否为范围内线条；只有模型reject可删除；uncertain/失败保留且记issue。原始框与保留候选坐标保持，完整原图＋局部干净图＋目标角标，旧语义描述不输入。

sam_addition_scope_review_enabled默认true，sam_scope_workers默认4。每个候选一次请求，默认thinking不传reasoning_effort，共享既有Model的API信号量。原有几何scope_review开关不变。

已完成的SAM整理结果可以补审核：

    python -B code/prepare_addition_scope.py --source-run 已完成的SAM整理目录 --run-dir 新运行目录 --cases task_0237
    python -B 新运行目录/code/replay_addition_scope.py --run-dir 新运行目录

省略--cases即沿用源运行全部case；可用--selection-json指定抽查ID，此时其余新增框会明确标为本轮未审，不应称作全图验收。凭据沿用ARROW_API_KEY_FILE。每个新运行冻结code/config/inputs与SHA-256。

几何入口的source_localization和bbox_discovery_run指向新运行目录，等各case状态完成再接入。旧缓存定位不会因为修改代码而自动获得本次审核；不要将旧定位产物标记为已通过新阶段。

验证：v044_sam_addition_scope7_20261030，4图7项，0237矩形边排除，其余6项保留。33项离线测试和准备到交接冒烟通过；保留组内混合真实线条/噪声为uncertain，未新增复杂拆分任务。
