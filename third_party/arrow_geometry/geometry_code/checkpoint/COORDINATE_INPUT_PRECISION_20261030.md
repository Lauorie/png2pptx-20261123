# 模型输入坐标精度修复

修复位置：当前 `working/debug_current`，不改历史冻结实验。定位、SAM 补漏及新增框审核、几何恢复/critic/refine、几何去重和样式请求，统一将模型输入的像素坐标四舍五入到一位小数。

- 示例：`[161.07608695652178,30.597302504816938,411.1847826086956,145.2177263969171]` → `[161.1,30.6,411.2,145.2]`。
- 在 `Model.ask` 生成请求身份、缓存判断和 `prompt.json` 前处理；`build_payload` 再做幂等处理，保证实际 wire 请求一致。四套独立 transport 副本同步覆盖，含兼容发现入口。
- 同时处理结构化 bbox/点/节点/端点，以及定位文字 prompt 内嵌的候选 JSON 和 GLOBAL 观察范围。极窄有效 bbox 不因取整退化为空框。
- 后台几何、坐标变换、模型输出不改精度。格式修复里的原回复及错误文本原样保留；非坐标的缩放因子、线宽策略、置信度和推理参数不受影响。
- 发生修改时保留 `coordinate_input_original_prompt.json` 和 `coordinate_input_precision.json`，便于核对。旧长小数请求不会被当作新请求复用。

验证：`code/tests/test_coordinate_precision.py`，20 项离线检查通过。对 v046 的 1,625 份历史 prompt 只读模拟发送，1,434 份发生坐标精度修改，处理后未发现残留长小数坐标。未调用模型，未重启实验。

逐阶段统计：`analysis/coordinate_input_precision_20261030/offline_check.json`（相对实验根目录）。本次保留了工作副本中已有的 `segmented_geometry.py` / `LINE_PARTS_COMPAT_20261030.md` 改动。
