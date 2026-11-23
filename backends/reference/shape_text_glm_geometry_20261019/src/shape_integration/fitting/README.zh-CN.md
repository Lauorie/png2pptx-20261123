# 对象级稀疏观察点拟合

核心入口 `fit_object` 接收一个对象的共享点表和全部 `parts[]`，返回各部件的局部拟合、状态及原图坐标。单图元和复合对象共用入口。该纯拟合入口不调用模型、不选择 crop、不修改定位候选或历史 scene。新增的 `local.py / observation.py` 单独负责局部两图观察接入。

```python
from shape_integration.fitting import FitFrame, fit_object, transform_fit

frame = FitFrame.from_metadata(frame_metadata)
local = fit_object(
    observation,
    frame,
    object_id=candidate_id,
    namespace="case_025/revision_001",
    provenance={"observation_path": "observation.json"},
)
original = transform_fit(local)
```

调用时将实验目录的 `src` 加入 Python 搜索路径。`namespace` 由调用方给出，用于区分图片和修订；原对象、part、点 ID 均原样保留。持久化的 `key` 是 `[namespace, object_id, local_id]`，避免跨对象 ID 冲突。

## 输入

模型使用 `model_observation_schema(input_size)`：仅必填 `status / points / parts`；`reason` 在非 ok 或遮挡推断时给出简短依据，普通成功对象可省略。`points` 是 `{id, xy}` 数组，坐标为当前 crop 的像素中心。建议整数或两位小数，更多小数不会被拒绝；仍校验有限值及范围，不舍入、不截断。

`normalize_observation` 在后台绑定已知 `object_id`，补齐缺省 `reason / uncertain`，再按内部 `observation_schema` 校验。旧格式仍可使用，已有不确定性文本保留；仅白名单元数据 `version` 可移到审计记录，未知字段继续报错。每个成功 revision 保存 `observation_raw.json`、`observation_normalization.json` 和规范化的 `observation.json`；逐调用原始响应与失败也保留。

| part.kind | 观察方式 |
| --- | --- |
| rectangle | `point_ids` 引用 4 个顺时针角点 |
| ellipse | `point_ids` 引用 8 个轮廓点；圆沿用此入口，不强制等半径 |
| rounded_rectangle | `point_ids` 引用 16 个轮廓点；四角共享一个半径，不跨部件共享 |
| path | `segments` 中 line 引用 2 点、curve 引用至少 3 个有序观察点；相邻段复用端点 ID |

`compound_symbol` 是定位对象类型，内部仍用上述 primitive/path 类型表达。点数按 part 校验；父对象的定位 bbox 不参与数值拟合，也不会生成额外轮廓。

`FitFrame` 接收实际局部尺寸、全局尺寸、原图尺寸和全局半开 crop 边界。若 metadata 还带 `source_crop_half_open`，会检查原图裁剪与全局裁剪一致。变换保持 Pillow 半像素约定及浮点精度。

## 状态与错误

- 观察状态 `ok` 要求非空几何；`partial` 可保留部分已观察部件；`needs_context / unsupported` 要求空几何。非 `ok` 必须说明原因。
- 错误点数、未知类型、无效坐标、重复 ID、悬空引用、未使用点和不连续路径会抛出 `ObservationError`。调用方保留原始响应，并将整个对象作为未解决项记录。
- 几何退化、无效矩形角序、优化不收敛等按部件记录为 `failed`，其他合法部件继续拟合。没有旧方法回退。
- `fit_status` 为 `ok / partial / failed / not_run`。即使全部部件求解成功，`completeness_status` 也仅为 `unreviewed`；已知缺失或失败时为 `incomplete`。本模块不产生“视觉完整”结论。
- `bbox_2d` 只包围成功部件，使用路径采样估计；不能当作完整目标范围替代原定位框。定位框由调用方另行保存。

原始 `observation`、点引用、局部参数、求解诊断均保留。`transform_fit` 映射全部路径端点和三次曲线控制点；原图结果中使用 `parameters_local / diagnostics_local` 明确局部单位，不冒充原图原生 PPT 参数。

真实共享点保持同一 ID，并输出共享点在独立拟合中的位置差异；不会自动合并邻近点或强制等距。椭圆/圆角矩形的共享点差异采用最近轮廓采样估计，记录为近似；矩形使用拟合后的对应角点。

## 验证与来源

从实验目录执行：

```bash
python -B tests/test_fitting.py
python -B replay_fitting.py --run-name a_new_offline_run \
  --source ../../tests/test_fit_format/local_points_min1080_two_image_20260920_1520 \
  --source ../../tests/test_fit_format/compound_prompt_001_20260920_1752 \
  --workers 4
```

回放目录必须是新目录。回放仅使用历史保存的原始响应与 frame；冻结输入、旧拟合、新拟合、代码快照、对照图及来源哈希。原始提取来源见 [SOURCE.json](SOURCE.json)，整合范围见 [计划](../../../plans/FITTING_INTEGRATION.zh-CN.md)。

目前定位 `run.py` 仍只执行定位与 image 去重，`fitting_enabled` 保持关闭。局部 clean + bbox overlay 的模型观察已由独立入口 `run_fitting_examples.py` 接通；8 个真实 API 样例及限制见 [GLM 实测报告](../../../runs/fitting_glm_examples_v2/REPORT.zh-CN.md)。

从实验目录执行 `python -B tests/test_fitting_local.py` 检查裁剪、输入隔离和一次请求多 parts。新实验先编辑独立的选样 JSON，再运行 `python -B run_fitting_examples.py --run-name new_name --selection selection.json --prepare-only` 准备输入；检查后去掉 `--prepare-only` 提交。一个 run 仅允许提交一次，失败不自动重试；历史结果保留。


局部观察 prompt 当前为 `local_observation.5`：`rules` 包含四组语义规则，`sampling` 列出 4/8/16 点与 path 的精确约束。投影面使用封闭 path（不把斜四边形强制拟合成 rectangle）；堆叠实例按各自可见面、厚度边、偏移和遮挡顺序补全。返回的仍是二维局部点表，不新增三维参数或立体求解器。修订请求使用同一组规则，只附加修改意见与短的反馈核对要求。
