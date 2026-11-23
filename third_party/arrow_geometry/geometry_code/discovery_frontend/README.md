# 箭头发现主管线

输入 PNG，输出完整箭头/线条身份清单、bbox、来源与判断 trace、整体/子身份只读关系，以及下游重建交接 JSON。当前入口到发现与交接为止，不执行几何恢复或 PPT 导出。

## 流程

1. 全图及重叠分块 GLM 发现，局部 SAM 与 GLM 校准。bbox 相交/包含只提名复核，不能直接合并物理身份。
2. 对可分区域拆分；独立覆盖检查通过后父区域成为上下文，失败或不确定时保留父区域。
3. 核验独立身份，比较新旧候选与历史 owner。明确 same 才合并；整体/完整子身份保留并建立只读关系。全部提名关系处理，不按数量截断。
4. 最后执行分块 SAM 补漏。触及分块/原图边缘的 mask 排除；依据已有覆盖与跨块像素去重筛选，再由 GLM 判断。只追加新身份，原有对象逐字段不变。
5. 输出身份、候选/覆盖核验任务及只读交接。只读框不等于矩形禁绘区；仍需重建穿过该框的其他目标线。

## 运行

Python 3.9 及以上。本工作区已验证环境为 `/mnt/shared-storage-user/yinjinhui/H-conda/anaconda3/envs/internsvg/bin/python`。

```bash
python integrated_run.py prepare --run-dir /absolute/path/new_run --input-dir /absolute/path/png_directory
python integrated_run.py run --run-dir /absolute/path/new_run
```

单张或指定多张图片可用 `--images /path/a.png /path/b.png`。同一目录内文件名主体须唯一；`--input-dir`读取该目录的PNG，不递归扫描其他数据。

准备阶段冻结代码、配置、输入和范围约定，保存 SHA-256 清单。执行自动切换至运行目录的冻结代码；再次运行同一 `run` 命令会恢复已完成阶段与相同请求。不要修改运行中的 code/config/input，也不要再次 prepare 已存在的目录。

默认配置在 `default_config.json`。保持默认思考开启、省略 reasoning_effort、max_tokens=113000、单请求总时限1200秒、API并发16；关系检查不设条数上限。SAM 服务和模型 API 连接由配置提供；API凭证沿用 transport/credentials.py 的私有配置读取机制，不写入运行产物。

## 输出

- `reports/完整结果对照.md`：所有输入的原图/结果对照及只读身份示例。
- `cases/<case>/result.json`：最终身份、待核验计数、SAM追加数量和各阶段结果链接。
- `cases/<case>/reconstruction_handoff.json`：重建任务、只读身份、需复核标记及退役父区域上下文。
- `stages/{discovery,partition,owners,post_sam}/`：全部阶段材料；每次模型请求保留输入、原始回复、reasoning、验证/格式修复和请求参数。
- `progress.json` 与 `cases/<case>/status.json`：全局进度和单图当前阶段。
- `manifest.json`：运行采用的代码、配置、输入哈希。

`completed`表示流程执行结束。`pending_identity_count`、`pending_relation_count`、`post_sam_pending_count`及`issues`应继续检查，不能用任务数证明召回率。`requires_review=true`的交接任务仍需覆盖或身份复核。

## 受控回放

复用输入图而重新运行算法：

```bash
python integrated_run.py prepare --run-dir /path/new_run --from-run /path/old_run --cases task_0320 task_0183
```

固定某个已完成阶段，验证下游：显式增加 `--seed-stage discovery|partition|owners --seed-run /path/source_stage_run`。源图像哈希必须一致；种子结果单独冻结并标明复用，不声称新发现或同prompt复用。

## 原则性修复

- 提示文字中的未标帧坐标在发送前移除，保留原文与清理audit；bbox仍由后台精确换算。节点编号、n=3、3x3等语义数字保留。
- 同一owner历史明确same的single观察保持关联，不再直接创建第二个任务；未标关系的历史记录保守复核。
- false只读匹配视为无匹配null；核验中的显式null视为未通过，不丢弃同批有效结果。true但未给出具体只读ID仍拒绝。
- 格式修复与请求重试使用递增attempt目录，保留旧请求文件。
- 整体—子身份不强行same合并；即使双方kind=single，也会提供已核验子身份的只读标记。矛盾的成员循环不能形成相互排除。
- 未经独立覆盖核验的组范围建议不能改写父任务责任；原始区域和来源保留。

## 验证

```bash
python -m unittest discover -p 'test_*.py'
```

发布验证记录由主管线根目录的合入报告提供。历史16图关系回放、四图SAM补漏和真实局部PNG全流程验证，分别衡量程序一致性、已有补漏行为和运行连通性；不替代新版本125图全量质量评测。
