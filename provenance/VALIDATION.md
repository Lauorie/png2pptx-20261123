# 交接验证：2026-11-19

本记录针对交接目录中的代码、权重与部署方式，原目录保留原样。

## 已通过

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 实际权重复制 | 19 个文件，45,152,405,448 字节；复制时逐文件核对源与目标 SHA-256 | `weights_manifest.json`、`weight_copy_status.json` |
| 完整任务准备 | 图形、文字、geometry-first 箭头与合成准备成功；禁止读取原代码和实验目录；活动箭头配置无旧批次路径和历史复用开关 | `prepare_check.json`、`scripts/check_prepare.py` |
| 主流程相关回归 | 40 passed，2 subtests passed | `pytest.log` |
| 请求优先 / 低功率保活 | 15 tests，OK；含流式响应、取消、SAM busy / health 行为 | `keepalive_tests.log` |
| 原生渲染 | 同一个临时原生 PPT 经文字和箭头两套渲染器成功生成 PDF，单页文字可提取 | `render_check.json`、`scripts/check_render.py` |
| 字体 | 5 个常用字体家族及数学 / 符号 fallback 哈希和实际解析匹配 | `render_check.json` |
| 箭头渲染器版本 | LibreOffice 26.8.0.3，build `bce0998afefdbc355585ca324285661a2170ba77` | `render_check.json` |
| SenseNova Python runtime | 从包内 overlay 在临时目录重建成功；torch、transformers、diffusers、tokenizers 实际 import 成功 | `sensenova_runtime_import_check.json` |
| GPU 部署参数 | SAM 与 SenseNova 均完成新目录准备及提交命令 dry-run；Shell 语法通过 | `sam_submit_dry_run.txt`、`sensenova_submit_dry_run.txt` |
| 密钥清理 | 无凭据文件；字面量扫描的两处匹配均为渲染器自带 SPDX license 标识 | `credential_scan.json` |
| 文件完整性 | 最终文件、大小、SHA-256 与目录内软链接列入清单；软链接目标位于包内 | `bundle_manifest.json` |

主流程回归命令：

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_full_api.py tests/test_full_arrow_methods.py \
  tests/test_full_text_occlusion.py tests/test_fresh_pipeline.py
python -B -m unittest discover -s deploy/sensenova/code_snapshot/tests -v
```

离线验证使用独立 CPython 3.12.13 环境，依赖快照为根目录的
`requirements.cpu-validated.txt`。该环境是本次检查环境，不冒充原集群 CPU / SAM 环境。

## 验证范围

任务准备测试使用本地模拟 SAM health、临时占位凭据与临时 PNG。
渲染测试使用人工构造的原生形状和文字 PPT。
本次没有新提交 rjob、重启现有服务或消耗多模态 API / GPU 推理额度；
没有从真实 PNG 新跑一遍恢复，因此没有新增端到端视觉准确率结论。

SAM、SenseNova 的基础 Conda 环境与 CUDA 保持外部挂载；本包包含实际权重、
推理代码、SenseNova 定制 overlay、启动器及环境说明。
替换这些基础环境或搬到其他集群后，需要重新核验服务加载、真实分割、
真实图片编辑及最终 PPT 渲染。
