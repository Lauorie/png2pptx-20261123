# 当前调试基线 v039（2026-10-29）

用户确认冻结当前版本，后续在此基础上逐项debug。本版本是可回退的调试基线，不表示所有case都已修好。

## 纳入的改动

- v036 receiver_v2作为重建代码基础：三类头型、头尺寸固定mid、原图线宽1px下限，样式估计与实际PPT critic/refine；修正版原生渲染及样式JSON接收。
- v037块状入口修复：定位已经输出block_arrow时直接进入shape，不再先调用普通线条几何接口重新分类；GLM shape、crop SAM证据、轮廓fallback和轻量语义审核保持。
- v038圆头说明：独立小圆端点即使贴着大节点边界仍可为圆头；大节点本身、圆线帽和交叉不自动算头。只调整样式说明，普通线条几何表达及几何critic未改。
- 相对v036生产代码仅bbox_geometry.py和simple_style_contract.py变化，另新增块状入口回归测试。

## 固定配置

沿用v036全24图配置：16图、每图8任务、API上限128；默认thinking enabled、不传reasoning_effort；几何、块状几何、样式最多各refine一次。保留每阶段实际crop、坐标和schema。后置范围审核与几何后补漏保持关闭；定位后的SAM补漏及新增框整理来源为v031 receiver_v3_clamp。

定位v022、SAM协议v031的代码/配置也复制在checkpoint/upstream；阶段runner和原始配置保留在checkpoint/runners、checkpoint/configs。凭证仍通过环境变量引用外部文件，未复制密钥。

## 验证与未解决问题

23项离线回归通过；本次冻结没有新模型调用，也未把组合版重跑24图。已有24图结果来自v036 receiver_v2，两个修复分别由v037、v038的小实验验证，不把旧产物称为v039新跑结果。

- 0024、0028：块状箭头填充恢复。
- 0077：R0020圆头恢复并通过样式审核；其他圆头/普通头对照保留。
- 0028普通头、0056串联两段头型未退化。
- 0639：新的GLM轮廓更干净，弯曲方向与头部保留，允许轮廓偏差。
- 0104/R0002：最新shape的头部变钝，几何critic误accept，后续重点debug；不要用该退化结果覆盖旧24图。
- 0122：网络误加头仍存在，用户明确暂缓处理。保留记录，后续不围绕它改prompt或加实验。

## 后续工作位置

本实验目录中的working/debug_current/code是可修改副本，versions/v039_debug_checkpoint_20261029是冻结快照。每次修改前核对CURRENT_DEBUG_VERSION.json；修改后另开同实验下的新run冻结代码和输入。旧runs、旧版本、原24图与外部主管线均保持。

使用本实验tools/debug_version.py status / verify / snapshot / checkout管理调试版本，底层复用工作区tools/version.py。checkout会先自动保存当前工作副本。不要直接编辑versions。

版本来源见provenance.json；测试记录见verification.json；已完成图/PPT/JSON的路径与SHA-256见evidence_index.json。保留外部依赖的绝对路径引用，本快照不是脱离工作区的独立部署包。
