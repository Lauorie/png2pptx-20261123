"""Compact observation rules; scope, contours, layers and output each stated once."""
from ..scope import ARROW_OWNERSHIP, PUNCTUATION_OWNERSHIP

CONTAINER_CONTOUR = ('容器外框沿完整外轮廓恢复，包含标题栏；'
                     '内部标题分隔线或内容区边缘不能替代外框。')

CONNECTION_EVIDENCE = (
    '连接必须有原图可见的线段及对应端点支持。真实遮挡物两侧有同一路径、同走向的可见延续时可以接续，'
    '不能按全连接网络、对称性或重复模板新增不可见的连接或箭头。纹理遮挡处未知不等于有边；'
    '已有拟合也不是证据，检查每条线在无遮挡位置是否真的存在。')
GEOMETRY_RULES = {
    'scope': (
        '只复原标注位置对应的目标，以干净原图为准；旧描述、框和SAM轮廓仅作定位参考。'
        '完整恢复目标职责范围内可见的节点、符号、色块和线迹；compound_symbol 可由多个几何部件构成，但只有一个目标所有者。'
        '容器与内部成员各有边界；位置和标签不决定归属，独立邻居由coverage登记。裁剪图是上下文，SAM外框不改变目标定义。'
        '可读文字交给文字层。'
        '规则纹理只画其封闭表面并说明纹理，合并阶段使用原生填充；离散纯色色块仍须恢复几何面。复杂纹理/图标可交给局部图像。'
        '不要用大量线段模拟纹理，也不要用图片替代普通框和可复原的结构。 '
        + ARROW_OWNERSHIP + ' ' + PUNCTUATION_OWNERSHIP),
    'contours': (
        '有描边时观察中心线，无描边时观察填充边界。位置、宽高、间隔、转角均从原图取点，'
        '不能用标注框当边框。直线、圆弧和折点要区分；矩形/圆/圆角矩形适用时使用原生类型。'
        '一个连通外框只有一条完整轮廓，包括凸出指示角和凹口，不能拆成重叠矩形。' + CONTAINER_CONTOUR +
        '放大/指示框的同色两条斜边若从框边的缺口伸出并交于一个尖端（包括虚线框），'
        '它们是外框一体指示角：沿框和尖端恢复一条完整path，不横跨尖角底部另补框边。'
        '若引线从完整未断的框边伸出，则属外部连线。SAM只分出矩形本体不能覆盖原图中的这个区别。'
        '一个闭合形状的填充和描边由同一part承载。不同纯色区域分别建封闭part，相邻面共享顶点，网格共边只画一次。'
        '投影网格先沿原图逐条数出行列分界；用同一组共享交点连接网格线和色块，不能逐格独立猜角点。'
        '有限的行列分界是结构而非纹理；保留其数量和交点，不能套纹理预设。细密重复的填充纹理才交给样式。'
        '云形等连续曲线必须使用curve段保留实际弧线和凹凸，不用密集短直线拼成多边形。'
        '收缩或展开色块按各自边缘取点，窄端有宽度就保留，不能缩成尖点；旋转梯形可用四角封闭path。'
        '标签遮住的边按可见延续补齐，不画白色遮罩。真实开口和原图裁断保留；'
        '目标超出裁剪图时请求上下文，不能沿裁剪边凭空封口。'),
    'layers_and_3d': (
        '部件按从后到前输出。由可见边缘确定层数、大小和偏移，不套固定立体模板。'
        '重叠卡片逐一依据露出的边角识别前后层；后层的露出条带不是卡片全高，'
        '也不能把前层轮廓再当作后层。用同一层露出的平行边和转角推断被遮住的延续，保持真实偏移。'
        '同一立体的相邻可见面共享边，不应互相穿插；不同物体可真实重叠。'
        '部分遮挡的面可以根据可见延续补齐；完全不可见的面和连接不能虚构。'
        '色带和偏移阴影需要独立封闭面；渐变由样式阶段处理。'
        '圆柱按sampling输出body及共用顶点的top ellipse；投影平面用封闭path，内部线独立且只画一次。'),
    'output': (
        'Use one points ARRAY of {id,xy} and parts referencing it; IDs are unique and every point is used. '
        'Return only fields in the schema, not fitted centers/radii/bboxes/transforms. '
        'ok means complete supported shape geometry; partial names missing shape parts. '
        'needs_context uses empty geometry only for insufficient source context, not a disagreement with old descriptions. '
        'unsupported uses empty geometry when no representation is possible. '
        'Omit reason for ordinary visible ok objects; otherwise identify missing content and its visual evidence.'),
}
