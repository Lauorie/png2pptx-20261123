"""Compact observation rules; scope, contours, layers and output each stated once."""

GEOMETRY_RULES = {
    'scope': (
        'Keep every intrinsic target part and internal boundary. Each cell, including white/unfilled cells, '
        'is a closed part; do not merge cells into an outer silhouette or row/column outline. '
        'Descriptions, annotation boxes and neighbor relations are fallible identity hints, not contours '
        'or proof of ownership. A container does not own everything inside it. Exclude separately represented '
        'peers/image assets, text and external connectors/axes; keep intrinsic symbol strokes without '
        'extending them along attached routes.'
    ),
    'contours': (
        'Observe visible points on stroke centerlines, including thick-stroked filled cells; '
        'without a stroke use the fill boundary. Dashed outlines follow the underlying continuous contour. '
        'One shared boundary uses the same endpoints unless a real gap separates it. '
        'Stroke plus fill is one shape; shading, gradients and shadows alone add no parts. '
        'Read visible samples from the image, not a guessed center/radius/template; preserve observed '
        'deviations rather than regularizing them.'
    ),
    'layers_and_3d': (
        'Complete partially occluded objects from visible continuation, symmetry, repeated instances '
        'and overlap order; return parts back-to-front, not just exposed fragments. Count layers from '
        'exposed edges and offsets; infer hidden structure only where these support it. '
        'A usual cuboid showing three faces has 7 unique projected vertices and 3 closed quadrilateral '
        'faces per solid; adjoining faces share vertex IDs. Cuboid drawings often have horizontal edges '
        'and parallel opposite edges; follow the relationships supported by the image. '
        'Keep stacked solids distinct, with source-supported sizes and offsets; preserve cell divisions '
        'and face seams rather than reducing the stack to its silhouette. '
        'Represent projected faces as closed path polygons. '
        'All geometry remains a 2D projection.'
    ),
    'output': (
        'Use one points ARRAY of {id,xy} and parts referencing it; IDs are unique and every point is used. '
        'Return only fields in the schema, not fitted centers/radii/bboxes/transforms. '
        'ok means complete supported geometry; partial retains valid parts and names omissions. '
        'needs_context uses empty geometry for ambiguous identity, insufficient evidence or completion '
        'outside the crop; never clamp to crop edges. Occlusion alone does not require more context. '
        'unsupported uses empty geometry when no representation is possible. '
        'Omit reason for ordinary visible ok objects; otherwise briefly explain non-ok status or '
        'inferred point/part IDs, visual basis, layer order and material ambiguity.'
    ),
}
