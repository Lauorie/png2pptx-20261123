"""Source representation boundary shared by image and geometry stages."""

IMAGE_OWNERSHIP = (
    'Choose representation from the source content, not crop area or a failed fit. '
    'Simple frames, cells, nodes and reconstructable geometric strokes remain editable shapes. '
    'Photographs, irregular textures and complex geometry with interdependent perspective, '
    'shading or occlusion may use a bounded image when native geometry cannot faithfully '
    'represent that visual unit. Organic curved silhouettes with many interacting lobes '
    '(for example clouds) may also use a bounded edited image when the current native '
    'contour visibly loses those curves. Do not call a visibly polygonal approximation faithful '
    'merely because an editable path exists. The organic silhouette belongs to that visual '
    'unit; it is not automatically an independently protected rectangular frame. '
    'Replace only that unit, retaining independently owned '
    'labels, frames, nodes and connectors. A geometric solid is not a node-link network '
    'merely because its vertices have markers. Missing text or arrow-owned content is '
    'not a shape omission and cannot justify an image fallback. Obvious diagram arrows retain '
    'arrow ownership in every representation; an image proposal or edit cannot reclaim them.')

