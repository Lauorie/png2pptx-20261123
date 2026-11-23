"""Change only the original exclusion sentence; share it across both stages."""
from native_qwen.element_protocol import SHAPE_SCOPE as _BASE_SHAPE_SCOPE

SHAPE_SCOPE_VERSION = 'shape_integration.scope.3'
SHAPE_SCOPE = _BASE_SHAPE_SCOPE.replace(
    'Exclude standalone arrows/connectors, photographs/image panels, tables, regular cell '
    'matrices and heatmaps: those have separate experiments.',
    'Exclude standalone arrows/connectors, photographs/image panels and tables; for matrices '
    'and heatmaps, retain each easily reconstructable group of simple geometric cells as ONE '
    'compound_symbol and exclude the rest.',
    1,
)
