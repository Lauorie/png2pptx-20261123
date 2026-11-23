"""Source geometry eligibility shared by localization and fitting."""
from native_qwen.element_protocol import SHAPE_SCOPE as _BASE_SHAPE_SCOPE

SHAPE_SCOPE_VERSION = 'shape_integration.scope.27'
COMPOUND_OWNERSHIP = (
    'A compound represents one coherent icon or diagram motif. Shared alignment or background '
    'does not combine independent motifs; register repeated occurrences separately. '
    'It owns its visible in-scope constituent geometry, including rear layers '
    'and bounded background color patches. Its box covers that complete group. A frame target '
    'owns its contour; distinct enclosed objects retain their own identities. Match each '
    'constituent to its registered owner using the clean source. Each source occurrence is '
    'represented once within this branch. Fit geometry and paint order here; composition '
    'resolves overlap between actual branch products.')
COMPOUND_OWNERSHIP += (
    ' A coherent stack of overlapping parallel sheets/cards is ONE compound with separately '
    'editable back-to-front sheet parts; use exposed corners to count its actual layers. '
    'Do not register each partially hidden backing strip as an independent complete card. '
    'Distinct stacks remain separate owners. Independently registered diagrams/nodes placed '
    'on the front sheet remain separate and do not become intrinsic stack wiring.')
ARROW_OWNERSHIP = (
    'The arrow branch owns standalone arrows and external connector/leader routes, even '
    'without a separately registered arrow ID or endpoint. Cropping or screenshot occlusion '
    'does not turn their visible fragments into intrinsic shape strokes. A visible arrowhead '
    'assigns its complete shaft and head to that branch. Geometric nodes and port/junction circles '
    'belong to shape even when they touch an external route. A registered compound owns its nodes '
    'and complete source-supported internal headless wiring, including self-loops; establish internal '
    'attachment from source topology, not bbox inclusion or a missing arrowhead. Frames, '
    'dividers, grouping braces and intrinsic symbol strokes remain shape geometry. An integral '
    'callout pointer replaces a span of its closed body contour; a separate leader is an external '
    'route. An enclosing panel is a separate container.')
ARROW_OWNERSHIP += (
    ' Filled/block arrows, chevrons used as direction indicators, and broad transition ribbons '
    'with a directional tip are arrows too; a closed outline or solid fill never makes them '
    'shape-owned bodies. Do not register or reconstruct them in this branch. '
    'An internal headless line requires BOTH attachments to be visible inside the owned motif. '
    'A line cut by the observation crop is context, not an eligible internal connection: '
    'request a larger view to establish both ends, or delegate the fragment. '
    'Never infer an endpoint at a crop edge or complete a clipped external route.')
PUNCTUATION_OWNERSHIP = (
    'A standalone ellipsis is one text occurrence, including horizontal, vertical or slanted '
    'dot arrangements; do not add a shape copy. Dotted connector shafts, port markers and '
    'independent geometric nodes retain their actual geometric owner. Judge the source '
    'arrangement and attachment, rather than treating every small dot as punctuation.')
TEXT_OWNERSHIP = (
    'Readable letters, digits and label abbreviations belong to text reconstruction, even when '
    'inside a node, legend swatch or operator icon. Their semantic role does not make their font '
    'outlines shape geometry. Retain the container and non-typographic geometric symbol strokes; '
    'do not trace character outlines as paths or add them during a geometry correction. '
    'When a label masks an otherwise source-supported frame edge, recover the COMPLETE underlying '
    'rectangle/rounded frame, including the span behind the label. Text-background occlusion is '
    'decided from CLEAN at composition, not by cutting shape paths or adding white mask shapes. '
    'A text interruption is not a genuine opening. Preserve real openings and original-image clipping; '
    'never complete a frame without visible continuation or other source evidence. '
    + PUNCTUATION_OWNERSHIP)
SHAPE_SCOPE = _BASE_SHAPE_SCOPE.replace(
    'Exclude standalone arrows/connectors, photographs/image panels, tables, regular cell '
    'matrices and heatmaps: those have separate experiments.',
    'Exclude standalone arrows/connectors and photographs/image panels; for matrices '
    'and heatmaps, retain each easily reconstructable group of simple geometric cells as ONE '
    'compound_symbol and exclude the rest.',
    1,
)
SHAPE_SCOPE = SHAPE_SCOPE.replace(
    'Do not box the entire figure as one symbol.',
    'Do not invent a bounding shape around the entire diagram. A frame or background panel '
    'with its own source-visible stroke or fill boundary is a separate geometric object even '
    'when it encloses the whole diagram; its contents retain their own object identities.',
    1,
)
SHAPE_SCOPE += ' ' + TEXT_OWNERSHIP + ' ' + ARROW_OWNERSHIP + ' ' + COMPOUND_OWNERSHIP
SHAPE_SCOPE += (
    ' Register a compact neural network as one compound with its nodes and internal wiring. '
    'When nodes already have separate IDs, their inter-object connectors remain with the '
    'arrow branch; do not add a shape owner merely to recover those connections. '
    'Geometric nodes and port markers are shape parts, including hollow or filled repeated markers; punctuation within a '
    'readable expression belongs to text. Recover each geometric marker independently of the '
    'route passing through it; clipped straight-route fragments remain with the arrow branch.')
SHAPE_SCOPE += (
    ' Simple table grids have no separate geometry owner in this pipeline: retain their '
    'source-visible borders, dividers and filled cells as compound_symbol geometry; '
    'table words and numbers belong to text reconstruction. Source-visible geometric '
    'color bands inside a compound need their own closed parts when no registered image '
    'asset owns them; do not discard them as background or replace them with annotation marks.')
SHAPE_SCOPE += (
    ' Source-visible braces and brackets that group diagram elements are shape geometry, '
    'including those grouping text-bearing elements. Distinguish these grouping marks from '
    'characters within a text label and from diagnostic overlays added to the review images.')
SHAPE_SCOPE += (
    ' One connected grouping outline has one owner even when stepped or notched. '
    'Its localization box must cover the WHOLE connected contour; do not split it into '
    'overlapping rectangular frames or omit a continuation to fit a primitive template.')
