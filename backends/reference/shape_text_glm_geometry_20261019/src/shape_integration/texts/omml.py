"""Convert validated formula semantics into editable PowerPoint OMML.

PowerPoint stores equations as Office Math (OMML) inside an ``a14:m``
extension.  Scene IR keeps the source raster image as the authoritative visual
fallback and records only compact, reviewable formula metadata; the renderer
materializes the OMML at PPTX creation time.
"""

from __future__ import annotations

from copy import deepcopy
import math
import os
import re
from typing import Any, Mapping, Sequence

from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree

from ..font_policy import (
    DEFAULT_FORMULA_FONT_FAMILY,
    normalize_formula_font_family,
)


MATHML_NS = "http://www.w3.org/1998/Math/MathML"
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
DRAWINGML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
PML_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"

OMML_REPORT_FORMAT = "pdf2pptx-formula-omml-v1"


class FormulaConversionError(ValueError):
    """Raised when recognized LaTeX cannot be represented by this converter."""


def normalize_latex_input(latex: str) -> str:
    """Canonicalize a common JSON/VLM over-escaping error.

    A LaTeX command contains one runtime backslash. Structured model clients
    occasionally return two runtime backslashes (four in the JSON file), for
    example ``\\\\mathbb`` instead of ``\\mathbb``. ``latex2mathml`` accepts
    that input as a line break followed by literal text, so conversion appears
    successful while PowerPoint displays ``mathbbR``. Collapse that form for
    command names outside matrix environments, where a double backslash is a
    meaningful row separator.
    """

    expression = str(latex).strip()
    # xcolor's optional model is valid TeX but not understood by latex2mathml.
    # Canonicalize only explicit, bounded numeric models; never guess names.
    def canonical_color(match):
        command, model, value = match.groups()
        if model == 'HTML':
            if not re.fullmatch(r'[0-9A-Fa-f]{6}', value.strip()):
                raise FormulaConversionError('Invalid HTML formula color')
            color = value.strip().upper()
        else:
            parts = value.split(',')
            if len(parts) != 3:
                raise FormulaConversionError('Formula RGB color requires three channels')
            try: channels = [float(v.strip()) for v in parts]
            except ValueError as exc: raise FormulaConversionError('Invalid RGB channels') from exc
            upper = 1 if model == 'rgb' else 255
            if any(not math.isfinite(v) or not 0 <= v <= upper for v in channels):
                raise FormulaConversionError('RGB formula color channel outside range')
            if model == 'RGB' and any(v != int(v) for v in channels):
                raise FormulaConversionError('RGB formula color requires integer channels')
            color = ''.join(f'{round(v*255/upper):02X}' for v in channels)
        return '\\' + command + '{#' + color + '}'
    expression = re.sub(r'\\(textcolor|color)\s*\[\s*(rgb|RGB|HTML)\s*\]\s*\{([^{}]*)\}',
                        canonical_color, expression)
    if "\\\\" not in expression:
        return expression
    if re.search(
        r"\\begin\{(?:array|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases)\}",
        expression,
    ):
        return expression
    expression = re.sub(
        r"\\{2,}(?=[A-Za-z]{2,})",
        lambda _match: "\\",
        expression,
    )
    # Outside an explicit multiline environment, a doubled slash immediately
    # followed by whitespace is a common JSON over-escape of TeX's control
    # space, not a meaningful matrix row break. Leaving it intact makes the
    # converter silently discard intended gaps between numerical columns.
    if not re.search(r'\\begin\{(?:aligned|align\*?|gathered|gather\*?|split|multline\*?)\}', expression):
        expression = re.sub(r'\\{2,}(?=\s)', lambda _match: '\\', expression)
    return expression


def _q(namespace: str, local_name: str) -> str:
    return etree.QName(namespace, local_name)


def _element(namespace: str, local_name: str, **attributes: Any) -> etree._Element:
    node = etree.Element(_q(namespace, local_name))
    for name, value in attributes.items():
        node.set(_q(namespace, name), str(value))
    return node


def _local(node: etree._Element) -> str:
    return etree.QName(node).localname


def _text_run(
    value: str,
    *,
    font_size_pt: float,
    font_family: str,
    color: str,
    plain: bool,
) -> etree._Element:
    run = _element(OMML_NS, "r")
    if plain:
        run_properties = _element(OMML_NS, "rPr")
        run_properties.append(_element(OMML_NS, "sty", val="p"))
        run.append(run_properties)

    drawing_properties = _element(DRAWINGML_NS, "rPr")
    drawing_properties.set("lang", "en-US")
    drawing_properties.set("sz", str(max(100, int(round(font_size_pt * 100)))))
    drawing_properties.set("dirty", "0")
    fill = _element(DRAWINGML_NS, "solidFill")
    rgb = _element(DRAWINGML_NS, "srgbClr")
    rgb.set("val", color.lstrip("#").upper())
    fill.append(rgb)
    drawing_properties.append(fill)
    latin = _element(DRAWINGML_NS, "latin")
    latin.set("typeface", font_family)
    drawing_properties.append(latin)
    run.append(drawing_properties)

    text = _element(OMML_NS, "t")
    text.text = value
    if value[:1].isspace() or value[-1:].isspace():
        text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(text)
    return run


def _noncollapsing_math_space(width: str) -> str:
    """Approximate MathML spacing with Office-preserved Unicode spaces."""

    normalized = str(width or "1em").strip().lower()
    named = {
        "thinmathspace": "\u2006",       # about 1/6 em
        "mediummathspace": "\u2009",     # about 1/5 em
        "thickmathspace": "\u2005",      # about 1/4 em
        "verythinmathspace": "\u200a",
        "verythickmathspace": "\u2004",  # about 1/3 em
    }
    if normalized in named:
        return named[normalized]
    if normalized.startswith("negative"):
        # Absence of a run is the closest portable OMML representation of
        # TeX negative spacing; importantly, never turn it into positive space.
        return ""
    match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+)em", normalized)
    if not match:
        return "\u2003"
    em = float(match.group(1))
    if em <= 0:
        return ""
    whole = int(math.floor(em + 1e-9))
    result = "\u2003" * whole
    remainder = em - whole
    if remainder < 0.05:
        return result
    approximations = [
        (0.5, "\u2002"),
        (1 / 3, "\u2004"),
        (0.25, "\u2005"),
        (0.2, "\u2009"),
        (1 / 6, "\u2006"),
        (0.1, "\u200a"),
    ]
    return result + min(approximations, key=lambda item: abs(item[0] - remainder))[1]


class _MathMLConverter:
    def __init__(self, *, font_size_pt: float, font_family: str, color: str,
                 numeric_font_family: str | None = None) -> None:
        self.font_size_pt = font_size_pt
        self.font_family = font_family
        self.color = color
        self.numeric_font_family = numeric_font_family
        self.tags: set[str] = set()

    def _runs(self, value: str, *, plain: bool) -> list[etree._Element]:
        if not value:
            return []
        return [
            _text_run(
                value,
                font_size_pt=self.font_size_pt,
                font_family=self.font_family,
                color=self.color,
                plain=plain,
            )
        ]

    def _argument(self, name: str, node: etree._Element) -> etree._Element:
        argument = _element(OMML_NS, name)
        argument.extend(self.convert(node))
        return argument

    def _argument_nodes(
        self, name: str, nodes: Sequence[etree._Element]
    ) -> etree._Element:
        argument = _element(OMML_NS, name)
        for node in nodes:
            argument.extend(self.convert(node))
        return argument

    def _script(self, tag: str, node: etree._Element) -> list[etree._Element]:
        children = list(node)
        expected = 3 if tag == "sSubSup" else 2
        if len(children) != expected:
            raise FormulaConversionError(
                f"MathML {_local(node)} requires {expected} operands"
            )
        result = _element(OMML_NS, tag)
        result.append(self._argument("e", children[0]))
        if tag == "sSub":
            result.append(self._argument("sub", children[1]))
        elif tag == "sSup":
            result.append(self._argument("sup", children[1]))
        else:
            result.append(self._argument("sub", children[1]))
            result.append(self._argument("sup", children[2]))
        return [result]

    def _limit(self, tag: str, node: etree._Element) -> list[etree._Element]:
        children = list(node)
        if len(children) != 2:
            raise FormulaConversionError(f"MathML {_local(node)} requires two operands")
        result = _element(OMML_NS, tag)
        result.append(self._argument("e", children[0]))
        result.append(self._argument("lim", children[1]))
        return [result]

    def _matrix(self, node: etree._Element) -> etree._Element:
        matrix = _element(OMML_NS, "m")
        for row_node in list(node):
            if _local(row_node) != "mtr":
                continue
            row = _element(OMML_NS, "mr")
            for cell_node in list(row_node):
                if _local(cell_node) != "mtd":
                    continue
                row.append(self._argument_nodes("e", list(cell_node)))
            matrix.append(row)
        if not list(matrix):
            raise FormulaConversionError("MathML matrix contains no rows")
        return matrix

    def _delimited_matrix(self, node: etree._Element) -> list[etree._Element] | None:
        children = list(node)
        if len(children) != 3 or _local(children[1]) != "mtable":
            return None
        if _local(children[0]) != "mo" or _local(children[2]) != "mo":
            return None
        begin = "".join(children[0].itertext()).strip()
        end = "".join(children[2].itertext()).strip()
        if (begin, end) not in {("(", ")"), ("[", "]"), ("{", "}"), ("|", "|")}:
            return None
        delimiter = _element(OMML_NS, "d")
        properties = _element(OMML_NS, "dPr")
        properties.append(_element(OMML_NS, "begChr", val=begin))
        properties.append(_element(OMML_NS, "endChr", val=end))
        delimiter.append(properties)
        matrix_argument = _element(OMML_NS, "e")
        matrix_argument.append(self._matrix(children[1]))
        delimiter.append(matrix_argument)
        return [delimiter]

    def convert(self, node: etree._Element) -> list[etree._Element]:
        previous = self.color
        color = node.get('mathcolor') or node.get('color')
        if color:
            if not re.fullmatch(r'#[0-9A-Fa-f]{6}', color):
                raise FormulaConversionError('Formula colors must be #RRGGBB: ' + color)
            self.color = color.upper()
        try:
            return self._convert(node)
        finally:
            self.color = previous

    def _convert(self, node: etree._Element) -> list[etree._Element]:
        tag = _local(node)
        self.tags.add(tag)
        if tag in {"math", "mstyle", "semantics"}:
            output: list[etree._Element] = []
            for child in list(node):
                if _local(child) in {"annotation", "annotation-xml"}:
                    continue
                output.extend(self.convert(child))
            return output
        if tag == "mrow":
            matrix = self._delimited_matrix(node)
            if matrix is not None:
                return matrix
            output = []
            for child in list(node):
                output.extend(self.convert(child))
            return output
        if tag in {"mi", "mn", "mo", "mtext"}:
            value = "".join(node.itertext())
            variant = str(node.get("mathvariant", "")).lower()
            if not variant:
                variant=next((str(p.get('mathvariant')).lower() for p in node.iterancestors()
                              if p.get('mathvariant') in ('script','bold-script')), '')
            plain = tag != "mi" or variant in {"normal", "upright"}
            runs=self._runs(value, plain=plain)
            # Text-mode words and whitespace are literal, not StarMath
            # commands. m:sty=p alone does not protect "and" or keep spaces.
            literal = tag == 'mtext'
            numeric_font = (self.numeric_font_family if tag=='mn' and
                            any(_local(p)=='mtd' for p in node.iterancestors()) else None)
            if literal or numeric_font:
                for run in runs:
                    properties=run.find(_q(OMML_NS,'rPr'))
                    if properties is None:
                        properties=_element(OMML_NS,'rPr');run.insert(0,properties)
                    properties.append(_element(OMML_NS,'nor'))
                    if numeric_font:
                        run.find('a:rPr/a:latin',{'a':DRAWINGML_NS}).set('typeface',numeric_font)
            if variant in {'bold','italic','bold-italic'}:
                for run in runs:
                    properties=run.find(_q(OMML_NS,'rPr'))
                    if properties is None:
                        properties=_element(OMML_NS,'rPr');run.insert(0,properties)
                    style=properties.find(_q(OMML_NS,'sty'))
                    if style is None:style=etree.SubElement(properties,_q(OMML_NS,'sty'))
                    style.set(_q(OMML_NS,'val'),{'bold':'b','italic':'i','bold-italic':'bi'}[variant])
                    drawing=run.find(_q(DRAWINGML_NS,'rPr'))
                    if drawing is not None:
                        drawing.set('b','1' if 'bold' in variant else '0')
                        drawing.set('i','1' if 'italic' in variant else '0')
            if variant in {'script','bold-script'}:
                # Keep ASCII letters editable; Office maps script+letter to
                # the mathematical alphabet. Font family alone cannot encode
                # mathcal(LN), and silently falling back to roman is incorrect.
                for run in runs:
                    properties=run.find(_q(OMML_NS,'rPr'))
                    if properties is None:
                        properties=_element(OMML_NS,'rPr');run.insert(0,properties)
                    properties.insert(0,_element(OMML_NS,'scr',val='script'))
                    style=properties.find(_q(OMML_NS,'sty'))
                    if style is None:style=etree.SubElement(properties,_q(OMML_NS,'sty'))
                    style.set(_q(OMML_NS,'val'),'b' if variant=='bold-script' else 'p')
            return runs
        if tag == "mspace":
            # A normal U+0020 is collapsed when PowerPoint normalizes OMML,
            # which turns H(\qquad\qquad) into H().  MathML expresses TeX
            # \quad/\qquad as 1em/2em mspace nodes.  Encode those widths as
            # non-collapsing Unicode em spaces so Office preserves intentional
            # gaps around embedded visual operands.
            value = _noncollapsing_math_space(str(node.get("width", "1em")))
            runs=self._runs(value, plain=True) if value else []
            for run in runs:run.find(_q(OMML_NS,'rPr')).append(_element(OMML_NS,'nor'))
            return runs
        if tag == "msub":
            return self._script("sSub", node)
        if tag == "msup":
            return self._script("sSup", node)
        if tag == "msubsup":
            return self._script("sSubSup", node)
        if tag == "mfrac":
            children = list(node)
            if len(children) != 2:
                raise FormulaConversionError("MathML mfrac requires numerator and denominator")
            fraction = _element(OMML_NS, "f")
            properties = _element(OMML_NS, "fPr")
            control = _element(OMML_NS, "ctrlPr")
            control.append(deepcopy(self._runs('x', plain=False)[0].find(_q(DRAWINGML_NS, 'rPr'))))
            properties.append(control)
            fraction.append(properties)
            fraction.append(self._argument("num", children[0]))
            fraction.append(self._argument("den", children[1]))
            return [fraction]
        if tag in {"msqrt", "mroot"}:
            children = list(node)
            if not children:
                raise FormulaConversionError(f"MathML {tag} has no radicand")
            radical = _element(OMML_NS, "rad")
            properties = _element(OMML_NS, "radPr")
            degree = _element(OMML_NS, "deg")
            if tag == "msqrt":
                properties.append(_element(OMML_NS, "degHide", val="1"))
            elif len(children) == 2:
                degree.extend(self.convert(children[1]))
            else:
                raise FormulaConversionError("MathML mroot requires radicand and degree")
            control = _element(OMML_NS, "ctrlPr")
            control.append(deepcopy(self._runs('x', plain=False)[0].find(_q(DRAWINGML_NS, 'rPr'))))
            properties.append(control)
            radical.append(properties)
            radical.append(degree)
            radical.append(self._argument_nodes("e", children[:1] if tag == "mroot" else children))
            return [radical]
        if tag == "mover":
            children = list(node)
            if len(children) != 2:
                raise FormulaConversionError("MathML mover requires two operands")
            accent_text = "".join(children[1].itertext()).strip()
            if accent_text in {"~", "˜", "^", "ˆ", "¯", "→", "⃗", "˙", "¨"}:
                office_accent = {
                    "~": "\u0303",
                    "˜": "\u0303",
                    "^": "\u0302",
                    "ˆ": "\u0302",
                    "¯": "\u0305",
                    "→": "\u20d7",
                    "⃗": "\u20d7",
                    "˙": "\u0307",
                    "¨": "\u0308",
                }[accent_text]
                accent = _element(OMML_NS, "acc")
                properties = _element(OMML_NS, "accPr")
                # Office Math expects a combining accent code point. ASCII
                # tilde/caret are accepted syntactically but PowerPoint draws
                # them as a misplaced bar through the base glyph.
                properties.append(_element(OMML_NS, "chr", val=office_accent))
                accent.append(properties)
                accent.append(self._argument("e", children[0]))
                return [accent]
            return self._limit("limUpp", node)
        if tag == "munder":
            return self._limit("limLow", node)
        if tag == "munderover":
            children = list(node)
            if len(children) != 3:
                raise FormulaConversionError("MathML munderover requires three operands")
            lower = _element(OMML_NS, "limLow")
            lower.append(self._argument("e", children[0]))
            lower.append(self._argument("lim", children[1]))
            upper = _element(OMML_NS, "limUpp")
            base = _element(OMML_NS, "e")
            base.append(lower)
            upper.append(base)
            upper.append(self._argument("lim", children[2]))
            return [upper]
        if tag == "mtable":
            return [self._matrix(node)]
        if tag in {"mtr", "mtd"}:
            output = []
            for child in list(node):
                output.extend(self.convert(child))
            return output
        if tag == "mfenced":
            delimiter = _element(OMML_NS, "d")
            properties = _element(OMML_NS, "dPr")
            properties.append(_element(OMML_NS, "begChr", val=node.get("open", "(")))
            properties.append(_element(OMML_NS, "endChr", val=node.get("close", ")")))
            properties.append(_element(OMML_NS, "sepChr", val=node.get("separators", ",")[:1]))
            delimiter.append(properties)
            delimiter.append(self._argument_nodes("e", list(node)))
            return [delimiter]
        if tag in {"mpadded", "mphantom"}:
            output = []
            for child in list(node):
                output.extend(self.convert(child))
            return output
        if tag in {"annotation", "annotation-xml"}:
            return []
        raise FormulaConversionError(f"unsupported MathML element: {tag}")


def latex_to_omml(
    latex: str,
    *,
    font_size_pt: float = 12.0,
    font_family: str = DEFAULT_FORMULA_FONT_FAMILY,
    color: str = "#000000",
    numeric_vectors: bool = False,
    vector_gap_em: float | None = None,
    numeric_font_family: str | None = None,
) -> etree._Element:
    """Return one ``m:oMath`` tree for a LaTeX expression."""

    expression = normalize_latex_input(latex)
    # latex2mathml leaves \\arg as the literal token "\\arg", which turns
    # into a question-mark error glyph in Office-compatible renderers.
    expression = re.sub(r'\\arg\b', lambda _: r'\operatorname{arg}', expression)
    if not expression:
        raise FormulaConversionError("LaTeX expression is empty")
    try:
        from .math_text import prepare,restore
        expression,text_spans=prepare(expression)
        # This latex2mathml version emits ASCII '~' for \\sim; that becomes
        # spacing/a literal tilde in Office importers. Only canonicalize the
        # exact mathematical command, after protecting literal text groups.
        expression=re.sub(r'(?<!\\)\\sim(?![A-Za-z])','\u223c',expression)
        mathml = latex_to_mathml(expression)
        source = restore(etree.fromstring(mathml.encode("utf-8")),text_spans)
    except Exception as exc:
        raise FormulaConversionError(f"LaTeX to MathML conversion failed: {exc}") from exc
    if numeric_vectors:
        from .math_vectors import restructure
        restructure(source,vector_gap_em)
    converter = _MathMLConverter(
        font_size_pt=float(font_size_pt),
        font_family=normalize_formula_font_family(font_family),
        color=str(color),
        numeric_font_family=normalize_formula_font_family(numeric_font_family) if numeric_font_family else None,
    )
    math = _element(OMML_NS, "oMath")
    math.extend(converter.convert(source))
    if not list(math):
        raise FormulaConversionError("formula conversion produced empty OMML")
    return math


def estimate_formula_font_size(layer: Mapping[str, Any]) -> float:
    """Estimate an equation font size from the authoritative PDF ink bbox."""

    # Text raster fallbacks retain the original PDF span size. Prefer that
    # typographic measurement over the visible bbox: a bbox containing a base
    # glyph plus subscript is taller than the base font, while PowerPoint OMML
    # will independently scale the subscript again.
    source_size = layer.get("source_font_size_pt")
    try:
        source_size_value = float(source_size)
    except (TypeError, ValueError):
        source_size_value = 0.0
    if math.isfinite(source_size_value) and source_size_value > 0:
        # Superscript/subscript fragments in scientific figures routinely use
        # 2.5--4pt source fonts. Enlarging them to the historical 4pt floor
        # changes both visual scale and baseline relative to adjacent native
        # PDF text. PowerPoint supports these sizes, so preserve the audited
        # source measurement and keep only a defensive 1pt lower bound.
        return round(max(1.0, min(96.0, source_size_value)), 3)

    bbox = layer.get("bbox")
    try:
        height = abs(float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError, IndexError):
        height = 14.0
    scale = os.environ.get("PDF2PPTX_FORMULA_FONT_HEIGHT_RATIO", "0.82")
    try:
        ratio = float(scale)
    except ValueError:
        ratio = 0.82
    if not math.isfinite(ratio) or ratio <= 0:
        ratio = 0.82
    return round(max(4.0, min(96.0, height * ratio)), 3)


def promote_formula_layers(
    layers: Sequence[Mapping[str, Any]],
    *,
    policy: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mark VLM-resolved semantic formula layers for OMML plus fallback."""

    selected = str(
        policy if policy is not None else os.environ.get("PDF2PPTX_FORMULA_OUTPUT", "omml")
    ).strip().lower()
    output = [deepcopy(dict(layer)) for layer in layers]
    report: dict[str, Any] = {
        "format": OMML_REPORT_FORMAT,
        "policy": selected,
        "input_layer_count": len(output),
        "promoted_layer_count": 0,
        "fallback_layer_count": 0,
        "failed_layer_count": 0,
        "records": [],
    }
    if selected in {"", "off", "none", "disabled", "raster", "png"}:
        report["status"] = "skipped_disabled"
        report["fallback_layer_count"] = len(output)
        return output, report
    if selected not in {"omml", "editable", "native"}:
        report["status"] = "skipped_unsupported_policy"
        report["error"] = f"unsupported formula output policy: {selected}"
        report["fallback_layer_count"] = len(output)
        return output, report

    report["promotion_policy"] = (
        "every complete VLM semantic decision is promoted when OMML conversion succeeds"
    )

    for layer in output:
        recognition = layer.get("formula_semantics")
        record = {"id": str(layer.get("id", "")), "path": str(layer.get("path", ""))}
        if not isinstance(recognition, Mapping) or not str(
            recognition.get("latex", "")
        ).strip():
            report["fallback_layer_count"] += 1
            report["records"].append(
                {**record, "status": "raster_fallback", "reason": "missing_vlm_semantics"}
            )
            continue
        latex = normalize_latex_input(str(recognition.get("latex", "")))
        recognition["latex"] = latex
        font_size_pt = estimate_formula_font_size(layer)
        color = str(layer.get("source_color") or recognition.get("color") or "#000000")
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
            color = "#000000"
        try:
            latex_to_omml(latex, font_size_pt=font_size_pt, color=color)
        except FormulaConversionError as exc:
            report["failed_layer_count"] += 1
            report["fallback_layer_count"] += 1
            report["records"].append(
                {**record, "status": "raster_fallback", "reason": "conversion_failed", "error": str(exc)}
            )
            continue
        layer["formula_render"] = {
            "policy": "omml_with_raster_fallback",
            "latex": latex,
            "font_family": DEFAULT_FORMULA_FONT_FAMILY,
            "font_size_pt": font_size_pt,
            "color": color,
            "editable": True,
        }
        source_rotation = layer.get("source_rotation")
        if (
            isinstance(source_rotation, (int, float))
            and math.isfinite(float(source_rotation))
            and abs(float(source_rotation)) > 0.5
        ):
            layer["formula_render"]["rotation"] = float(source_rotation)
        report["promoted_layer_count"] += 1
        report["records"].append(
            {
                **record,
                "status": "promoted",
                "latex": latex,
                "font_size_pt": font_size_pt,
                "source_font_size_pt": layer.get("source_font_size_pt"),
                "confidence": recognition.get("confidence"),
            }
        )
    report["status"] = "complete" if report["failed_layer_count"] == 0 else "partial"
    return output, report
