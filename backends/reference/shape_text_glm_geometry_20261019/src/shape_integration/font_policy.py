"""Portable font vocabulary shared by the VLM schema and PPTX renderer."""

from __future__ import annotations


DEFAULT_FORMULA_FONT_FAMILY = "Cambria Math"

# Keep this list to ordinary family names commonly available in PowerPoint.
# VLM output is constrained to these exact values; the renderer still applies
# the same policy as a defense against stale or hand-written reports.
PPTX_FONT_FAMILIES = (
    "Aptos",
    "Aptos Display",
    "Aptos Narrow",
    "Arial",
    "Calibri",
    "Cambria",
    "Cambria Math",
    "Courier New",
    "Cascadia Code",
    "Comic Sans MS",
    "DengXian",
    "FangSong",
    "Georgia",
    "KaiTi",
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "NSimSun",
    "Palatino Linotype",
    "Segoe UI",
    "Segoe UI Symbol",
    "SimHei",
    "SimSun",
    "Symbol",
    "Tahoma",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
)
_CANONICAL_FONTS = {family.casefold(): family for family in PPTX_FONT_FAMILIES}


def normalize_formula_font_family(value: object) -> str:
    """Return an allowed font family, falling back to the math default."""

    name = " ".join(str(value or "").strip().split()).strip("'\"")
    return _CANONICAL_FONTS.get(name.casefold(), DEFAULT_FORMULA_FONT_FAMILY)
