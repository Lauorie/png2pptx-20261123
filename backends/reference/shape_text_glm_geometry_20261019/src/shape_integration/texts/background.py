"""Text-frame paint, independent of glyph color and surrounding node fill."""
import re


def model_fields(value):
    """Strip CPU-owned paint metadata from outgoing model context, recursively."""
    if isinstance(value,dict):
        return {k:model_fields(v) for k,v in value.items()
                if k not in ('background_fill','background_evidence','background_selection')}
    if isinstance(value,(list,tuple)):
        return [model_fields(v) for v in value]
    return value


def validate(row):
    fill=row.get('background_fill')
    if fill is not None and (not isinstance(fill,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',fill)):
        raise ValueError('background_fill must be null (transparent) or #RRGGBB')
    evidence=row.get('background_evidence','')
    if not isinstance(evidence,str):
        raise ValueError('background_evidence must be a string')
    if fill is not None and not evidence.strip():
        raise ValueError('Solid text-frame fill requires visible background_evidence')
    return {'background_fill':fill.upper() if fill is not None else None,
            'background_evidence':evidence}


def paint(shape, row):
    """Set native shape fill; never insert an extra delivered masking image."""
    from pptx.dml.color import RGBColor
    fill=validate(row)['background_fill']
    if fill is None:shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb=RGBColor.from_string(fill[1:])
