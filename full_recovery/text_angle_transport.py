"""Normalize equivalent clockwise turns before the existing strict validator."""
from copy import deepcopy
import math

PROTOCOL='clockwise_angle_transport.1'


def normalize(raw):
    result=deepcopy(raw)
    rows=result.get('items',[]) if isinstance(result,dict) else result if isinstance(result,list) else []
    for row in rows:
        if not isinstance(row,dict):continue
        targets=[row,*row.get('units',[])] if isinstance(row.get('units',[]),list) else [row]
        for item in targets:
            if not isinstance(item,dict):continue
            angle=item.get('rotation_deg')
            if type(angle) in (int,float) and math.isfinite(angle) and not -180<=angle<=180:
                item['rotation_deg']=(angle+180)%360-180
    return result


def install():
    from shape_integration.texts import two_stage_crop
    if getattr(two_stage_crop,'_angle_transport_protocol',None)==PROTOCOL:return
    original=two_stage_crop.validate_attributes
    def validate_attributes(raw,*args,**kwargs):
        return original(normalize(raw),*args,**kwargs)
    two_stage_crop.validate_attributes=validate_attributes
    two_stage_crop._angle_transport_protocol=PROTOCOL
