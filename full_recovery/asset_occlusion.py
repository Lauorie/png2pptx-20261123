"""Use source-visible line evidence to order verified image interiors.

Only measured simple source enclosures participate. A rectangular image bbox
alone never authorizes hiding an external connection.
"""
import numpy as np


def reconcile(assets, arrows, source, dropped, texts=()):
    from .handoff import source_occlusion
    parts={};styles={};references={}
    for index,asset in enumerate(assets):
        e=asset.get('native_enclosure')
        if not e or asset.get('role')!='icon':continue
        theta=np.linspace(0,2*np.pi,96,endpoint=False);angle=np.deg2rad(e['rotation_deg'])
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        radii=np.array(e['size'])/2-e['width_px']/2
        xy=np.column_stack([radii[0]*np.cos(theta),radii[1]*np.sin(theta)])@rot.T+e['center']
        key='asset_surface_'+str(index);oid='asset_'+str(index)
        parts[key]={'id':'interior','object_id':oid,'kind':'ellipse','status':'ok',
                    'commands':[['M',*xy[0]],*[['L',*p] for p in xy[1:]],['Z']]}
        styles[key]={'fill':'#FFFFFF','stroke':None,'width_px':0}
        references[oid+'/interior']='IMG '+asset['id']
    if not parts:return {},[]
    placements,audit=source_occlusion({'parts':parts},styles,arrows,source,set(dropped),
                                     part_placements=True,texts=texts)
    return {ident:{side:[references.get(ref,ref) for ref in refs] for side,refs in spec.items()}
            for ident,spec in placements.items()},audit
