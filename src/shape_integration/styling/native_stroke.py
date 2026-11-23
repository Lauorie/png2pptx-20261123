"""Fit fixed-geometry stroke templates in native upsampled SAM crops.

SAM supplies observed foreground cross-sections, rather than a yes/no profile
veto. CLEAN pixels, conditioned on the selected stroke colour, separate the
stroke from fill and shadow. All returned lengths are original-image pixels.
"""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.optimize import minimize_scalar
from ..common import read, save, sha, digest
from ..fitting.paths import sample_commands
from .geometry import bounds, closed, subpaths, footprint
from . import lines

PROTOCOL = 'native_sam_rgb_template.27'


def sample(array, xy, order=1):
    xy = np.asarray(xy)
    if array.ndim == 2:
        return map_coordinates(array, [xy[..., 1], xy[..., 0]], order=order, mode='nearest')
    return np.stack([map_coordinates(array[..., c], [xy[..., 1], xy[..., 0]],
                                    order=order, mode='nearest') for c in range(3)], axis=-1)


def side_backgrounds(rgb, offsets, stroke):
    """Estimate each background over source-pixel lengths, not sample counts.

    At native SAM resolution four samples cover less than half a source
    pixel. A neighboring thin line can fill that entire probe. A four-pixel
    side window keeps the background median independent of sampling density.
    """
    offsets=np.asarray(offsets,float)
    span=min(4.,float(offsets[-1]-offsets[0])/4)
    broad=np.stack([np.median(rgb[:,offsets<=offsets[0]+span],axis=1),
                    np.median(rgb[:,offsets>=offsets[-1]-span],axis=1)],axis=1)
    edge=np.stack([np.median(rgb[:,:4],axis=1),np.median(rgb[:,-4:],axis=1)],axis=1)
    # A broad foreground can fill most of the side window. Its stroke colour
    # is not background; retain the contrasting far edge when it is observed.
    use_edge=(np.linalg.norm(broad-stroke,axis=-1)<25.)&(np.linalg.norm(edge-stroke,axis=-1)>25.)
    result=np.where(use_edge[...,None],edge,broad)
    # A nearby enclosing frame can put BOTH far probes beyond its fill. Then
    # the inner stroke, intervening grey fill and outer stroke form one alpha
    # band. Prefer a stable local background only when an independently darker
    # central stroke separates it from the target, never the stroke's own core.
    central=rgb[:,abs(offsets)<=2.]
    if not central.shape[1]:return result
    distances=np.linalg.norm(central-stroke,axis=-1)
    core=central[np.arange(len(rgb)),np.argmin(distances,axis=1)]
    for side,sign in enumerate((-1,1)):
        band=(offsets*sign>=3.)&(offsets*sign<=6.)
        if band.sum()<4:continue
        values=rgb[:,band];local=np.median(values,axis=1)
        spread=np.quantile(np.linalg.norm(values-local[:,None],axis=-1),.9,axis=1)
        old=result[:,side];ray=stroke-old
        projection=((local-old)*ray).sum(-1)/np.maximum((ray*ray).sum(-1),1.)
        use=(spread<8)&(np.linalg.norm(local-old,axis=-1)>20)&(projection>0)&(projection<.8)
        use&=(np.linalg.norm(core-local,axis=-1)>25)&(np.linalg.norm(core-stroke,axis=-1)+25<np.linalg.norm(local-stroke,axis=-1))
        result[:,side]=np.where(use[:,None],local,old)
    return result


def profile_alpha(rgb, offsets, stroke):
    """Colour-conditioned alpha; off-colour shadow/text is unknown foreground."""
    rgb = np.asarray(rgb, np.float32)
    sides = side_backgrounds(rgb,offsets,stroke)
    bg = np.where((offsets <= 0)[None, :, None], sides[:, :1], sides[:, 1:])
    delta = np.asarray(stroke, np.float32) - bg
    den = np.maximum((delta * delta).sum(-1), 1.)
    raw = ((rgb - bg) * delta).sum(-1) / den
    # A darker crossing on the same colour ray is not an empty dash gap.
    residual = np.linalg.norm(rgb - (bg + np.maximum(raw, 0)[..., None] * delta), axis=-1)
    # A raster border pixel may mix stroke AND both side backgrounds. A hard
    # background switch at the fitted path creates a false hole when geometry
    # is fractionally offset. Project onto the two-background colour triangle.
    left=sides[:,0];right=sides[:,1];u=np.asarray(stroke,np.float32)-left;v=right-left
    uu=(u*u).sum(-1);vv=(v*v).sum(-1);uv=(u*v).sum(-1);det=uu*vv-uv*uv
    eligible=(vv>25**2)&(det>.005*np.maximum(uu*vv,1))
    if eligible.any():
        for background in (left,right):
            direction=np.asarray(stroke,np.float32)-background
            t=np.maximum(((rgb-background[:,None])*direction[:,None]).sum(-1)/np.maximum((direction*direction).sum(-1)[:,None],1),0)
            error=np.linalg.norm(rgb-background[:,None]-t[...,None]*direction[:,None],axis=-1)
            use=eligible[:,None]&(error<residual);raw=np.where(use,t,raw);residual=np.minimum(residual,np.where(eligible[:,None],error,np.inf))
        z=rgb-left[:,None];zu=(z*u[:,None]).sum(-1);zv=(z*v[:,None]).sum(-1)
        a=(zu*vv[:,None]-zv*uv[:,None])/np.maximum(det[:,None],1)
        b=(zv*uu[:,None]-zu*uv[:,None])/np.maximum(det[:,None],1)
        error=np.linalg.norm(z-a[...,None]*u[:,None]-b[...,None]*v[:,None],axis=-1)
        use=eligible[:,None]&(a>=0)&(b>=0)&(a+b<=1)&(error<residual)
        raw=np.where(use,a,raw);residual=np.where(use,error,residual)
        # Background-only mixtures must not become a fictitious border.
        t=np.clip(zv/np.maximum(vv[:,None],1),0,1)
        error=np.linalg.norm(z-t[...,None]*v[:,None],axis=-1)
        use=eligible[:,None]&(error<residual);raw=np.where(use,0,raw);residual=np.where(use,error,residual)
    valid = residual < 28.
    alpha = np.clip(raw, 0., 1.)
    alpha[~valid] = 0.
    return alpha, valid, sides


def fit_profiles(rgb, offsets, mask, mask_known, stroke, scale=10.5, review=False,width_preference=None):
    """Robust native-crop fit with separate RGB and SAM residuals."""
    offsets = np.asarray(offsets, float)
    step = float(offsets[1] - offsets[0])
    requested_stroke=np.asarray(stroke,np.float32)
    # If the requested colour is a pale antialias sample, an opaque core can
    # lie beyond its colour ray. Measure with the observed persistent core;
    # do not compensate the rendered line width for a colour contrast error.
    sides=side_backgrounds(rgb,offsets,requested_stroke);side_left=sides[:,0];side_right=sides[:,1]
    backgrounds=(side_left+side_right)/2
    direction=requested_stroke-backgrounds;den=np.maximum((direction*direction).sum(-1),1.)
    projection=((rgb-backgrounds[:,None])*direction[:,None]).sum(-1)/den[:,None]
    residual=np.linalg.norm(rgb-backgrounds[:,None]-np.maximum(projection,0)[...,None]*direction[:,None],axis=-1)
    # A side background can lie beyond a pale requested colour when the two
    # sides differ. It is not an opaque stroke core: require contrast against
    # BOTH backgrounds before using a pixel to recalibrate the pigment.
    core_contrast=np.minimum(np.linalg.norm(rgb-side_left[:,None],axis=-1),
                             np.linalg.norm(rgb-side_right[:,None],axis=-1))
    plausible=(abs(offsets)[None]<=5)&(residual<np.maximum(8.,.08*np.sqrt(den))[:,None])&(core_contrast>25.)
    score=np.where(plausible,projection-.015*abs(offsets)[None],-1e6)
    indexes=np.argmax(score,axis=1);peak_projection=projection[np.arange(len(rgb)),indexes]
    enough=score.max(axis=1)>0.5
    calibrated=False
    if enough.sum()>=12 and np.median(peak_projection[enough])>1.15:
        cores=rgb[np.arange(len(rgb))[enough],indexes[enough]];observed=np.median(cores,axis=0)
        if np.linalg.norm(observed-requested_stroke)>25 and np.median(np.linalg.norm(cores-observed,axis=-1))<30:
            stroke=observed;calibrated=True
    alpha, valid, _ = profile_alpha(rgb, offsets, stroke)
    widths, centres, rows, peaks = [], [], [], []
    for i, a in enumerate(alpha):
        candidates = []
        for left, right in lines.runs((a > .25) & valid[i]):
            centre = (offsets[left] + offsets[right - 1]) / 2
            if left > 0 and right < len(offsets) and abs(centre) <= 5.:
                candidates.append((abs(centre)-2*float(a[left:right].max()), left, right))
        if not candidates:
            continue
        _, left, right = min(candidates)
        while left > 0 and a[left - 1] > .04 and valid[i, left - 1]:
            left -= 1
        while right < len(a) and a[right] > .04 and valid[i, right]:
            right += 1
        if left == 0 or right == len(a):
            continue
        width = float(a[left:right].sum() * step)
        if width <= .05:
            continue
        centres.append(float(np.sum(offsets[left:right] * a[left:right]) / max(a[left:right].sum(), 1e-6)))
        widths.append(width); rows.append(i);peaks.append(float(a[left:right].max()))
    if len(rows) < 4:
        return {'measured': False, 'reason': 'insufficient_colour_matched_stroke', 'samples': len(rows)}
    widths = np.asarray(widths); centres = np.asarray(centres);peaks=np.asarray(peaks);rows=np.asarray(rows)
    # A weak antialias fragment is not a second physical border-width mode.
    # Relative strength preserves genuinely faint/subpixel strokes when the
    # entire boundary is faint, while preferring complete observed profiles.
    strong=peaks>=.7*np.quantile(peaks,.9)
    weak_profiles=int((~strong).sum())
    if strong.sum()>=max(8,.25*len(rows)):
        widths=widths[strong];centres=centres[strong];rows=rows[strong]
    # Persistent shaft/border width is the mode; broad joins and filled regions
    # must not turn the whole contour into a broad line.
    bins = np.rint(widths / .2).astype(int)
    counts=np.bincount(bins);mode = float(np.argmax(counts) * .2)
    # A cast shadow affects only part of a closed boundary. Prefer a narrower
    # persistent core only when it is a competing mode with substantial support.
    modes=[]
    for value in np.flatnonzero(counts):
        centre=value*.2; mass=np.abs(widths-centre)<=max(.3,.2*centre)
        if mass.sum()>=max(8,.25*len(widths)):modes.append((centre,int(mass.sum())))
    if modes:
        narrow=min(modes)[0]
        if width_preference=='increase':mode=max(modes)[0]
        elif width_preference=='decrease':mode=narrow
        elif mode>1.5*max(narrow,.1):mode=narrow
        # Resolve competing RGB modes against independent, finite SAM bands.
        # A filled silhouette touching the profile boundary is not a band.
        mask_modes=[]
        for j,i in enumerate(rows):
            if not mask_known[i]:continue
            spans=[(a,b) for a,b in lines.runs(mask[i]>.5)
                   if a>0 and b<len(offsets) and
                   abs((offsets[a]+offsets[b-1])/2-centres[j])<=1.5]
            if spans:
                a,b=min(spans,key=lambda ab:abs((offsets[ab[0]]+offsets[ab[1]-1])/2-centres[j]))
                mask_modes.append((b-a)*step)
        if len(mask_modes)>=12:
            mw=float(np.median(mask_modes))
            agreeing=[value for value,mass in modes if abs(value-mw)<=max(.75,.35*mw)]
            # The critic already inspected CLEAN and the actual render. A
            # wider SAM band must not undo its explicit narrower-mode choice.
            if width_preference=='decrease':agreeing=[v for v in agreeing if v<=mode+1e-9]
            elif width_preference=='increase':agreeing=[v for v in agreeing if v>=mode-1e-9]
            if agreeing and np.median(abs(np.asarray(mask_modes)-mw))<=max(.3,.25*mw):
                mode=min(agreeing,key=lambda value:abs(value-mw))
    selected = np.abs(widths - mode) <= max(.3, .25 * mode)
    if selected.sum() < 4:
        selected = np.ones(len(widths), bool)
    keep = np.flatnonzero(selected)
    rgb_width = float(np.median(widths[selected]))
    aligned = np.asarray([np.interp(offsets + centres[j], offsets, alpha[rows[j]], left=0, right=0) for j in keep])
    observed_rgb = np.median(aligned, axis=0)
    observed_mask, mask_widths = [], []
    for j in keep:
        i = rows[j]
        if not mask_known[i]:
            continue
        m = mask[i] > .5
        components = [(a, b) for a, b in lines.runs(m) if a > 0 and b < len(offsets)
                      and abs((offsets[a] + offsets[b - 1]) / 2 - centres[j]) <= max(1.5, rgb_width)]
        if not components:
            continue  # A silhouette covering one whole side is not a line-width observation.
        a, b = min(components, key=lambda ab: abs((offsets[ab[0]] + offsets[ab[1] - 1]) / 2 - centres[j]))
        mw = float((b - a) * step)
        if abs(mw - rgb_width) > max(.75, .5 * rgb_width):
            continue  # A filled body or a merged neighbouring stroke has a different role.
        selected_mask=np.zeros_like(m,dtype=float);selected_mask[a:b]=1.
        mask_centre=(offsets[a]+offsets[b-1])/2
        observed_mask.append(np.interp(offsets + mask_centre, offsets, selected_mask, left=0, right=0))
        mask_widths.append(mw)
    target_mask = np.median(observed_mask, axis=0) if observed_mask else None
    mask_weight = .5 if len(observed_mask) >= 4 else .0

    def loss(width, return_parts=False):
        # CLEAN is a raster image, with an original-pixel antialias transition.
        pred_rgb = np.clip(width / 2 + .5 - np.abs(offsets), 0, 1)
        rgb_loss = float(np.mean(np.minimum(abs(pred_rgb - observed_rgb), .3)))
        mask_loss = 0.
        if target_mask is not None:
            x = np.clip((width / 2 - abs(offsets)) * scale / .65, -60, 60)
            pred_mask = 1 / (1 + np.exp(-x))
            mask_loss = float(np.mean(abs(pred_mask - target_mask)))
        total = (1 - mask_weight) * rgb_loss + mask_weight * mask_loss
        return (total, rgb_loss, mask_loss) if return_parts else total

    lower, upper = max(.08, rgb_width * .55), min(16., max(rgb_width * 1.55, rgb_width + .6))
    if rgb_width > 16. or lower >= upper:
        # A broad face/crossing can fill almost the entire sampling corridor.
        # It is outside this line template's supported range, not a width to
        # force to the upper limit. Preserve the semantic width and let the
        # existing source/render critic review the visible stroke.
        return {'measured': False, 'present': True, 'width_reliable': False,
                'reason': 'observed_profile_outside_supported_width_range',
                'rgb_width_px': round(rgb_width, 4), 'fit_bounds_px': [lower, upper],
                'supported_width_range_px': [.08, 16.]}
    fit = minimize_scalar(loss, bounds=(lower, upper), method='bounded', options={'xatol': .0005})
    width = float(fit.x)
    total, rgb_loss, mask_loss = loss(width, True)
    spread = float(np.median(abs(widths[selected] - rgb_width)))
    weak_displaced=(selected.sum()<.35*len(rgb) and np.median(peaks)<.5 and
                    float(np.median(abs(centres[selected])))>max(2.,2*width))
    return {'measured': True, 'present': True, 'width_px': round(width, 4),
            'source_colour': lines.hexcolor(stroke), 'rgb_width_px': round(rgb_width, 4),
            'requested_stroke_colour':lines.hexcolor(requested_stroke),'measurement_colour_calibrated':calibrated,
            'native_mask_width_px': round(float(np.median(mask_widths)) * scale, 3) if mask_widths else None,
            'mask_width_source_px': round(float(np.median(mask_widths)), 4) if mask_widths else None,
            'native_to_source_scale': scale, 'mask_width_samples': len(mask_widths), 'rgb_width_samples': int(selected.sum()),
            'geometry_offset_px': round(float(np.median(abs(centres[selected]))), 4),
            'signed_geometry_offset_px': round(float(np.median(centres[selected])), 4),
            'width_mad_px': round(spread, 4),
            'width_reliable': bool(not weak_displaced and selected.sum() >= max(12, .05 * len(rgb)) and spread < max(.3, width * .3)),
            'width_reliability_reasons': ['weak_or_displaced_colour_fragments'] if weak_displaced else [],
            'profile_width_quantiles': [round(float(v),3) for v in np.quantile(widths,[.1,.25,.5,.75,.9])],
            'selected_mode_fraction': round(float(selected.mean()),3),
            'weak_antialias_profiles':weak_profiles,
            'loss': round(total, 7), 'rgb_loss': round(rgb_loss, 7), 'mask_loss': round(mask_loss, 7),
            'mask_loss_weight': mask_weight, 'critic_refit': bool(review)}


def classify_pattern(rgb, offsets, stroke, rows, centre=0., width=2., mask=None, known=None):
    alpha, valid, _ = profile_alpha(rgb, offsets, stroke)
    # Real CLEAN gaps determine the pattern. A missing SAM component is not a gap.
    # Sample the observed band, not a fixed corridor around a biased fitted
    # contour. Geometry stays fixed; only the measurement coordinates move.
    band=(np.abs(offsets)<=3.) | (np.abs(offsets-centre)<=max(.5,min(1.5,width/2)))
    if not band.any():band[np.argmin(abs(offsets-centre))]=True
    signal = alpha[:, band].max(axis=1)
    # As in the reference stroke-to-mask fitter, compare the source ink with
    # the associated thin mask band. A filled silhouette is not a stroke band;
    # absent masks cannot manufacture empty gaps. Contradictory ink is unknown.
    localized=np.zeros(len(signal),bool);conflict=np.zeros(len(signal),bool)
    mask_on=np.zeros(len(signal),bool);mask_usable=np.zeros(len(signal),bool)
    if mask is not None and known is not None:
        mask_usable=known[:,band].any(1) & ~(np.asarray(mask)[:,band]>.5).any(1)
        for i,m in enumerate(np.asarray(mask)>.5):
            spans=[(a,b) for a,b in lines.runs(m & known[i]) if a>0 and b<len(offsets)
                   and (offsets[b-1]-offsets[a])<=max(3.,2.5*width)
                   and abs((offsets[a]+offsets[b-1])/2-centre)<=max(1.5,width)]
            if not spans:continue
            a,b=min(spans,key=lambda ab:abs((offsets[ab[0]]+offsets[ab[1]-1])/2-centre))
            own=(offsets>=offsets[a]-.75)&(offsets<=offsets[b-1]+.75)&band
            if not own.any():continue
            mask_on[i]=True;mask_usable[i]=True
            matched=float(alpha[i,own].max());localized[i]=matched>=.08
            conflict[i]=signal[i]>=.08 and matched<.25*signal[i]
            if not conflict[i]:signal[i]=matched
    level = float(np.quantile(signal, .9)) if len(signal) else 0.
    if level < .08:
        return {'dash': 'solid', 'dash_reliable': False, 'dash_reason': 'weak_colour_signal'}
    on_all, gaps_all, painted, total = [], [], 0, 0
    # A different colour beside the shaft is not an occluder of its centre.
    # Testing the whole corridor erased real gaps near coloured annotations.
    core=abs(offsets-centre)<=max(.5,min(1.5,width/2))
    if not core.any():core[np.argmin(abs(offsets-centre))]=True
    foreign=((~valid[:,core]).mean(axis=1)>.5)&(signal<.5*level) | conflict
    for row in rows:
        ids = row['indices']; usable = ~row['corner'] & ~foreign[ids]
        on = signal[ids] >= .5 * level
        step = row['step']
        for a, b in lines.runs(usable):
            marks, gaps = lines.pattern(on[a:b], step)
            on_all.extend(marks); gaps_all.extend(gaps)
            painted += int(on[a:b].sum()); total += b - a
    result={'on_lengths': on_all, 'gap_lengths': gaps_all, 'coverage': painted / max(total, 1),
            'signal_peak': round(level, 3),'pattern_sam_matched_samples':int(localized.sum()),
            'pattern_sam_rgb_conflicts':int(conflict.sum()),'pattern_foreign_samples':int(foreign.sum())}
    # A mask can propose a periodic template when RGB run boundaries were
    # split by unrelated side colours. Match that template against CLEAN:
    # both its painted marks AND its empty gaps need independent RGB support.
    # Mask holes in a continuous source stroke therefore cannot create dashes.
    if localized.sum()>=12:
        marks=[];gaps=[];valid_ids=[];painted=total=0
        for row in rows:
            ids=row['indices'];usable=~row['corner']&mask_usable[ids]&~conflict[ids]
            for a,b in lines.runs(usable):
                selected=ids[a:b];o,g=lines.pattern(mask_on[selected],row['step'])
                marks+=o;gaps+=g;painted+=int(mask_on[selected].sum());total+=b-a
                valid_ids.extend(selected.tolist())
        coverage=painted/max(1,total)
        proposed=lines.classify(marks,gaps,coverage,coverage,width)
        ids=np.asarray(valid_ids,int)
        matched=ids[mask_on[ids]]
        empty=ids[~mask_on[ids] & (valid[ids][:,core].mean(1)>=.5)]
        if proposed.get('dash_reliable') and proposed['dash']!='solid' and len(empty)>=12 and len(matched)>=12:
            on_agreement=float(np.mean(signal[matched]>=.5*level))
            gap_agreement=float(np.mean(signal[empty]<.5*level))
            result['sam_template_rgb_agreement']={'marks':round(on_agreement,4),'gaps':round(gap_agreement,4)}
            if min(on_agreement,gap_agreement)>=.8:
                result['sam_pattern_candidate']=proposed
    return result


class NativeFitter:
    def __init__(self, source, table, evidence_root, case_id, folder):
        self.source = Path(source); self.image = Image.open(source).convert('RGB')
        self.source_rgb = np.asarray(self.image, np.float32)
        self.table = table; self.folder = Path(folder) / 'native_fits'
        path = Path(evidence_root) / case_id / 'line_evidence.json' if evidence_root else None
        if path is not None and not path.is_file():
            raise ValueError('Requested SAM evidence is missing: '+str(path))
        self.evidence = read(path) if path and path.exists() else {'parts': {}, 'size': 1008}
        if path is not None and self.evidence.get('case')!=case_id:
            raise ValueError('SAM evidence case does not match the geometry')
        self.scale = self.evidence['size'] / self.evidence.get('config', {}).get('side', 96)
        self.identity = {'source': sha(source), 'evidence': sha(path) if path and path.exists() else None,
                         'protocol': PROTOCOL, 'code': sha(Path(__file__))}

    @lru_cache(maxsize=6)
    def tile_rgb(self, directory):
        with Image.open(Path(directory) / 'tile.png') as im:
            return np.asarray(im.convert('RGB'), np.float32)

    @lru_cache(maxsize=12)
    def tile_mask(self, path):
        with Image.open(path) as im:
            return np.asarray(im.convert('L'), np.float32) / 255.

    def read_pixels(self, key, xy, masks=False):
        xy = np.asarray(xy); shape = xy.shape[:-1]
        q = xy.reshape(-1, 2)
        values = sample(self.source_rgb, q)
        ownership = np.zeros(len(q), np.float32); known = np.zeros(len(q), bool)
        tiles = self.evidence['parts'].get(key, {}).get('tiles', [])
        # The closest crop centre owns a sample; do not blend masks across parts.
        best = np.full(len(q), np.inf)
        mask_best = np.full(len(q), np.inf)
        for tile in tiles:
            x0, y0, side = tile['box']; centre = np.array([x0 + side / 2, y0 + side / 2])
            distance = np.sum((q - centre) ** 2, axis=1)
            use = (q[:, 0] >= x0) & (q[:, 0] < x0 + side) & (q[:, 1] >= y0) & (q[:, 1] < y0 + side) & (distance < best)
            scale = self.evidence['size'] / side
            if use.any():
                local = (q[use] - [x0, y0] + .5) * scale - .5
                values[use] = sample(self.tile_rgb(tile['dir']), local)
                best[use] = distance[use]
            path=tile.get('target_mask')
            if masks and tile['target'] and path:
                valid=(q[:,0]>=x0)&(q[:,0]<x0+side)&(q[:,1]>=y0)&(q[:,1]<y0+side)&(distance<mask_best)
                if valid.any():
                    local=(q[valid]-[x0,y0]+.5)*scale-.5
                    ownership[valid]=sample(self.tile_mask(path),local,0)
                    known[valid]=True;mask_best[valid]=distance[valid]
        return values.reshape(*shape, 3), ownership.reshape(shape), known.reshape(shape)

    @lru_cache(maxsize=6)
    def features(self, key):
        part = self.table[key]
        arc = lines.arc_samples(part, 3.)
        points, normals, pattern_rows = [], [], []
        start = 0
        for row in arc:
            n = len(row['points'])
            points.extend(row['points']); normals.extend(np.c_[-row['tangent'][:, 1], row['tangent'][:, 0]])
            pattern_rows.append({'indices': np.arange(start, start + n), 'corner': row['corner'],
                                 'step': row['length'] / n})
            start += n
        if not points:
            return None
        points, normals = np.asarray(points), np.asarray(normals)
        corner = np.concatenate([row['corner'] for row in arc])
        eligible = np.flatnonzero(~corner)
        if len(eligible) < 4:
            eligible = np.arange(len(points))
        ids = eligible[np.linspace(0, len(eligible) - 1, min(384, len(eligible))).astype(int)]
        offsets = np.arange(-16., 16.001, 1 / self.scale)
        xy = points[ids, None] + normals[ids, None] * offsets[None, :, None]
        rgb, mask, known = self.read_pixels(key, xy, True)
        pattern_offsets = np.r_[np.arange(-16.,-11.999,.5),np.arange(-8.,8.001,.5),np.arange(12.,16.001,.5)]
        prgb, pmask, pknown = self.read_pixels(key, points[:, None] + normals[:, None] * pattern_offsets[None, :, None],True)
        return {'rgb': rgb, 'mask': mask, 'mask_known': known.any(axis=1), 'offsets': offsets,
                'profile_coords': xy, 'profile_points':points[ids], 'pattern_points': points,
                'pattern_mask':pmask,'pattern_mask_known':pknown,
                'pattern_rgb': prgb, 'pattern_offsets': pattern_offsets, 'pattern_rows': pattern_rows}

    def compact_fill(self, key, style, excluded=None):
        part = self.table[key]
        if not closed(part):
            return None
        # This shortcut can remove a same-pigment outline used to draw a dot.
        # A contrasting border is a semantic decision, not a width estimate.
        # In particular, an undersized contour samples the fill inside a real
        # border; that absence must not override a source-reviewed stroke.
        fill, stroke = style.get('fill'), style.get('stroke')
        if stroke and (not isinstance(fill, str) or
                       np.linalg.norm(lines.rgb(stroke) - lines.rgb(fill)) > 25):
            return None
        b = bounds(part); width, height = b[2] - b[0], b[3] - b[1]
        if min(width, height) < .5 or max(width, height) > 24. or len(list(subpaths(part))) != 1:
            return None
        colour = style.get('fill') if isinstance(style.get('fill'), str) else style.get('stroke')
        if not colour:
            return None
        scale = 10.5; origin = np.floor(np.array(b[:2]) - 3)
        size = np.ceil((np.array(b[2:]) + 3 - origin) * scale).astype(int)
        yy, xx = np.indices((size[1], size[0])); coords = np.stack([xx, yy], -1) / scale + origin
        rgb, sam, known = self.read_pixels(key, coords, True)
        polygon = Image.new('L', tuple(size), 0)
        pts = sample_commands(part['commands'], 96)
        ImageDraw.Draw(polygon).polygon([tuple(p) for p in (pts - origin) * scale], fill=255)
        body = np.asarray(polygon) > 0
        depth = distance_transform_edt(body) / scale
        inner = depth >= max(.25, .22 * min(width, height))
        if excluded is not None:
            inner &= ~excluded(coords.reshape(-1,2)).reshape(inner.shape)
        if inner.sum() < 4:
            return None
        outside = (~body) & (distance_transform_edt(~body) / scale > 1.2)
        bg = np.median(rgb[outside], axis=0) if outside.any() else np.full(3, 255.)
        fg = lines.rgb(colour); delta = fg - bg
        den = max(float(delta @ delta), 1.)
        alpha = np.clip(np.sum((rgb - bg) * delta, -1) / den, 0, 1)
        residual = np.linalg.norm(rgb - (bg + alpha[..., None] * delta), axis=-1)
        coverage = float(((alpha > .7) & (residual < 25))[inner].mean())
        edge = body & (depth < .75)
        distinct_ring = float(((residual > 35) & (np.linalg.norm(rgb - bg, axis=-1) > 40))[edge].mean()) if edge.any() else 0.
        if coverage < .9 or distinct_ring > .15:
            return None
        return {'fill': colour, 'stroke': None, 'width_px': 0., 'dash': 'solid'}, {
            'measured': True, 'role': 'filled_compact', 'present': False, 'width_px': 0.,
            'interior_colour_coverage': round(coverage, 4), 'distinct_ring_fraction': round(distinct_ring, 4),
            'sam_foreground_fraction': round(float(sam[body].mean()), 4) if known[body].any() else None,
            'geometry_bbox_px': [round(width, 3), round(height, 3)], 'separate_outline': False}

    def supported_stroke_colour(self, keys, style, *, excluded=None, paint=None):
        """Reject an absent border colour only when a repeated CLEAN core is clear.

        Shared groups pool evidence and receive one colour; numerical widths
        remain independent. Supported light colours are preserved even when a
        darker shadow is nearby.
        """
        colour=style.get('stroke');fill=style.get('fill')
        if not colour or (fill is not None and not isinstance(fill,str)) or (
                isinstance(fill,str) and np.linalg.norm(lines.rgb(colour)-lines.rgb(fill))<35):
            return style,None
        closest=[];cores=[];core_positions=[];core_backgrounds=[];contrasts=[];total=0
        for key in keys:
            feat=self.features(key)
            if feat is None:continue
            rgb=feat['rgb'];xy=feat['profile_coords'];offsets=feat['offsets'];central=abs(offsets)<=5
            if excluded is not None:
                visible=~np.asarray(excluded(feat['profile_points']),bool)
                rgb=rgb[visible];xy=xy[visible]
                if not len(rgb):continue
            local=rgb[:,central];sides=side_backgrounds(rgb,offsets,lines.rgb(colour));side0=sides[:,0];side1=sides[:,1]
            closest.extend(np.linalg.norm(local-lines.rgb(colour),axis=-1).min(axis=1).tolist())
            contrast=np.minimum(np.linalg.norm(local-side0[:,None],axis=-1),np.linalg.norm(local-side1[:,None],axis=-1))
            indexes=np.argmax(contrast-2*abs(offsets[central])[None],axis=1)
            peak=contrast[np.arange(len(local)),indexes];good=peak>40
            cores.extend(local[np.arange(len(local))[good],indexes[good]].tolist())
            core_positions.extend(xy[:,central][np.arange(len(local))[good],indexes[good]].tolist())
            core_backgrounds.extend(((side0+side1)/2)[good].tolist())
            contrasts.extend(peak[good].tolist());total+=len(local)
        if len(cores)<12:return style,None
        support=float(np.median(closest));core=np.median(cores,axis=0);spread=float(np.median(np.linalg.norm(np.asarray(cores)-core,axis=-1)))
        audit={'requested':colour,'median_source_distance':round(support,3),'core':lines.hexcolor(core),
               'core_colour_mad':round(spread,3),'core_profiles':len(cores),'all_profiles':total,'changed':False}
        background=np.median(core_backgrounds,axis=0);direction=lines.rgb(colour)-background
        alpha=float(np.dot(core-background,direction)/max(np.dot(direction,direction),1.))
        residual=float(np.linalg.norm(core-background-alpha*direction))
        if 0<=alpha<=1.15 and residual<25:
            # A light source core on the same colour ray may be antialiasing,
            # not a different pigment. Let width fitting model its alpha.
            return style,{**audit,'reason':'pigment_or_opacity_ambiguous'}
        # Transparent/open contours still need pigment validation. A wrong
        # saturated colour can fit only a weak fringe and collapse the width.
        # Broken lines need not paint a majority of the whole contour.
        coverage=.25 if fill is None else .55
        distinct_fill=fill is None or np.linalg.norm(core-lines.rgb(fill))>35
        if support>35 and len(cores)>=coverage*total and spread<30 and distinct_fill and np.linalg.norm(core-lines.rgb(colour))>25:
            # An open path has no fill of its own. A displaced grid/cross can
            # therefore sample its containing face and mistake that pigment
            # for a different stroke. The source/render model still owns the
            # semantic colour when the sampled core is an existing face fill.
            # This only vetoes a numerical recolour; it creates no geometry,
            # class ownership rule, or new model call.
            if paint:
                from shapely.geometry import Point
                positions=[Point(p) for p in core_positions]
                for face_key,face_style in paint.items():
                    face_fill=face_style.get('fill')
                    if (face_key in keys or face_key not in self.table or
                            not isinstance(face_fill,str) or
                            face_style.get('fill_opacity',1.)<.98 or
                            np.linalg.norm(core-lines.rgb(face_fill))>25 or
                            not closed(self.table[face_key])):
                        continue
                    face,error=footprint(self.table[face_key])
                    if error or face.is_empty or face.area<=0:
                        continue
                    covered=sum(face.covers(p) for p in positions)/len(positions)
                    if covered>=.55:
                        return style,{**audit,'reason':'observed_core_matches_existing_face_fill',
                                      'background_part':face_key,
                                      'core_inside_face_fraction':round(covered,4)}
            audit['changed']=True;return {**style,'stroke':lines.hexcolor(core)},audit
        return style,audit

    def fit(self, key, style, *, feedback=None, excluded=None, visibility_id=None, source_pattern=None):
        style = {k: v for k, v in style.items() if k != 'issue'}
        identity = {**self.identity, 'part': digest(self.table[key]),
                    'semantic_style': {k: style.get(k) for k in ('fill', 'fill_opacity', 'stroke', 'dash', 'dash_pattern_px')},
                    'review': feedback or None,
                    'source_pattern':source_pattern,
                    'width_revision_baseline':style.get('width_px') if any(r.get('expected_stroke_width') for r in (feedback or [])) else None,
                    'visibility': visibility_id}
        root = self.folder / digest(key)[:16] / digest(identity)[:16]
        path = root / 'result.json'
        if path.exists():
            saved = read(path)
            return saved['style'], saved['measurement']
        compact = self.compact_fill(key, style, excluded)
        if compact:
            out, measurement = compact
        elif not style.get('stroke'):
            out = {**style, 'stroke': None, 'width_px': 0., 'dash': 'solid'}
            out.pop('dash_pattern_px', None)
            measurement = {'measured': False, 'role': 'no_separate_stroke', 'present': False}
        else:
            feat = self.features(key)
            width_keep=np.ones(len(feat['rgb']),bool) if feat else None
            pattern_rows=feat['pattern_rows'] if feat else None
            if feat and excluded is not None:
                # Visibility belongs to the target contour, not to the whole
                # +/-16px measurement corridor. A neighbouring face inside
                # that corridor must not erase an otherwise visible edge.
                width_keep=~excluded(feat['profile_points'])
                hidden=excluded(feat['pattern_points'])
                pattern_rows=[{**r,'corner':r['corner'] | hidden[r['indices']]} for r in feat['pattern_rows']]
            directions=[r['expected_stroke_width'] for r in (feedback or []) if r.get('verdict')=='revise' and
                'stroke_width' in r.get('affected_fields',[]) and r.get('expected_stroke_width') in ('increase','decrease')]
            direction=directions[0] if directions and len(set(directions))==1 else None
            measurement = fit_profiles(feat['rgb'][width_keep], feat['offsets'], feat['mask'][width_keep], feat['mask_known'][width_keep],
                                       lines.rgb(style['stroke']), scale=self.scale, review=bool(feedback),width_preference=direction) if feat else {'measured': False}
            if direction:measurement['source_observed_width_direction']=direction
            # A fit can converge on neighbouring ink even when too few target
            # profiles survive. Convergence alone must not replace the width.
            usable=bool(measurement.get('measured') and measurement.get('width_reliable'))
            out = {**style, 'width_px': (measurement.get('width_px') if usable else None) or style.get('width_px') or 1.}
            measurement['width_measurement_applied']=usable
            if not usable:measurement['width_fallback_reason']='insufficient_reliable_target_profiles'
            previous_width=style.get('width_px')
            if direction and previous_width:
                wrong=(out['width_px']<previous_width-.01 if direction=='increase' else out['width_px']>previous_width+.01)
                if wrong:
                    measurement.update(width_revision_applied=False,width_revision_reason='no_supported_mode_in_requested_direction',
                        retained_width_px=previous_width)
                    out['width_px']=previous_width
            if feat and measurement.get('measured'):
                pattern = classify_pattern(feat['pattern_rgb'], feat['pattern_offsets'], lines.rgb(measurement['source_colour']), pattern_rows,
                                           measurement.get('signed_geometry_offset_px',0.),out['width_px'],
                                           feat['pattern_mask'],feat['pattern_mask_known'])
                if 'on_lengths' in pattern:
                    sam_pattern=pattern.pop('sam_pattern_candidate',None)
                    classified = lines.classify(pattern.pop('on_lengths'), pattern.pop('gap_lengths'), pattern['coverage'],
                                                pattern['coverage'], out['width_px'])
                    if not classified.get('dash_reliable') and sam_pattern:
                        classified={**sam_pattern,'pattern_origin':'sam_template_matched_to_source_rgb'}
                    measurement.update(pattern, **classified)
                else:
                    measurement.update(pattern)
                # Periodic pixels are not enough: a displaced contour crossing
                # text or a weak wrong-colour fringe can also look like dots.
                # Require a persistent localized stroke, with independent SAM
                # support or substantial colour-matched cross-sections.
                count=int(width_keep.sum())
                minimum=max(12,.05*count if closed(self.table[key]) else 0)
                localized=bool(measurement.get('mask_width_samples',0)>0 or
                    measurement.get('width_reliable') and measurement.get('rgb_width_samples',0)>=minimum)
                measurement.update(width_profile_count=count,pattern_stroke_localized=localized)
                if measurement.get('dash_reliable') and measurement.get('dash')!='solid' and not localized:
                    measurement.update(dash='unknown',dash_reliable=False,dash_reason='insufficient_localized_stroke')
                if measurement.get('dash_reliable'):
                    out['dash'] = measurement['dash']
                    out.pop('dash_pattern_px',None)
                    if measurement.get('dash_pattern_px'):
                        out['dash_pattern_px'] = measurement['dash_pattern_px']
                else:
                    corrections=[row['expected_line_pattern'] for row in (feedback or [])
                                 if row.get('verdict')=='revise' and 'line_pattern' in row.get('affected_fields',[])
                                 and row.get('expected_line_pattern') in ('solid','dash','dot','dash_dot')]
                    if corrections and len(set(corrections))==1:
                        out['dash']=corrections[0]
                        out.pop('dash_pattern_px',None)
                        measurement['pattern_origin']='critic_category_with_unknown_numeric_pattern'
                if measurement.get('measurement_colour_calibrated'):
                    out['stroke']=measurement['source_colour']
            measurement['role'] = 'stroked_contour'
        source_fit=self.table[key].get('diagnostics_local',{}).get('source_primitive_registration',{})
        if (source_fit.get('perimeter_support',0)>=.8 and source_fit.get('source_width_px') is not None
                and source_fit.get('source_stroke') is not None and style.get('stroke')
                and np.linalg.norm(lines.rgb(style['stroke'])-source_fit['source_stroke'])<25):
            out['width_px']=source_fit['source_width_px']
            measurement.update(width_px=out['width_px'],width_reliable=True,source_perimeter_width_applied=True)
        # A fully observed continuous stroke is not a dotted line merely
        # because enlarged antialiasing looks periodic. Numeric categories with
        # gaps, foreign ink or SAM/RGB conflicts remain open to visual review.
        visual=[r['expected_line_pattern'] for r in (feedback or [])
                if r.get('verdict')=='revise'
                and (r.get('sam_visual_evidence') or (r.get('source_evidence') and r.get('rendered_difference')))
                and 'line_pattern' in r.get('affected_fields',[])
                and r.get('expected_line_pattern') in ('solid','dash','dot','dash_dot')]
        continuous=(measurement.get('dash_reliable') and measurement.get('dash')=='solid'
            and measurement.get('coverage',0)>=1.-1e-6 and measurement.get('width_reliable')
            and measurement.get('pattern_stroke_localized')
            and measurement.get('pattern_foreign_samples')==0
            and measurement.get('pattern_sam_rgb_conflicts')==0)
        if out.get('stroke') and visual and len(set(visual))==1 and not continuous:
            category=visual[0]
            if out.get('dash')!=category or category=='solid':out.pop('dash_pattern_px',None)
            out['dash']=category
            measurement.update(pattern_origin=('glm_sam_rgb_visual_observation' if any(r.get('sam_visual_evidence') for r in (feedback or [])) else 'critic_source_render_observation'),
                               numeric_pattern=measurement.get('dash','unknown'),visual_pattern=category)
        elif continuous and visual and any(v!='solid' for v in visual):
            measurement.update(pattern_origin='continuous_source_without_foreign_ink',
                               numeric_pattern='solid',visual_pattern=visual[0])
        explicit_pattern=any(r.get('verdict')=='revise' and 'line_pattern' in r.get('affected_fields',[])
            and r.get('expected_line_pattern') in ('solid','dash','dot','dash_dot') for r in (feedback or []))
        conflicted=(measurement.get('pattern_sam_rgb_conflicts',0)>0 or
                    measurement.get('pattern_foreign_samples',0)>0)
        if (source_pattern in ('solid','dash','dot','dash_dot') and out.get('stroke')
                and out.get('dash')!=source_pattern and conflicted and not explicit_pattern):
            # Crossing ink or a displaced path can imitate periodic gaps.
            # An ambiguous numeric category must not silently replace the
            # existing source-only style selection. Keep measuring width;
            # the ordinary rendered critic can still revise the category.
            measurement.update(pattern_origin='source_style_with_conflicting_numeric_pattern',
                numeric_pattern=out.get('dash'),source_pattern=source_pattern,dash_reliable=False)
            out['dash']=source_pattern
            out.pop('dash_pattern_px',None)
        measurement.update(protocol=PROTOCOL, geometry_unchanged=True)
        save(path, {'identity': identity, 'style': out, 'measurement': measurement})
        return out, measurement
