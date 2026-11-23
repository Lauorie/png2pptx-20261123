"""Pixel-center transforms; only the model chooses a semantic crop region."""
from dataclasses import dataclass
import math
from PIL import Image


def resized_size(size, long_edge=2048):
    if min(size)<2:raise ValueError('Need at least two pixels on each axis')
    return tuple(max(2,round(v*long_edge/max(size))) for v in size)


def scale_point(xy, source_size, target_size):
    return [(float(v)+0.5)*target_size[i]/source_size[i]-0.5 for i,v in enumerate(xy)]


def scale_box(box, source_size, target_size):
    return scale_point(box[:2],source_size,target_size)+scale_point(box[2:],source_size,target_size)


@dataclass(frozen=True)
class Crop:
    global_size: tuple
    box: tuple  # PIL half-open bounds in global 2048 pixels
    input_size: tuple
    requested_scale: float = None
    max_long_edge: int = None

    @classmethod
    def from_model(cls, bbox, global_size):
        if len(bbox)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in bbox):
            raise ValueError('Crop bbox needs four finite JSON numbers')
        if not (0<=bbox[0]<bbox[2]<=global_size[0]-1 and 0<=bbox[1]<bbox[3]<=global_size[1]-1):
            raise ValueError('Model crop outside global pixel-center bounds')
        x0,y0=math.floor(bbox[0]),math.floor(bbox[1]);x1,y1=math.ceil(bbox[2])+1,math.ceil(bbox[3])+1
        box=(x0,y0,x1,y1);return cls(tuple(global_size),box,resized_size((x1-x0,y1-y0)))

    @property
    def size(self):return (self.box[2]-self.box[0],self.box[3]-self.box[1])

    def scaled(self, factor, max_long_edge=None):
        """Resize a crop by a multiplier relative to the global-image crop pixels."""
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<1:
            raise ValueError('Crop upsampling factor must be a finite number >= 1')
        if max_long_edge is not None and (isinstance(max_long_edge,bool) or not isinstance(max_long_edge,int) or max_long_edge<2):
            raise ValueError('Crop maximum long edge must be an integer >= 2')
        effective=min(factor,max_long_edge/max(self.size)) if max_long_edge is not None else factor
        return Crop(self.global_size,self.box,tuple(max(2,round(v*effective)) for v in self.size),factor,max_long_edge)

    def to_global(self,xy):
        return [self.box[i]+(v+0.5)*self.size[i]/self.input_size[i]-0.5 for i,v in enumerate(xy)]

    def to_local(self,xy):
        return [(v-self.box[i]+0.5)*self.input_size[i]/self.size[i]-0.5 for i,v in enumerate(xy)]

    def metadata(self, original_size):
        return {'actual_crop_input_size_px':list(self.input_size),'global_image_size_px':list(self.global_size),
                'original_image_size_px':list(original_size),'crop_bounds_global_half_open':list(self.box),
                'crop_size_before_resize_px':list(self.size),
                'effective_crop_scale_xy':[self.input_size[i]/self.size[i] for i in (0,1)],
                'requested_crop_scale':self.requested_scale,'crop_max_long_edge':self.max_long_edge,
                'local_to_global':{'scale_xy':[self.size[i]/self.input_size[i] for i in (0,1)],'offset_xy':[self.box[i]+0.5*self.size[i]/self.input_size[i]-0.5 for i in (0,1)]},
                'global_to_original':{'scale_xy':[original_size[i]/self.global_size[i] for i in (0,1)],'offset_xy':[0.5*original_size[i]/self.global_size[i]-0.5 for i in (0,1)]},
                'output_units':'absolute pixel-center coordinates in ACTUAL CROP INPUT ONLY',
                'output_min_xy':[0,0],'output_max_xy':[self.input_size[0]-1,self.input_size[1]-1],
                'convention':'Pillow half-pixel sampling: dst=(src+0.5)*dst_size/src_size-0.5. Controller maps local to global as local*scale+offset, then global to original. Metadata is context only; do not add offsets to local output. Pixel footprints reach half a pixel outside the center range.'}

    def image(self, global_image):
        if tuple(global_image.size)!=self.global_size:raise ValueError('Wrong global input size')
        return global_image.crop(self.box).resize(self.input_size,Image.Resampling.LANCZOS)
