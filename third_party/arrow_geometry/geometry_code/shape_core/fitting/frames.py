"""Explicit pixel-center transport, independent of VLM and historical paths."""
from dataclasses import dataclass
import math

from .contracts import input_dimensions


@dataclass(frozen=True)
class FitFrame:
    input_size: tuple
    global_size: tuple
    original_size: tuple
    crop_bounds: tuple

    def __post_init__(self):
        for size in (self.input_size, self.global_size, self.original_size):
            input_dimensions(size)
        b = self.crop_bounds
        if (len(b) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in b)
                or not 0 <= b[0] < b[2] <= self.global_size[0]
                or not 0 <= b[1] < b[3] <= self.global_size[1]):
            raise ValueError('Invalid half-open crop bounds')

    @classmethod
    def from_metadata(cls, meta):
        frame = cls(tuple(meta['actual_crop_input_size_px']), tuple(meta['global_image_size_px']),
                    tuple(meta['original_image_size_px']), tuple(meta['crop_bounds_global_half_open']))
        if 'source_crop_half_open' in meta:
            expected = [frame.crop_bounds[i]*frame.original_size[i % 2]/frame.global_size[i % 2]
                        for i in range(4)]
            source = meta['source_crop_half_open']
            if len(source) != 4 or any(not math.isfinite(v) or abs(v-e) > 1e-6
                                      for v, e in zip(source, expected)):
                raise ValueError('Native-source crop and global crop disagree')
        return frame

    def metadata(self):
        return {'actual_crop_input_size_px': list(self.input_size),
                'global_image_size_px': list(self.global_size),
                'original_image_size_px': list(self.original_size),
                'crop_bounds_global_half_open': list(self.crop_bounds)}

    def to_global(self, xy):
        return [self.crop_bounds[i]+(v+.5)*(self.crop_bounds[i+2]-self.crop_bounds[i])
                / self.input_size[i]-.5 for i, v in enumerate(xy)]

    def to_original(self, xy):
        return [(v+.5)*self.original_size[i]/self.global_size[i]-.5
                for i, v in enumerate(self.to_global(xy))]

    def to_local(self, xy):
        global_xy = [(v+.5)*self.global_size[i]/self.original_size[i]-.5
                     for i, v in enumerate(xy)]
        return [(v-self.crop_bounds[i]+.5)*self.input_size[i]
                / (self.crop_bounds[i+2]-self.crop_bounds[i])-.5
                for i, v in enumerate(global_xy)]
