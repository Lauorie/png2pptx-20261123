"""Source contrast evidence independent of a model's predicted text color."""
import numpy as np

MIN_CONTRAST = 16


def contrast_ink(rgb, core_box):
    x0,y0,x1,y1=core_box
    core=rgb[max(0,y0):min(rgb.shape[0],y1),max(0,x0):min(rgb.shape[1],x1)]
    if not core.size:core=rgb
    pixels=core.reshape(-1,3).astype(np.float32)
    quantized=(pixels//8).astype(np.int16)
    keys,counts=np.unique(quantized,axis=0,return_counts=True)
    dominant=keys[counts.argmax()]
    background=np.median(pixels[np.all(quantized==dominant,axis=1)],axis=0)
    contrast=np.linalg.norm(rgb-background,axis=2)
    # Estimate strength from foreground only. Taking a percentile over the
    # entire crop silently erases real text when whitespace exceeds 97%.
    core_contrast=np.linalg.norm(pixels-background,axis=1)
    foreground=core_contrast[core_contrast>=MIN_CONTRAST]
    if not foreground.size:return np.zeros(rgb.shape[:2],dtype=np.float32)
    amplitude=float(np.percentile(foreground,97))
    ink=np.clip(contrast/amplitude,0,1).astype(np.float32)
    ink[contrast<MIN_CONTRAST]=0
    ink[ink<.20]=0
    return ink
