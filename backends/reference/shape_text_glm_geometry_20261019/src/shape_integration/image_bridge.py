"""Use the symlinked image pipeline for detection only, sharing the API limiter."""
from argparse import Namespace
from png2pptx.images import pipeline
from .common import read
from .image_boundary import BoundaryScheduler

def configure(runtime):
    # Configure once before workers start. No source files or credentials are written.
    def factory(root,model='glm-5.3-flash',max_tokens=100000):
        if model!=runtime.config['model']:raise ValueError('Unexpected image model')
        return runtime.client(root,max_tokens)
    pipeline.make_detector=factory
    # One pool shared by every figure; clients also share Runtime.slots with all
    # shape/fitting calls. The linked production image module stays unchanged.
    pipeline.review_boundaries=BoundaryScheduler(runtime.config['api_concurrency'])

def detect(source,root,config):
    options=config['image_detection']
    args=Namespace(image=source,output=root,scene=None,scene_root=None,candidates=None,legacy_assets=None,
                   model=config['model'],max_tokens=options['max_tokens'],prompt_max_tokens=options.get('prompt_max_tokens',100000),
                   coverage_rounds=options['coverage_rounds'],max_local_calls=options['max_local_calls'],
                   boundary_review_rounds=options['boundary_review_rounds'],max_boundary_calls=options['max_boundary_calls'],
                   boundary_max_tokens=options['boundary_max_tokens'],context_padding=.15,only=None,
                   detect_only=True,resume=(root/'config.json').exists(),
                   edit_endpoint=pipeline.EDIT_ENDPOINT,edit_model=pipeline.EDIT_MODEL,steps=30,seed=42,image_guidance=3.)
    pipeline.run(args)
    return read(root/'inventory.json')
