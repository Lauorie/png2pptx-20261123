import unittest

import test_pipeline as fixtures
from shape_integration.common import read,sha
from shape_integration.fitting.control import process_object
from shape_integration.fitting.seed import revision_seed


class ExistingCropScope(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object

    def test_coverage_scope_inside_existing_clean_crop_keeps_verified_geometry(self):
        self.run_object(fixtures.FakeRuntime(self.config))
        original=self.folder/'objects/G1';seed=revision_seed(original)
        crop=read(original/'revision_00/job.json')['frame']['source_crop_half_open']
        extent=[crop[0]+1,crop[1]+1,crop[2]-2,crop[3]-2]
        target=self.work/'same_scope'
        process_object(target,self.candidate,self.item,self.work,None,self.options,
            namespace='same_scope',prepare_only=True,seed=seed,
            input_context={'coverage_source_extent_original':extent})
        self.assertTrue((target/'revision_00/reused_revision.json').is_file())
        self.assertEqual(sha(original/'revision_00/fit_original.json'),sha(target/'revision_00/fit_original.json'))
        self.assertEqual(read(target/'revision_00/job.json')['frame']['source_crop_half_open'],crop)
        self.assertTrue(read(target/'source_scope_reuse.json')['geometry_reused'])
        self.assertFalse((target/'source_scope_seed.json').exists())
