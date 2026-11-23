"""A source-confirmed coverage repair executes with no old fitted geometry."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from shape_integration.common import save
import test_pipeline as fixtures


class ZeroFitRepair(fixtures.unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    def test_coverage_can_repair_an_owner_without_old_fitted_parts(self):
        source=self.work/'localization';case=source/'cases/case'
        save(source/'manifest.json',{'images':[self.item]})
        save(case/'geometry.json',{'image_size':self.item['source_size'],
                                 'coordinate_system':'image_pixel_centers','objects':[self.candidate]})
        save(case/'summary.json',{'image_coverage_status':'complete','errors':{},'local_failures':0})
        save(case/'dedup/image_boxes_original.json',{'assets':[]})
        provenance=fixtures.source_manifest(source,[self.item]);self.options['coverage_rounds']=1
        observations=[]
        def responder(tag,prompt,count):
            if tag=='observe':
                observations.append(prompt)
                return fixtures.observed('G1','unsupported') if len(observations)==1 else fixtures.observed('G1')
            if tag=='check':return fixtures.reviewed(prompt)
            if tag=='coverage':return {'status':'uncertain','new_objects':[],'possible_duplicates':[],
                'uncertain':['Explicit source-visible missing frame'],
                'repair_existing':[{'object_id':'G1','issues':[{'kind':'missing','part_ids':[],
                    'evidence':'The source has a visible frame absent from the fit','action':'Recover that frame'}]}]}
            raise AssertionError(tag)
        runtime=fixtures.FakeRuntime(self.config,responder)
        with ThreadPoolExecutor(max_workers=2) as pool, patch('shape_integration.batch.case',side_effect=AssertionError):
            result=fixtures.run_case(self.work,self.work/'new',self.item,runtime,self.config,pool,source,provenance,False)
        assert len(observations)==2 and result['successful_parts']==2
        assert observations[-1]['coverage_requested_corrections']['requests']
        assert len(list((self.work/'new/cases/case/fitting/objects/G1').glob(
            'coverage_repair_01_*/revision_00/fit_original.json')))==1
