from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment/shape_sam_guided'))
from models import Models


class ExternalSamIdentity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.image=self.root/'input.png';self.image.write_bytes(b'test-image')
        self.config={'glm_concurrency':1,'sam_urls':['http://sam.invalid:18103'],
            'sam_threshold':.3,'external_sam_epoch':7,'external_sam_checkpoint_sha256':'0'*64,
            'project_root':str(self.root)}
        self.result={'checkpoint':{'epoch':7,'sha256':'0'*64},'predictions':[]}

    def test_new_service_envelope_is_verified_and_normalized(self):
        model=Models(self.config)
        with patch.object(model,'post',return_value=self.result) as request:
            out=model.sam(self.image,self.root/'output',[{'name':'polygon'}])
            again=model.sam(self.image,self.root/'output',[{'name':'polygon'}])
        self.assertEqual(out['best_checkpoint']['epoch'],7)
        self.assertEqual(out,again);self.assertEqual(request.call_count,1)

    def test_wrong_epoch_or_hash_cannot_be_committed(self):
        for changed in ({'epoch':18,'sha256':'0'*64},{'epoch':7,'sha256':'1'*64}):
            with self.subTest(changed=changed):
                model=Models(self.config);folder=self.root/str(changed['epoch'])/changed['sha256'][:1]
                with patch.object(model,'post',return_value={**self.result,'checkpoint':changed}):
                    with self.assertRaises(ValueError):model.sam(self.image,folder,[{'name':'polygon'}])
                self.assertFalse((folder/'predictions.json').exists())

    def test_checkpoint_change_invalidates_cache_before_reusing_masks(self):
        model=Models(self.config);folder=self.root/'output'
        with patch.object(model,'post',return_value=self.result):model.sam(self.image,folder,[{'name':'polygon'}])
        changed=Models({**self.config,'external_sam_epoch':18})
        with patch.object(changed,'post') as request:
            with self.assertRaisesRegex(ValueError,'cache input'):changed.sam(self.image,folder,[{'name':'polygon'}])
        request.assert_not_called()

    def test_cross_run_cache_uses_exact_request_and_preserves_source(self):
        old=self.root/'old';new=self.root/'new';relative=Path('sam_styles/case/tiles/tile/sam')
        model=Models(self.config)
        mask=old/'mask.png';mask.parent.mkdir();mask.write_bytes(b'mask')
        result={**self.result,'predictions':[{'mask_path':str(mask)}]}
        with patch.object(model,'post',return_value=result):
            expected=model.sam(self.image,old/relative,[{'name':'line'}])
        receipt=(old/relative/'predictions.json').read_bytes()
        current=Models({**self.config,'sam_output_root':str(new),'sam_reuse_roots':[str(old)]})
        with patch.object(current,'post',side_effect=AssertionError('identical request must reuse')):
            self.assertEqual(current.sam(self.image,new/relative,[{'name':'line'}]),expected)
        self.assertEqual((old/relative/'predictions.json').read_bytes(),receipt)
        self.assertTrue((new/relative/'cache_reuse.json').is_file())
        changed=self.root/'changed'
        current=Models({**self.config,'sam_output_root':str(changed),'sam_reuse_roots':[str(old)]})
        with patch.object(current,'post',return_value=self.result) as request:
            current.sam(self.image,changed/relative,[{'name':'ellipse'}])
        self.assertEqual(request.call_count,1)


if __name__=='__main__':unittest.main()
