"""External epoch-checked SAM protocol and interrupted-output ownership."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import multiprocessing
import os
import time
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiment/shape_sam_guided'))
from models import Models


class ExternalSam(unittest.TestCase):
    def test_separate_client_processes_do_not_overrun_one_nonqueued_service(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';Image.new('RGB',(12,12),'white').save(image)
            (root/'dataset_manifest.json').write_text(json.dumps({'categories':[{'name':'rectangle'}]}))
            context=multiprocessing.get_context('fork');ready=context.Barrier(2)
            def client(index):
                model=Models({'glm_concurrency':1,'sam_urls':['http://sam.internal:18103'],
                    'sam_ports':[],'dataset_root':str(root),'project_root':str(root),
                    'sam_threshold':.3,'external_sam_epoch':18})
                def post(url,body,**kwargs):
                    # The deployed service rejects a concurrent admitted writer.
                    marker=root/'server_busy';fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
                    try:
                        time.sleep(.15);dest=Path(body['output']);dest.mkdir()
                        result={'predictions':[],'best_checkpoint':{'epoch':18}}
                        (dest/'predictions.json').write_text(json.dumps(result));return result
                    finally:os.close(fd);marker.unlink()
                model.post=post;ready.wait(timeout=5);model.sam(image,root/f'out{index}')
            workers=[context.Process(target=client,args=(i,)) for i in range(2)]
            for worker in workers:worker.start()
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():worker.terminate();worker.join();self.fail('Client admission stalled')
                self.assertEqual(worker.exitcode,0)
            self.assertTrue(all((root/f'out{i}/predictions.json').exists() for i in range(2)))

    def test_external_output_does_not_collide_with_client_request_and_resumes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';Image.new('RGB',(12,12),'white').save(image)
            (root/'dataset_manifest.json').write_text(json.dumps({'categories':[{'name':'line'},{'name':'rectangle'},{'name':'block_arrow'}]}))
            model=Models({'glm_concurrency':1,'sam_urls':['http://sam.internal:18103'],'sam_ports':[],
                          'dataset_root':str(root),'sam_threshold':.3,'external_sam_epoch':18})
            requests=[]
            def post(url,body,**kwargs):
                requests.append(body);dest=Path(body['output'])
                self.assertFalse(dest.exists());self.assertTrue((dest.parent/'request.json').exists())
                self.assertEqual(body['categories'],['rectangle'])
                dest.mkdir();result={'predictions':[],'best_checkpoint':{'epoch':18}}
                (dest/'predictions.json').write_text(json.dumps(result));return result
            model.post=post
            self.assertEqual(model.sam(image,root/'out')['best_checkpoint']['epoch'],18)
            self.assertEqual(model.sam(image,root/'out')['best_checkpoint']['epoch'],18)
            self.assertEqual(len(requests),1)
            # The server may have completed after the client lost its connection.
            (root/'out/predictions.json').unlink()
            model.sam(image,root/'out');self.assertEqual(len(requests),1)

    def test_incomplete_remote_output_is_not_resubmitted(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';Image.new('RGB',(12,12)).save(image)
            model=Models({'glm_concurrency':1,'sam_urls':['http://sam.internal:18103'],'sam_ports':[],
                          'dataset_root':str(root),'sam_threshold':.3,'external_sam_epoch':18})
            (root/'out/service_output').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'second writer'):model.sam(image,root/'out')


if __name__=='__main__':unittest.main()
