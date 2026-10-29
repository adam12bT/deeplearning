import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.schemas import ScenePlan,GenerationRequest,GenerationResult
from backend.services import ServiceError,generate,interpret
from backend.settings import Settings
from mlops.release import verify_release

SCENE=dict(sky='night',landform='mountains',ground='rock',water='lake',objects='none')
PLAN=dict(scene=SCENE,unsupported=[],explanation='A rocky mountain lake at night.')


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        registry=self.root/'registry.json'
        registry.write_text(json.dumps(dict(active_version='test-v1',releases={'test-v1':dict(status='approved',checkpoint_sha256='a'*64,epoch=20,parameters=12)})))
        self.settings=Settings('','llama-3.3-70b-versatile',None,'owner/space',('http://localhost:5173',),registry)
        self.client=TestClient(create_app(self.settings));self.addCleanup(self.client.close)

    def test_health_exposes_version_but_no_credentials(self):
        response=self.client.get('/health')
        self.assertEqual(response.json(),dict(status='ok',model_version='test-v1',text_configured=False))

    def test_missing_groq_key_is_actionable(self):
        response=self.client.post('/v1/interpret',json={'prompt':'a moonlit lake'})
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.json()['error']['code'],'llm_not_configured')

    def test_text_plan_retains_unsupported_requests(self):
        unsupported=dict(PLAN,unsupported=['dragon'])
        with patch('backend.services.interpret',new=AsyncMock(return_value=ScenePlan(**unsupported))):
            response=self.client.post('/v1/interpret',json={'prompt':'a dragon by the lake'})
        self.assertEqual(response.json()['unsupported'],['dragon'])

    def test_unknown_control_never_reaches_gpu(self):
        with patch('backend.services.generate') as generator:
            response=self.client.post('/v1/generations',json={'scene':dict(SCENE,objects='castle')})
        self.assertEqual(response.status_code,422);generator.assert_not_called()

    def test_quota_error_is_structured(self):
        with patch('backend.services.generate',side_effect=ServiceError('gpu_quota','Wait for quota reset.',429)):
            response=self.client.post('/v1/generations',json={'scene':SCENE})
        self.assertEqual(response.status_code,429)
        self.assertEqual(response.json()['error']['code'],'gpu_quota')

    def test_rate_limit_prevents_unbounded_llm_calls(self):
        with patch('backend.services.interpret',new=AsyncMock(return_value=ScenePlan(**PLAN))) as interpreter:
            responses=[self.client.post('/v1/interpret',json={'prompt':'a lake'}) for _ in range(9)]
        self.assertEqual(responses[-1].status_code,429)
        self.assertEqual(interpreter.await_count,8)

    def fake_remote(self,sha):
        image=self.root/'image.png';Image.new('RGB',(64,64)).save(image)
        class Job:
            def result(self,timeout):return [str(image),str(image),dict(checkpoint_sha256=sha,seed=42),42]
        class Client:
            def __init__(self,*args,**kwargs):pass
            def submit(self,*args,**kwargs):return Job()
        return Client

    def test_deployed_weight_mismatch_is_rejected(self):
        with patch('gradio_client.Client',self.fake_remote('b'*64)):
            with self.assertRaises(ServiceError) as caught:generate(GenerationRequest(scene=SCENE,seed=42),self.settings)
        self.assertEqual(caught.exception.code,'model_version_mismatch')

    def test_generation_returns_valid_native_png_and_version(self):
        with patch('gradio_client.Client',self.fake_remote('a'*64)):
            result=generate(GenerationRequest(scene=SCENE,seed=42),self.settings)
        self.assertEqual(result.model_version,'test-v1')
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(result.image_base64))).size,(64,64))

    def test_corrupt_release_is_rejected(self):
        (self.root/'release.json').write_text(json.dumps(dict(files={'model.pt':'a'*64,'tokenizer.pt':'b'*64})))
        (self.root/'model.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Corrupt artifact'):verify_release(self.root)


class GroqTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_llm_json_cannot_become_scene_controls(self):
        settings=Settings('test-secret','model',None,'owner/space',(),Path('unused'))
        response=httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(dict(PLAN,scene=dict(SCENE,sky='space')))}}]},request=httpx.Request('POST','https://api.groq.com'))
        mock=AsyncMock();mock.__aenter__.return_value=mock;mock.post.return_value=response
        with patch('backend.services.httpx.AsyncClient',return_value=mock):
            with self.assertRaises(ServiceError) as caught:await interpret('a landscape',settings)
        self.assertEqual(caught.exception.code,'llm_failed')
        self.assertNotIn('test-secret',caught.exception.message)

    async def test_valid_groq_json_uses_json_mode(self):
        settings=Settings('test-key','model',None,'owner/space',(),Path('unused'))
        response=httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(PLAN)}}]},request=httpx.Request('POST','https://api.groq.com'))
        mock=AsyncMock();mock.__aenter__.return_value=mock;mock.post.return_value=response
        with patch('backend.services.httpx.AsyncClient',return_value=mock):plan=await interpret('a moonlit lake',settings)
        self.assertEqual(plan.scene.sky,'night')
        self.assertEqual(mock.post.call_args.kwargs['json']['response_format'],{'type':'json_object'})


if __name__=='__main__':unittest.main()
