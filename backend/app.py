import asyncio
from collections import defaultdict,deque
import logging
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.schemas import PromptRequest,ScenePlan,GenerationRequest,GenerationResult
from backend.settings import Settings,ROOT
from backend import services

load_dotenv(ROOT/'.env')
logger=logging.getLogger('pixel_api')


def create_app(settings=None):
    settings=settings or Settings.from_env()
    app=FastAPI(title='Pixel Landscape API',version='1.0.0')
    arrivals=defaultdict(deque);global_arrivals=deque();generation_lock=asyncio.Lock()

    @app.middleware('http')
    async def observe(request,call_next):
        began=time.perf_counter();request_id=uuid.uuid4().hex
        if request.method=='POST' and '/v1/' in request.url.path:
            now=time.monotonic()
            # Do not trust client-supplied X-Forwarded-For headers here.
            key=request.client.host if request.client else 'unknown'
            for expired in [k for k,q in arrivals.items() if not q or now-q[-1]>=60]:del arrivals[expired]
            queue=arrivals[key]
            for q in (queue,global_arrivals):
                while q and now-q[0]>=60:q.popleft()
            if len(queue)>=8 or len(global_arrivals)>=32:
                return JSONResponse({'error':{'code':'rate_limit','message':'Too many requests. Please wait a minute.'}},429,headers={'Retry-After':'60'})
            queue.append(now);global_arrivals.append(now)
        response=await call_next(request)
        response.headers['X-Request-ID']=request_id
        logger.info('request id=%s method=%s status=%s seconds=%.3f',request_id,request.method,response.status_code,time.perf_counter()-began)
        return response

    @app.exception_handler(services.ServiceError)
    async def service_error(request,error):
        return JSONResponse({'error':{'code':error.code,'message':error.message}},status_code=error.status)

    @app.get('/health')
    def health():
        version,_=services.active_release(settings)
        return dict(status='ok',model_version=version,text_configured=bool(settings.groq_api_key))

    @app.get('/v1/model')
    def model():
        version,release=services.active_release(settings)
        return dict(version=version,parameters=release['parameters'],epoch=release['epoch'],
                    resolution=[64,64],checkpoint_sha256=release['checkpoint_sha256'])

    @app.post('/v1/interpret',response_model=ScenePlan)
    async def interpret(request:PromptRequest):
        return await services.interpret(request.prompt,settings)

    @app.post('/v1/generations',response_model=GenerationResult)
    async def generate(request:GenerationRequest):
        if generation_lock.locked():
            raise services.ServiceError('generation_busy','Another image is being generated. Please retry shortly.',429)
        async with generation_lock:
            return await run_in_threadpool(services.generate,request,settings)

    # Wrap error/rate-limit responses too, so browsers can read them.
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.cors_origins),
                       allow_methods=['GET','POST'],allow_headers=['Content-Type'])
    return app


app=create_app()
