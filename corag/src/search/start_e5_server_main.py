# Most of the code comes from https://huggingface.co/docs/transformers/en/pipeline_webserver
import os
import asyncio
import traceback
import sys
import random
sys.path.insert(0, 'src')

from typing import List, Dict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from search.e5_searcher import E5Searcher
from logger_config import logger


async def search(request: Request):
    payload = await request.json()
    query: str = payload['query']
    response_q = asyncio.Queue()
    await request.app.model_queue.put((query, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    searcher: E5Searcher = E5Searcher(
        index_dir=os.getenv('INDEX_DIR', 'data/e5-large-index/'),
        model_name_or_path=os.getenv('E5_MODEL_NAME_OR_PATH', 'intfloat/e5-large-v2'),
    )
    top_k = int(os.getenv('TOP_K', 5))
    logger.info(f'E5Searcher initialized, ready to serve requests, top_k={top_k}')
    # warmup the server a bit
    searcher.batch_search(queries=[f'test query {random.random()} {i}' for i in range(64)], k=top_k)

    while True:
        queries: List[str] = []
        queues: List[asyncio.Queue] = []
        while len(queries) < 64:
            try:
                (query, rq) = await asyncio.wait_for(q.get(), timeout=0.001)
            except asyncio.exceptions.TimeoutError:
                if len(queries) > 0:
                    break
                else:
                    continue
            except Exception as e:
                logger.error(f"Error while getting query: {e}")
                logger.error(traceback.format_exc())
                continue
            queries.append(query)
            queues.append(rq)

        try:
            results_list: List[List[Dict]] = searcher.batch_search(queries=queries, k=top_k)
            for rq, results in zip(queues, results_list):
                await rq.put(results)
        except Exception as e:
            logger.error(f"Error during batch search: {e}")
            logger.error(traceback.format_exc())


app = Starlette(
    routes=[
        Route("/", search, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
