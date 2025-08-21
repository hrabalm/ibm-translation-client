import asyncio
import sys
from dataclasses import dataclass

import httpx
import tenacity

limits = httpx.Limits(
    max_keepalive_connections=100,
    max_connections=200,
)
timeout = httpx.Timeout(
    5,
    read=600,
    write=60,
)


@dataclass
class TranslationJob:
    model: str
    content: str
    extension: str
    src_lang: str
    tgt_lang: str
    glossary_id: str
    do_not_translate_id: str


@dataclass
class TranslationJobResult:
    content: str


class TranslationClient:
    def __init__(self, base_url: str, token: str, max_concurrent: int):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            limits=limits,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_file(self, job: TranslationJob) -> TranslationJobResult | None:
        temp_filename = f"temp.{job.extension}"
        request = {
            "model": job.model,
            "src_lang": job.src_lang,
            "tgt_lang": job.tgt_lang,
            "files": {temp_filename: job.content},
            "glossary_id": job.glossary_id,
            "do_not_translate_id": job.do_not_translate_id,
        }

        async with self.semaphore:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(3),
                wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            )
            async def _send_request(request):
                response = await self.client.post("/translate-document", json=request)
                response.raise_for_status()
                return response

            try:
                response = await _send_request(request)
                content = response.json()["files"][temp_filename]
                return TranslationJobResult(content=content)
            except Exception as e:
                print(f"Error occurred: {e}", file=sys.stderr, flush=True)
                return None
