import asyncio
import threading
from contextvars import ContextVar
from types import SimpleNamespace

from core.vlm import VLMClient


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"description-{self.calls}"))]
        )


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _client_with_budget(max_pages: int) -> VLMClient:
    client = object.__new__(VLMClient)
    client.enabled = True
    client.max_pages = max_pages
    client.model = "fake-vlm"
    client.client = _FakeClient()
    client._job_state = ContextVar("test_vlm_job_state", default=None)
    client._description_cache = {}
    client._cache_lock = threading.Lock()
    return client


def test_vlm_budget_resets_per_job_and_caches_identical_images():
    client = _client_with_budget(max_pages=1)

    token = client.begin_job()
    assert asyncio.run(client.process_image(b"image-a")) == "description-1"
    assert asyncio.run(client.process_image(b"image-b")) == ""
    client.end_job(token)

    token = client.begin_job()
    assert asyncio.run(client.process_image(b"image-a")) == "description-1"
    assert client.get_processed_count() == 0
    client.end_job(token)
