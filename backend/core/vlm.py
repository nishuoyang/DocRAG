"""VLM 多模态客户端：页面读图（扫描件 OCR）+ 图片描述。"""
import asyncio
import base64
import hashlib
import logging
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)

# VLM 调用类型
VLMTask = Literal["page_read", "image_describe"]


@dataclass
class _VLMJobState:
    processed_count: int = 0

# 页面读图 prompt：扫描件 OCR，按原始版式输出 Markdown
PAGE_READ_PROMPT = """你是一个文档 OCR 助手。请将这张图片中的文字内容按原始版式提取为 Markdown 格式。

要求：
1. 保持原始排版结构（标题、段落、列表、表格等）
2. 表格转为 Markdown 表格格式
3. 公式保留原文（LaTeX 格式）
4. 图片/图表用 [图片: 简要描述] 占位
5. 如果页面完全空白或无法识别，返回空字符串

直接输出 Markdown 内容，不要额外解释。"""

# 图片描述 prompt：客观描述图表/图片信息
IMAGE_DESCRIBE_PROMPT = """你是一个图片描述助手。请客观描述这张图片/图表中的关键信息和数据。

要求：
1. 描述图片类型（照片、图表、流程图、截图等）
2. 提取关键数据、文字、标签
3. 如果是图表，说明数据趋势或对比关系
4. 保持客观，不要推测或解读
5. 如果图片无实质内容（装饰性图片、纯背景等），返回空字符串

直接输出描述，不要额外解释。"""


class VLMClient:
    """VLM 多模态客户端。"""

    def __init__(self):
        settings = get_settings()
        self.enabled = settings.VLM_ENABLED
        self.model = settings.VLM_MODEL
        self.base_url = settings.VLM_BASE_URL
        self.api_key = settings.VLM_API_KEY
        self.max_pages = settings.VLM_MAX_PAGES
        self._job_state: ContextVar[_VLMJobState | None] = ContextVar(
            f"vlm_job_state_{id(self)}",
            default=None,
        )
        self._description_cache: dict[tuple[str, str], str] = {}
        self._cache_lock = threading.Lock()

        if self.enabled and not self.api_key:
            logger.warning("VLM_ENABLED=true 但 VLM_API_KEY 未配置，VLM 功能将被禁用")
            self.enabled = False

        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0,
            )
            logger.info(f"VLM 客户端初始化: model={self.model}, max_pages={self.max_pages}")

    def begin_job(self) -> Token:
        """Start a fresh VLM budget for the current ingestion context."""
        return self._job_state.set(_VLMJobState())

    def end_job(self, token: Token) -> None:
        """Restore the previous context-local budget."""
        self._job_state.reset(token)

    async def process_image(
        self,
        image_data: bytes,
        task: VLMTask = "image_describe",
    ) -> str:
        """处理单张图片。

        Args:
            image_data: 图片二进制数据
            task: 任务类型（page_read 或 image_describe）

        Returns:
            VLM 输出文本，失败返回空字符串
        """
        if not self.enabled:
            return ""

        cache_key = (task, hashlib.sha256(image_data).hexdigest())
        with self._cache_lock:
            cached = self._description_cache.get(cache_key)
        if cached is not None:
            return cached

        # 每个上传任务独立熔断
        job_state = self._job_state.get()
        if job_state is None:
            job_state = _VLMJobState()
            self._job_state.set(job_state)
        if job_state.processed_count >= self.max_pages:
            logger.warning(f"VLM 处理已达上限 ({self.max_pages})，跳过后续处理")
            return ""

        try:
            # 编码为 base64
            b64 = base64.b64encode(image_data).decode()

            # 选择 prompt
            prompt = PAGE_READ_PROMPT if task == "page_read" else IMAGE_DESCRIBE_PROMPT

            # 调用 VLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=2000,
            )

            result = response.choices[0].message.content.strip()
            job_state.processed_count += 1

            # 过滤空结果
            if not result or result.lower() in ["", "空字符串", "无内容", "无法识别"]:
                result = ""

            with self._cache_lock:
                self._description_cache[cache_key] = result
            logger.debug(
                f"VLM 处理成功 (task={task}, count={job_state.processed_count})"
            )
            return result

        except Exception as e:
            logger.warning(f"VLM 处理失败 (task={task}): {e}")
            return ""

    async def process_page_image(self, image_data: bytes) -> str:
        """处理页面图片（扫描件 OCR）。"""
        return await self.process_image(image_data, task="page_read")

    async def process_inline_image(self, image_data: bytes) -> str:
        """处理内嵌图片（图片描述）。"""
        return await self.process_image(image_data, task="image_describe")

    def get_processed_count(self) -> int:
        """获取当前任务实际调用 VLM 的图片/页面数量。"""
        job_state = self._job_state.get()
        return job_state.processed_count if job_state is not None else 0


# 全局实例（延迟初始化）
_vlm_client: VLMClient | None = None


def get_vlm_client() -> VLMClient:
    """获取 VLM 客户端实例。"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client


def reset_vlm_client():
    """重置 VLM 客户端（用于测试或重新加载配置）。"""
    global _vlm_client
    _vlm_client = None


async def process_page_image_async(image_data: bytes) -> str:
    """异步处理页面图片（便捷函数）。"""
    client = get_vlm_client()
    return await client.process_page_image(image_data)


async def process_inline_image_async(image_data: bytes) -> str:
    """异步处理内嵌图片（便捷函数）。"""
    client = get_vlm_client()
    return await client.process_inline_image(image_data)


def process_page_image_sync(image_data: bytes) -> str:
    """同步处理页面图片（在异步上下文中使用）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已在异步上下文中，创建新任务
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, process_page_image_async(image_data))
            return future.result()
    else:
        # 不在异步上下文中，直接运行
        return asyncio.run(process_page_image_async(image_data))


def process_inline_image_sync(image_data: bytes) -> str:
    """同步处理内嵌图片（在异步上下文中使用）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, process_inline_image_async(image_data))
            return future.result()
    else:
        return asyncio.run(process_inline_image_async(image_data))
