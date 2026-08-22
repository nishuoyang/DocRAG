"""应用配置。环境变量经 .env 加载，未设置的用 .env.example 中的默认值。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "doc_collection"

    # Embedding (OpenAI 兼容 API)
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DIM: int = 1024  # bge-large-zh-v1.5 输出 1024 维

    # Rerank 重排序（默认复用 Embedding 的 base_url 与 key；不配 key 或关闭开关则跳过重排）
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_API_KEY: str = ""

    # LLM
    LLM_PROVIDER: str = "siliconflow"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3"
    LLM_BASE_URL: str = ""
    LLM_TEMPERATURE: float = 0.1

    # 检索与分块
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    # 语义切分开关：auto 时短文档(<AUTO_SEMANTIC_THRESHOLD)用语义切分、长文档固定切分；
    # true 全部语义、false 全部固定。上传时可被 split_mode 参数覆盖
    SEMANTIC_SPLIT: str = "auto"
    AUTO_SEMANTIC_THRESHOLD: int = 3000  # 语义切分的文档长度阈值（字符）
    # 检索前 Query 增强
    QUERY_TRANSFORM: bool = True   # LLM 改写 query 为检索友好形式
    HYDE: bool = True              # HYDE 假想答案检索

    # 上传限制 (MB)
    MAX_UPLOAD_MB: int = 20

    # ── 解析 v2（结构化解析 + Markdown 分块）──
    # 上传原件落盘目录（支持重解析/迁移重建 collection）
    UPLOAD_DIR: str = "uploads"
    # 解析引擎：v2 = 结构化解析器 + Markdown 结构分块；legacy = 旧链路（回滚开关）
    PARSER_ENGINE: str = "v2"
    # VLM 多模态（P2 预留）：扫描件页面读图 / 内嵌图片描述
    VLM_ENABLED: bool = True
    VLM_MODEL: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    VLM_BASE_URL: str = "https://api.siliconflow.cn/v1"
    VLM_API_KEY: str = ""
    VLM_MAX_PAGES: int = 30  # 单文档 VLM 处理页数上限（成本熔断）

    @property
    def llm_base_url(self) -> str:
        """根据 LLM_PROVIDER 解析 base_url，custom 时读 LLM_BASE_URL。"""
        if self.LLM_PROVIDER == "custom" and self.LLM_BASE_URL:
            return self.LLM_BASE_URL
        return {
            "siliconflow": "https://api.siliconflow.cn/v1",
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
        }[self.LLM_PROVIDER]


@lru_cache
def get_settings() -> Settings:
    return Settings()
