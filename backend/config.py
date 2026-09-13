"""应用配置。环境变量经 .env 加载，未设置的用 .env.example 中的默认值。"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "doc_collection"

    # Embedding (OpenAI 兼容 API)
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_API_KEY: str = Field(default="", repr=False)
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DIM: int = 1024  # bge-large-zh-v1.5 输出 1024 维

    # Rerank 重排序（默认复用 Embedding 的 base_url 与 key；不配 key 或关闭开关则跳过重排）
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_API_KEY: str = Field(default="", repr=False)

    # LLM
    LLM_PROVIDER: str = "siliconflow"
    LLM_API_KEY: str = Field(default="", repr=False)
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3"
    LLM_BASE_URL: str = ""
    LLM_TEMPERATURE: float = 0.1
    RAG_ANSWER_TEMPERATURE: float = 0.0
    QUERY_ENHANCEMENT_TEMPERATURE: float = 0.0

    # 检索与分块
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    # Parent-child retrieval: children are embedded, parents are reconstructed
    # from sibling children and supplied to the generation model.
    CHILD_CHUNK_SIZE: int = 450
    PARENT_CHUNK_SIZE: int = 2000
    CHILD_MIN_SIZE: int = 120
    RETRIEVAL_CANDIDATE_K: int = 20
    RETRIEVAL_MAX_PARENTS: int = 4
    PARENT_CONTEXT_MAX_CHARS: int = 8000
    PARENT_WINDOW_RADIUS: int = 1
    RETRIEVAL_MODE: str = "balanced"  # fast / balanced / quality
    ANSWERABILITY_GATE_ENABLED: bool = True
    RETRIEVAL_MIN_RERANK_SCORE: float = 0.1
    COLLECTION_SCHEMA_VERSION: str = "parent_child_v1"
    # 多 agent 研究助理工作台
    SEARCH_PROVIDER: str = "tavily"  # 联网搜索服务商：tavily / bocha
    SEARCH_API_KEY: str = Field(default="", repr=False)  # 联网搜索 key；为空时 Search agent 降级不可用
    DATA_EXEC_TIMEOUT: int = 30      # 数据分析子进程超时（秒）
    AGENT_MAX_TURNS: int = 8         # supervisor 最大决策轮数（防死循环）
    AGENT_TOOL_TIMEOUT: float = 60.0 # 单个成员 agent 最大执行时间（秒）
    # legacy 语义切分开关；v2 默认始终使用 parent_child。
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
    VLM_API_KEY: str = Field(default="", repr=False)
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
