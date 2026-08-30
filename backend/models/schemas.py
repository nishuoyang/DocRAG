"""Pydantic 请求/响应模型（带描述，用于生成完整 Swagger 文档）。"""
from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str = Field(description="引用来源的文件名")
    chunk_index: int | None = Field(default=None, description="命中的文档块序号")
    page: int | None = Field(default=None, description="命中的页码（PDF 有值）")
    content: str = Field(description="命中的文档块内容")


class DocumentInfo(BaseModel):
    filename: str = Field(description="文件名")
    chunk_count: int = Field(description="该文档入库的分块数量")
    upload_time: int | None = Field(default=None, description="上传时间戳（秒）")
    file_hash: str | None = Field(default=None, description="文件内容 SHA256 哈希（去重用）")


class DocumentUploadResponse(BaseModel):
    filename: str = Field(description="已上传的文件名")
    chunk_count: int = Field(description="入库的分块数量")
    ids: list[int] = Field(description="写入 Milvus 的向量 ID 列表")
    chunk_type: str | None = Field(default=None, description="实际使用的切分策略：semantic / fixed / markdown")
    file_hash: str | None = Field(default=None, description="文件内容 SHA256 哈希")
    vlm_pages: int = Field(default=0, description="VLM 多模态实际处理的页数/图片数（成本可见）")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(description="全部已入库文档")
    total: int = Field(description="文档总数")


class ChatMessage(BaseModel):
    role: str = Field(description="消息角色，user 或 assistant")
    content: str = Field(description="消息内容")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索返回的文档块数量，默认取配置值 5")
    history: list[ChatMessage] = Field(default_factory=list, description="历史对话（不含当前问题），用于多轮上下文")


class AgentChatRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题（主管 agent 将拆解并调度成员）")
    top_k: int | None = Field(default=None, ge=1, le=20, description="文档检索返回块数，默认取配置值 5")
    history: list[ChatMessage] = Field(default_factory=list, description="历史对话（不含当前问题）")


class ChatResponse(BaseModel):
    answer: str = Field(description="LLM 生成的回答")
    sources: list[Source] = Field(description="回答引用的文档块来源")


class MemoryResponse(BaseModel):
    messages: list[ChatMessage] = Field(description="历史对话消息（按时间升序）")


class JobInfoResponse(BaseModel):
    job_id: str = Field(description="任务 ID")
    filename: str = Field(description="文件名")
    status: str = Field(description="任务状态：pending / processing / completed / failed")
    progress: int = Field(description="处理进度（0-100）")
    message: str = Field(description="当前状态描述")
    created_at: str = Field(description="创建时间（ISO 格式）")
    updated_at: str = Field(description="更新时间（ISO 格式）")
    result: dict | None = Field(default=None, description="任务结果（仅 completed 状态有值）")
    error: str | None = Field(default=None, description="错误信息（仅 failed 状态有值）")
