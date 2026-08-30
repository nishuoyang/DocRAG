"""Agent 工具的统一输入/输出类型。"""
from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    """成员 agent 的统一输入。"""
    question: str = Field(description="要处理的用户问题")
    context: str = Field(default="", description="supervisor 提供的已有上下文（其他 agent 的结果摘要）")


class AgentResult(BaseModel):
    """成员 agent 的统一输出。content 由 supervisor 打包进工具消息返回 LLM。"""
    content: str = Field(description="文字产出（回答/摘要/错误说明）")
    sources: list[dict] = Field(default_factory=list, description="来源列表：doc 型 {title,file,page,content} / web 型 {title,url,content}")
    meta: dict = Field(default_factory=dict, description="附加元数据（图表 base64 等）")
