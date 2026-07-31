from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from models.llm.llm_chroma import LlmChroma


class ChromaDocumentTool:
    def __init__(self):
        self.llmChroma = LlmChroma()

    def create_chroma_tools(self) -> list[BaseTool]:
        class document_get_chapter_input(BaseModel):
            question: str = Field(description="用户问的问题")
            chapter_ids: list[int] = Field(description="章节ID列表，字段名：chapter_ids，数字列表，示例：[10, 11] ")
    
        document_get_chapter_tool = StructuredTool.from_function(
            name="document_get_chapter",
            func=self.llmChroma.handle_get_document,
            description="根据用户问题和章节Id列表，获取章节内容",
            args_schema=document_get_chapter_input
        )
        return [document_get_chapter_tool]