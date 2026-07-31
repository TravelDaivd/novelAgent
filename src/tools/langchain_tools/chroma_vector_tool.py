from langchain_core.tools import BaseTool, Tool, StructuredTool
from pydantic import BaseModel, Field

from models.llm.llm_chroma import LlmChroma


class ChromaTools:
    
    def __init__(self):
        self.llmChroma = LlmChroma()
    
    def create_chroma_tools(self)->list[BaseTool]:
        class vector_get_segments_input(BaseModel):
            question: str = Field(description="用户问的问题")
            segment_ids: list[str] = Field(description="片段Id列表,格式为 'seg_章节号_序号'，例如 ['seg_10_1', 'seg_10_2', 'seg_11_4']")
        vector_get_segments_tool = StructuredTool.from_function(
            name="vector_get_segments",
            func=self.llmChroma.handle_get_segments,
            description="根据用户问题和片段Id列表，获取片段",
            args_schema=vector_get_segments_input
        )
        class vector_get_chapter_input(BaseModel):
            question: str = Field(description="用户问的问题")
            chapter_ids: list[int] = Field(description="章节ID列表，字段名：chapter_ids，数字列表，示例：[10, 11] ")
        vector_get_chapter_tool = StructuredTool.from_function(
            name="get_chapter",
            func=self.llmChroma.handle_get_chapter,
            description="根据用户问题和章节Id列表，获取章节完整信息",
            args_schema=vector_get_chapter_input
        )
        
        class vector_get_segment_by_label_input(BaseModel):
            question: str = Field(description="用户问的问题")
            chapter_ids: list[int] = Field(description="章节ID列表，字段名：chapter_ids，数字列表，示例：[10, 11] ")
            label: list[str] = Field(description="标签列表，可选值：战斗、修炼、内心、探索、对话。必须从可选值中选择，示例：['战斗', '对话'] ")
        vector_get_segment_by_label_tool = StructuredTool.from_function(
            name="vector_get_segment_by_label",
            func=self.llmChroma.handle_get_segments_by_label,
            description="根据章节ID和标签，筛选片段",
            args_schema=vector_get_segment_by_label_input
        )
        
        class vector_search_segments_by_keyword_input(BaseModel):
            keywords: str = Field(description="搜索关键词，从用户问题中提取")
            chapter_ids: list[int] = Field(default= None,description="章节ID列表，字段名：chapter_ids，数字列表，示例：[10, 11] ")
        vector_search_segments_by_keyword_tool = StructuredTool.from_function(
            name="vector_search_segments_by_keyword",
            func=self.llmChroma.handle_search_segments_by_keyword,
            description="用关键词搜索片段内容。仅当没有图谱返回的片段ID和章节ID时使用",
            args_schema=vector_search_segments_by_keyword_input
        )
        return [vector_get_segments_tool,
                vector_get_chapter_tool,
                vector_get_segment_by_label_tool,
                vector_search_segments_by_keyword_tool]