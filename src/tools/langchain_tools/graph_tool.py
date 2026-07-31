from langchain_core.tools import BaseTool, Tool, StructuredTool
from pydantic import BaseModel, Field
from models.llm.llm_graph import LlmGraph


class GraphTools:

    def __init__(self):
        self.llmGraph = LlmGraph()

    def create_graph_tools(self) -> list[BaseTool]:
        class graph_find_segments_by_chapter_input(BaseModel):
            chapter_ids: list[int] = Field(description="章节ID列表，字段名：chapter_ids，数字列表，示例：[10, 11] ")

        graph_find_segments_by_chapter_tool = Tool.from_function(
            name="graph_find_segments_by_chapter",
            func=self.llmGraph.handle_get_segments_ids,
            description="根据章节ID查询片段ID，返回 segment_id 和 chapter_id，供后续片段检索阶段使用",
            args_schema=graph_find_segments_by_chapter_input

        )
        
        class graph_find_entity_metadata_by_segment_input(BaseModel):
            segment_ids: list[str] = Field(
                description="片段Id列表,格式为 'seg_章节号_序号'，例如 ['seg_10_1', 'seg_10_2', 'seg_11_4']")

        graph_find_entity_metadata_by_segment_tool = Tool.from_function(
            name="graph_find_entity_metadata_by_segment",
            func=self.llmGraph.handle_get_entity_names,
            description="""
            根据片段ID列表，获取实体元数据。

            【返回信息】
            - entity_names: 实体名称
            - segment_ids: 该实体出现的片段ID列表（必须提取，用于后续查内容）
            - chapter_ids: 该实体出现的章节ID列表（必须提取，用于后续查章节）
            
            【后续用途】
            - 提取 entity_name → 查关系
            - 提取 segment_ids → 传给 Segment 阶段
            - 提取 chapter_ids → 查片段列表
            【重要】调用此工具后，请从返回中提取所有字段
            """,
            args_schema=graph_find_entity_metadata_by_segment_input

        )
        class graph_find_relations_by_entities_input(BaseModel):
            entity_names: list[str] = Field(description="实体人物名称列表，从上一步返回的 entityName 或用户问题中提取，直接复制原值，禁止编")

        graph_find_relations_by_entities_tool = Tool.from_function(
            name="graph_find_relations_by_entities",
            func=self.llmGraph.handle_get_entity_relations,
            description="根据实体列表查询人物关系（师徒、仇敌、道侣，同门，认识），返回 relation、entity_one、entity_two，供后续分析使用",
            args_schema=graph_find_relations_by_entities_input

        )
        
        return [graph_find_segments_by_chapter_tool, 
                graph_find_entity_metadata_by_segment_tool,
                graph_find_relations_by_entities_tool]

