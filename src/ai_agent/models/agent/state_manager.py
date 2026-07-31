from typing import List, Annotated, TypedDict, Dict, Optional, Any

from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.graph import add_messages



class ReActState(TypedDict):
    """ReAct循环状态"""
    messages: list[BaseMessage]
    iteration: int
    max_iterations: int
    phase: str
    phase_complete: bool
    phase_result: List[ToolMessage]
    cache: Dict[str, str]
    tool_calls: int
    duplicates: int
    has_empty_result: bool
    decision_reason: str

class LangGraphState(TypedDict):
    """主状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_question: str
    session_id: str
    
    #当前活跃阶段
    current_active_phase : str 
    
    # 各阶段ReAct状态
    graph_react: ReActState
    segment_react: ReActState
    document_react: ReActState

    # 阶段结果（汇总）
    graph_result: List[ToolMessage]
    segment_result: List[ToolMessage]
    document_result: List[ToolMessage]

    # 阶段完成标记
    graph_complete: bool
    segment_complete: bool
    document_complete: bool

    # 缓存（会话级）
    session_cache: Dict[str, Dict]

    # 统计
    total_tool_calls: int
    total_duplicates: int

    # 最终结果
    final_answer: Optional[str]
    start_time: Optional[str]
    error: Optional[str]
