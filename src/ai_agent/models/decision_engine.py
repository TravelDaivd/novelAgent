import logging
from typing import Tuple, Optional, List

from langchain_core.messages import ToolMessage

from ai_agent.models.agent.state_manager import ReActState, LangGraphState
from ai_agent.models.agent_langgraph import AgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DecisionEngine:
    """统一决策引擎"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.phase_order = ["graph", "segment", "document"]

    def should_continue_react(self, react_state: ReActState) -> Tuple[bool, str]:
        """
        判断ReAct是否继续
        返回: (是否继续, 原因)
        """
        
        # 1. 阶段已完成
        if react_state.get("phase_complete", False):
            return False, "阶段已完成"

        # 2. 达到最大迭代次数
        iteration = react_state.get("iteration", 0)
        max_iter = react_state.get("max_iterations", 1)
        if iteration >= max_iter:
            return False, f"达到最大迭代次数 {max_iter}"

        # 3. 检测空结果（自动切换）
        if react_state.get("has_empty_result", False) and self.config.enable_auto_switch:
            return False, "工具返回空结果"
        # 4. 信息已充足
        phase = react_state.get("phase")
        if self.is_information_sufficient(react_state.get("phase_result", []),phase):
            return False, "信息已充足"
        
        return True, "继续ReAct"

    def get_next_phase(self, current_phase: str) -> Optional[str]:
        """获取下一阶段"""
        try:
            idx = self.phase_order.index(current_phase)
            return self.phase_order[idx + 1] if idx < len(self.phase_order) - 1 else None
        except ValueError:
            return None
    def get_prev_phase(self, current_phase: str) -> Optional[str]:
        """获取上一阶段"""
        try:
            idx = self.phase_order.index(current_phase)
            return self.phase_order[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def should_enter_next_phase(self, state: LangGraphState, current_phase: str) -> Tuple[bool, str]:
        """判断是否应该进入下一阶段"""
        next_phase = self.get_next_phase(current_phase)
        if not next_phase:
            return False, "已是最后阶段"

        react_state = state.get(f"{current_phase}_react", {})
        user_question = state.get("user_question", "")
        current_result = state.get(f"{current_phase}_result", [])
        
        # 1. 用户明确要求原文（直接跳到document阶段）
        if next_phase == "document":
            keywords = ["原文", "完整内容", "详细描述", "原话", "具体描写"]
            if any(kw in user_question for kw in keywords):
                return True, "用户明确要求原文"
            
        # 2. 片段chroma没有查询到任何数据，进入下一阶段(降级策略)
        if current_result == "segment" and len(current_result) == 0:
            return True, f"当前阶段是{current_phase} 结果为空,要进行下一个阶段"
        
        # 3. 图谱没有查询到任何数据，进入下一阶段(降级策略)
        if current_result == "graph" and len(current_result) == 0:
            return True, f"当前阶段是{current_phase} 结果为空,要进行下一个阶段"
        
        # 4.在图谱的有数据的情况下，主动进入下一阶段
        if current_phase == "graph" and next_phase == "segment":
            return True, f"当前阶段是{current_phase},要进行下一个阶段"
  
  
        return False, "信息充足，无需进入下一阶段"

    @staticmethod
    def is_information_sufficient(results: List[ToolMessage],phase:str) -> bool:
        """检查信息是否充足"""
        min_valid_count = 2
        valid_count = 0
        
        if phase == "graph":
            min_valid_count = 3
       
        for msg in results:
            if DecisionEngine.has_valid_result([msg]):
                valid_count += 1
        return valid_count >= min_valid_count

    @staticmethod
    def has_valid_result(results: List[ToolMessage]) -> bool:
        """检查是否有有效结果"""
        empty_patterns = ["[]", "{}", "null", "None", "无结果", "空", "失败", "不存在"]
        for msg in results:
            if isinstance(msg, ToolMessage) and msg.content:
                content = msg.content.strip()
                if content and not any(p in content for p in empty_patterns):
                    return True
        return False
