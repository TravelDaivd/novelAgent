import json
import logging
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_agent.models.agent.agent_config import AgentConfig
from ai_agent.models.agent.agent_utils import AgentUtils
from ai_agent.models.agent.cache_manager import CacheManager
from ai_agent.models.agent.state_manager import LangGraphState
from ai_agent.models.agent.system_prompt import SystemPrompt
from ai_agent.models.decision_engine import DecisionEngine
from utils.config import DEEPSEEK_API_KEY, DEEPSEEK_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PhaseReActExecutor:
    
    def __init__(self,agentConfig: AgentConfig,phase_config,cacheManager :CacheManager):
        # 初始化工具
        
        self.agent_config  = agentConfig
        self.phase_config  = phase_config
        self.cache_manager  = cacheManager
        self.decisionEngine = DecisionEngine(agentConfig)
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=agentConfig.model_name,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            temperature=agentConfig.temperature
        )
        self.messages = [SystemMessage(content=SystemPrompt.get_prompt())]
        
    # ========================================================================
    # 各个阶段入口初始化
    # ========================================================================

    def graph_entry(self, state: LangGraphState) -> Dict[str, Any]:
        """图谱阶段入口"""
        return self.prepare_phase(state, "graph")

    def segment_entry(self, state: LangGraphState) -> Dict[str, Any]:
        """片段阶段入口"""
        return self.prepare_phase(state, "segment")

    def document_entry(self, state: LangGraphState) -> Dict[str, Any]:
        """原文阶段入口"""
        return self.prepare_phase(state, "document")

    def prepare_phase(self, state: LangGraphState, phase: str) -> Dict[str, Any]:
        """
        通用阶段入口 检查阶段缓存（复用结果）
        加载上一阶段结果作为上下文 初始化 ReAct 状态
        :param state: 
        :param phase: 阶段名称
        :return: 
        """
        logger.info("=" * 60)
        logger.info(f"阶段: {self.phase_config[phase]['name']}")
        
        # 获取该阶段已有结果（从上一阶段传递） 
        prev_phase = self.decisionEngine.get_prev_phase(phase)
        existing_react = state.get(f"{prev_phase}_react", [])
    
        history_message = state.get("messages", [])
    
        if existing_react and existing_react["messages"]:
            human_message = existing_react["messages"]
        else:
            human_message = history_message
            if history_message:
                human_message = AgentUtils.build_history_message(history_message,state.get("user_question"))
            
        # 如果 messages 不是列表，强制转换
        if not isinstance(human_message, list):
            human_message = [human_message] if human_message else []
        
        # 初始化ReAct状态
        react_state = {
            "messages": human_message,
            "iteration": 0,
            "max_iterations": self.phase_config[phase]["max_iterations"],
            "phase": phase,
            "phase_complete": False,
            "phase_result": [],
            "cache": state.get("session_cache", {}).get(phase, {}),
            "tool_calls": 0,
            "duplicates": 0,
            "has_empty_result": False,
            "decision_reason": ""
        }
        
        return {
            f"{phase}_react": react_state,
            "current_active_phase": phase,
            f"{phase}_complete": False
        }
    # ========================================================================
    # ReAct步骤->各个阶段执行中
    # ========================================================================

    def graph_react_wrapper(self, state: LangGraphState) -> Dict[str, Any]:
        return self.react_step(state, "graph")

    def segment_react_wrapper(self, state: LangGraphState) -> Dict[str, Any]:
        return self.react_step(state, "segment")

    def document_react_wrapper(self, state: LangGraphState) -> Dict[str, Any]:
        return self.react_step(state, "document")

    def react_step(self, state: LangGraphState, phase: str) -> Dict[str, Any]:
        """通用ReAct步骤"""

        react_state = state.get(f"{phase}_react", {})
        
        phase_config = self.phase_config[phase]
        
       
        prompt = self.build_react_prompt(phase)
        messages = self.messages + [prompt] +  react_state["messages"]
        if react_state["phase_complete"]:
            cleaned_messge = AgentUtils.clean_messages_for_new_phase(react_state["messages"])
            messages = self.messages + [prompt] + cleaned_messge

        # 调用LLM
        logger.info(f"{phase} ReAct迭代 {react_state.get('iteration', 0) + 1}")
        llm_with_tools = self.llm.bind_tools(phase_config["tools"])
        response = llm_with_tools.invoke(messages)

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_results, cache_updates, stats = self.execute_tools_with_cache(
                response.tool_calls,
                phase,
                react_state.get("cache", {})
            )
            has_valid = AgentUtils.has_valid_result(tool_results)
            response.content = f"【我的思考】{response.content}"
            # 更新状态
            react_state["messages"]= react_state.get("messages", []) + [response] + tool_results
            react_state["phase_result"].extend(tool_results)
            react_state["iteration"] += 1
            react_state["tool_calls"] += len(response.tool_calls)
            react_state["duplicates"] += stats["duplicates"]
            react_state["has_empty_result"] = not has_valid

            # 判断是否完成
            if not has_valid:
                react_state["decision_reason"] = "工具返回空结果"
                react_state["phase_complete"] = True
            # 更新缓存
            react_state["cache"] = cache_updates

            return {
                f"{phase}_react": react_state,
                f"{phase}_result": react_state["phase_result"],
                "total_tool_calls": state.get("total_tool_calls", 0) + len(response.tool_calls),
                "session_cache": {**state.get("session_cache", {}), phase: cache_updates}
            }

        else:
            # LLM选择不调用工具
            response.content = f"【我的结论】{response.content}"
            react_state["messages"].append(response)
            react_state["phase_complete"] = True
            react_state["decision_reason"] = "LLM主动停止"
            return {
                f"{phase}_react": react_state,
                f"{phase}_complete": True,
            }

    def execute_tools_with_cache(self, tool_calls: List[Dict], phase: str, cache: Dict):
        """
        带缓存的工具执行
        :param tool_calls: LLM在当前阶段所具有的tools中,决策选中的tool
        :param phase: 阶段名称
        :param cache: 缓存 把查询结果存储到ReAct状态中
        :return: 
        """

        results = []
        updated_cache = cache.copy()
        stats = {"calls": 0, "duplicates": 0}
        
        phase_tools = self.phase_config[phase]["tools"]
        tool_map = {t.name: t for t in phase_tools}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            if tool_name not in tool_map:
                # 跳过不在当前阶段工具列表中的工具
                logger.warning(f"工具 {tool_name} 不在 {phase} 阶段工具列表中，跳过")
                continue
            try:
                tool = tool_map.get(tool_name)
                logger.info(f"执行工具: {tool_name}")
                result = tool.invoke(tool_args)
                # 生成签名
                signature = AgentUtils.generate_signature(tool_name, tool_args)
                # 检查缓存
                if self.agent_config.enable_cache:
                    cached_result = self.cache_manager.get_tool_result(signature)
                    if cached_result:
                        logger.info(f"工具缓存命中: {tool_name}")
                        stats["duplicates"] += 1
                        results.append(ToolMessage(
                            content=cached_result,
                            tool_call_id=tool_call_id
                        ))
                        continue
                
                compbild ={"method":tool_name,"response":result.get("data",[]),"request":tool_args}
                result_str = json.dumps(compbild, ensure_ascii=False)
                logger.info(f"执行工具后的结果：{result_str}")
                # 缓存结果
                if self.agent_config.enable_cache:
                    self.cache_manager.set_tool_result(signature, result_str)
                    updated_cache[signature] = result_str

                stats["calls"] += 1
                results.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tool_call_id
                ))
               

            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                results.append(ToolMessage(
                    content=f"执行失败: {str(e)}",
                    tool_call_id=tool_call_id
                ))

        return results, updated_cache, stats
    
    @staticmethod
    def build_react_prompt(phase: str) -> SystemMessage:
        """构建ReAct上下文提示词"""

        phase_prompt=None
        if phase == "graph":
            phase_prompt ="""
                你现在处于【第一阶段：知识图谱查询】。
                【当前任务】
                  使用 Neo4j 图谱工具输出实体、关系、片段ID、章节ID。
                  目标：为后续片段阶段提供精确的片段ID，实现精确定位查询。
                
                【为什么必须先查图谱】
                  - 图谱能快速定位用户关心的内容出现在哪些片段中。
                  - 拿到片段ID后，后续的片段阶段可以直接用ID精确取内容，比全文搜索更快、更准。
                  - 如果把图谱跳过，片段阶段只能靠关键词搜索，效率低且可能漏掉内容。
                
                【规则】
                - 必须调用至少一个工具，禁止在未调用任何工具的情况下结束此阶段
                - 禁止直接从历史消息中提取答案
                - 即使你认为已经知道答案，也必须执行工具调用验证
                - 如果查询无结果，记录空结果 
                - 只能调用本阶段提供的工具，禁止编造或猜测工具名称
            """
        if phase == "segment":
            phase_prompt ="""
                你现在处于【片段检索】阶段。

                【当前任务】
                根据图谱阶段查到的信息，结合当前阶段的工具要求，获取相关片段的具体内容。
                
                【获取方式】（根据手头数据选择对应工具）
                 1. 如果有片段ID → 用片段ID直接获取对应片段内容
                 2. 如果有章节ID → 用章节ID作为过滤条件，检索该章节内相关片段
                 3. 如果只有实体名称 → 用实体名作为关键词，检索包含该实体的片段
                
                【重点关注】
                - 片段里发生了什么情节？
                - 涉及哪些人物？
                - 是否有用户关心的内容：打斗、对话、修炼、内心活动、探索发现？
                
                【停止条件】（满足其一即停止）
                - 已获取足够回答用户问题的片段
                - 连续检索返回空结果
                
            """
        if phase == "document":
            phase_prompt ="""
                你现在处于【原文检索】阶段。

                【当前任务】
                 根据图谱阶段或片段阶段查到的章节ID，结合当前阶段的工具要求，获取完整章节原文。
                
                【章节ID来源】（按优先级）
                 1. 优先使用片段阶段返回的章节ID
                 2. 如果片段阶段没有返回，使用图谱阶段返回的章节ID
                
                【规则】
                - 返回完整的章节原文内容
                - 如果根据章节ID查不到原文，记录“未找到原文”
                - 禁止编造或补充原文中没有的内容
            """
       
        return SystemMessage(content=phase_prompt)