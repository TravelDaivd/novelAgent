"""
================================================================================
单文件Agent实现 - 外层Pipeline + 内层ReAct混合架构
支持多轮对话、智能缓存、自主决策
================================================================================
"""

import logging
from datetime import datetime
from typing import  List,  Dict, Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai_agent.models.agent.agent_config import AgentConfig
from ai_agent.models.agent.agent_utils import AgentUtils
from ai_agent.models.agent.cache_manager import CacheManager
from ai_agent.models.agent.session_manager import SessionManager
from ai_agent.models.agent.state_manager import LangGraphState
from ai_agent.models.decision_engine import DecisionEngine
from ai_agent.models.phase_react_executor import PhaseReActExecutor
from tools.langchain_tools.chroma_document_tool import ChromaDocumentTool
from tools.langchain_tools.chroma_vector_tool import ChromaTools
from tools.langchain_tools.graph_tool import GraphTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentLangGraph:
    """
    外层Pipeline + 内层ReAct混合架构Agent

    流程：
    1. 知识图谱查询（ReAct循环）
    2. 片段检索（ReAct循环）
    3. 原文检索（ReAct循环）
    4. 生成最终答案
    """

    def __init__(self, config: AgentConfig = None, session_id: str = None):
        """初始化Agent"""
        self.graph_tool = GraphTools()
        self.chroma_tool = ChromaTools()
        self.chroma_document_tool = ChromaDocumentTool()
        self.config = config or AgentConfig()
        self.session_id = session_id

        # 阶段配置
        self.phase_config = {
            "graph": {
                "name": "知识图谱",
                "tools": self.graph_tool.create_graph_tools(),
                "max_iterations": len(self.graph_tool.create_graph_tools())
            },
            "segment": {
                "name": "片段检索",
                "tools": self.chroma_tool.create_chroma_tools(),
                "max_iterations": len(self.chroma_tool.create_chroma_tools()),
            },
            "document": {
                "name": "原文检索",
                "tools": self.chroma_document_tool.create_chroma_tools(),
                "max_iterations": len(self.chroma_document_tool.create_chroma_tools()),
            }
        }
        
        # 初始化管理器
        self.cache_manager = CacheManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.decision_engine = DecisionEngine(self.config)
        self.phaseReActExecutor = PhaseReActExecutor(self.config, self.phase_config, self.cache_manager)

        # 构建图
        self.graph = self.build_graph()
        self.memory = MemorySaver()
        self.agent = self.graph.compile(checkpointer=self.memory)
        logger.info(f"Agent初始化完成，会话: {self.session_id or '未创建'}")
        logger.info(f"阶段配置: {list(self.phase_config.keys())}")

    # ========================================================================
    # 图构建
    # ========================================================================

    def build_graph(self) -> StateGraph:
        """构建LangGraph图"""

        graph = StateGraph(LangGraphState)

        # 添加节点
        graph.add_node("graph_entry", self.phaseReActExecutor.graph_entry)
        graph.add_node("graph_react", self.phaseReActExecutor.graph_react_wrapper)
        graph.add_node("segment_entry", self.phaseReActExecutor.segment_entry)
        graph.add_node("segment_react", self.phaseReActExecutor.segment_react_wrapper)
        graph.add_node("document_entry", self.phaseReActExecutor.document_entry)
        graph.add_node("document_react", self.phaseReActExecutor.document_react_wrapper)
        graph.add_node("finalize", self.finalize)

        # 启动
        graph.add_edge(START, "graph_entry")

        # 阶段1：图谱
        graph.add_edge("graph_entry", "graph_react")
        graph.add_conditional_edges(
            "graph_react",
            self.decide_after_react,
            {
                "continue": "graph_react",
                "next_phase": "segment_entry",
                "finalize": "finalize"
            }
        )

        # 阶段2：片段
        graph.add_edge("segment_entry", "segment_react")
        graph.add_conditional_edges(
            "segment_react",
            self.decide_after_react,
            {
                "continue": "segment_react",
                "next_phase": "document_entry",
                "finalize": "finalize"
            }
        )

        # 阶段3：原文
        graph.add_edge("document_entry", "document_react")
        graph.add_conditional_edges(
            "document_react",
            self.decide_after_react,
            {
                "continue": "document_react",
                "next_phase": "finalize",
                "finalize": "finalize"
            }
        )

        graph.add_edge("finalize", END)

        return graph


   
    # ========================================================================
    # 决策节点
    # ========================================================================

    def decide_after_react(self, state: LangGraphState) -> Literal["continue", "next_phase", "finalize"]:
        """ReAct后决策"""
        # 检测活跃阶段
        active_phase = state.get('current_active_phase')
        logger.info(f"当前活跃阶段：{active_phase}")

        if not active_phase:
            return "finalize"

        react_state = state.get(f"{active_phase}_react", {})

        # 判断是否继续ReAct
        should_continue, reason = self.decision_engine.should_continue_react(react_state)
        if should_continue:
            logger.info(f"继续ReAct: {reason}")
            return "continue"
    
        state[f"{active_phase}_complete"] = True
        logger.info(f"ReAct结束: {reason}")

        # 判断是否进入下一阶段
        should_enter, reason = self.decision_engine.should_enter_next_phase(state, active_phase)
        if should_enter:
            logger.info(f"进入下一阶段: {reason}")
            return "next_phase"

        logger.info("生成最终答案")
        return "finalize"

    # ========================================================================
    # 最终答案生成
    # ========================================================================

    def finalize(self, state: LangGraphState) -> Dict[str, Any]:
        """生成最终答案"""

        logger.info("=" * 60)
        logger.info("生成最终答案")

        # 收集所有结果
        graph_results = state.get("graph_result", [])
        segment_results = state.get("segment_result", [])
        document_results = state.get("document_result", [])
        
        # 构建最终提示词
        final_prompt = SystemMessage(content="""
             请根据所有阶段查到的结果，生成最终答案。

            【回答要求】
             1. 只基于当前轮次查询到的图谱、片段、原文结果，归纳出用户问题的答案
             2. 逐条回答用户问题中的每个具体问题，不能跳过任何一个
             3. 用连贯的段落叙述，先回答核心问题，再补充相关背景
             4. 只讲查到的内容，不要提"我查了图谱"、"片段返回了"这类操作过程
             5. 只输出信息本身，不要输出任何ID、JSON结构、字段名
             6. 回答要详细，基于查到的片段内容展开描述
             7. 每个用户问题都要有对应的详细回答，不能合并成一段话敷衍
             8. 禁止使用历史对话中已出现的答案。当前轮次必须基于本轮查询结果重新回答
            【组织方式】
             按用户问题的顺序，一个问题一个问题地回答。
             用户问了几件事，就分几部分回答，让用户知道哪个回答对应哪个问题。
            【信息不足时】
            - 完全没有查到相关内容：只说一句"关于[用户问的内容]，目前还查不到更多信息"
            - 查到一部分内容：用已有信息回答，结尾加一句"以上是目前能查到的内容"
            
            【禁令】
            - 禁止复制原文，要用自己的话归纳
            - 禁止编造细节、添加原文没有的内容
            - 禁止说"根据查询结果"、"基于已有信息"等套话
            - 禁止从历史消息或上一轮对话中提取答案
        """)

        messages =self.phaseReActExecutor.messages+[
            final_prompt,
            HumanMessage(content=f"【图谱结果】{AgentUtils.format_results(graph_results)}"),
            HumanMessage(content=f"【片段结果】{AgentUtils.format_results(segment_results)}"),
            HumanMessage(content=f"【原文结果】{AgentUtils.format_results(document_results)}")
        ]
        response = self.phaseReActExecutor.llm.invoke(messages)
        final_answer = response.content
     
        # 缓存答案
        question = state.get("user_question", "")
        if question:
            self.cache_manager.set_answer(question, final_answer)
        
        logger.info(f"最终答案生成完成，长度: {len(final_answer)} 字符")

        return {
            "messages":[],
            "final_answer": final_answer
        }
    # ========================================================================
    # 对外接口
    # ========================================================================

    def execute(self, question: str, session_id: str = None) -> str:
        
        # 会话管理
        session_id = session_id or self.session_id
        if not session_id:
            session_id = self.session_manager.get_or_create()
        else:
            session_id = self.session_manager.get_or_create(session_id)
        
        logger.info(f"会话ID：{session_id}")
        # 检查答案缓存
        cached_answer = self.cache_manager.get_answer(question)
        if cached_answer:
            logger.info("返回缓存答案")
            return cached_answer

        # 加载历史
        history_message = self.session_manager.get_history(session_id)
        lang_graph_message = [HumanMessage(content=question)]
        if history_message :
            lang_graph_message =history_message
        
        # 构建状态
        initial_state = {
            "messages":  lang_graph_message,
            "user_question": question,
            "session_id": session_id,
            "graph_react": {},
            "segment_react": {},
            "document_react": {},
            "graph_result": [],
            "segment_result": [],
            "document_result": [],
            "graph_complete": False,
            "segment_complete": False,
            "document_complete": False,
            "session_cache": self.session_manager.get_cache(session_id),
            "total_tool_calls": 0,
            "total_duplicates": 0,
            "final_answer": None,
            "start_time": datetime.now().isoformat(),
            "error": None
        }

        config = {"configurable": {"thread_id": session_id}}

        try:
            logger.info(f"处理问题: {question[:50]}...")
            result = self.agent.invoke(initial_state, config)

            final_answer = result.get("final_answer")
            if not final_answer:
                # 提取最后一条消息
                messages = result.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content
                        break

            # 更新会话
            if final_answer:
                self.session_manager.add_message(session_id, HumanMessage(content=question))
                self.session_manager.add_message(session_id, AIMessage(content=final_answer))
                self.session_manager.set_cache(session_id, result.get("session_cache", {}))

                
                # 更新统计
                tool_calls = result.get("total_tool_calls", 0)
                duplicates = result.get("total_duplicates", 0)
                self.session_manager.update_stats(session_id, tool_calls, duplicates)
                
            AgentUtils.print_summary(result)
            return final_answer or "抱歉，未能生成回答"

        except Exception as e:
            logger.error(f"执行失败: {e}",exc_info=True)
            return f"处理失败: {str(e)}"

    # ========================================================================
    # 公共接口-给前端调用
    # ========================================================================

    def stream(self, question: str, session_id: str = None):
        """流式执行（TODO）"""
        # 可扩展流式输出
        yield self.execute(question, session_id)

    def get_session_history(self, session_id: str = None) -> List[BaseMessage]:
        """获取会话历史"""
        session_id = session_id or self.session_id
        if session_id:
            return self.session_manager.get_history(session_id)
        return []

    def clear_session(self, session_id: str = None):
        """清除会话"""
        session_id = session_id or self.session_id
        if session_id:
            self.session_manager.clear_session(session_id)
            logger.info(f"会话已清除: {session_id}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "sessions": len(self.session_manager.sessions),
            "answer_cache": len(self.cache_manager.answer_cache),
            "phase_cache": len(self.cache_manager.phase_cache),
            "tool_cache": len(self.cache_manager.tool_cache)
        }
    