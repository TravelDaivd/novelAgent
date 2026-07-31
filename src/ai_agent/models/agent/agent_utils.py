import hashlib
import logging
from typing import List, Dict

from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class AgentUtils:

    def format_results(results: List[ToolMessage]) -> str:
        """格式化结果（完整内容，不截断）"""
        if not results:
            return "暂无结果"

        formatted = [f"{msg.content}" for msg in results if isinstance(msg, ToolMessage) and msg.content]
        return "\n".join(formatted) if formatted else "暂无结果"

    @staticmethod
    def has_valid_result(results: List[ToolMessage]) -> bool:
        """检查是否有有效结果"""
        empty_patterns = ["[]", "{}", "null", "None", "无结果", "空", "失败", "不存在", "工具执行成功，但无返回内容"]
        for msg in results:
            if isinstance(msg, ToolMessage) and msg.content:
                content = msg.content.strip()
                if content and not any(p in content for p in empty_patterns):
                    return True
        return False

    @staticmethod
    def generate_signature(tool_name: str, args: Dict) -> str:
        """生成工具调用签名"""
        # 忽略动态参数
        ignore_keys = {"timestamp", "time", "random", "seed", "date", "datetime"}
        filtered_args = {k: v for k, v in args.items() if k not in ignore_keys}
        sorted_args = sorted(filtered_args.items())
        signature_str = f"{tool_name}:{sorted_args}"
        return hashlib.md5(signature_str.encode()).hexdigest()

    @staticmethod
    def extract_user_question(messages: List[BaseMessage]) -> str:
        """提取用户问题"""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""


    @staticmethod
    def build_history_message(history_messages:list[BaseMessage],user_question:str):
        rewritten_history = []
        for message in history_messages:
            if isinstance(message, HumanMessage):
                rewritten_history.append(
                    HumanMessage(content=f"【历史】用户问：{message.content}")
                )
            elif isinstance(message, AIMessage) and not message.tool_calls:
                # 只保留不含 tool_calls 的 AIMessage（最终答案）
                rewritten_history.append(
                    HumanMessage(content=f"【历史】你答：{message.content}")
            )
        if rewritten_history:
            rewritten_history = [
                SystemMessage("=== 历史对话（仅用于理解上下文，不用于提供答案） ==="),
            ] + rewritten_history + [
                SystemMessage("=== 历史结束，以下是当前轮次 ==="),
            ] + [
                HumanMessage(content=f"【当前】用户问：{user_question}")
            ]
            
        return rewritten_history
    
    @staticmethod
    def clean_messages_for_new_phase(messages: List[BaseMessage]) -> List[BaseMessage]:
        """清洗消息，移除未完成的 tool_calls，只保留纯文本对话"""
        cleaned = []
        for msg in messages:
            # 如果是 AIMessage 且包含 tool_calls，跳过
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            # 如果是 ToolMessage，也跳过（因为对应的 AIMessage 被移除了）
            if isinstance(msg, ToolMessage):
                continue
            cleaned.append(msg)
        return cleaned
        
        
        
    @staticmethod
    def print_summary(result: Dict):
        """打印执行总结"""
        logger.info("=" * 80)
        logger.info("执行总结")
        logger.info("-" * 80)

        for phase in ["graph", "segment", "document"]:
            react_state = result.get(f"{phase}_react", {})
            iterations = react_state.get("iteration", 0)
            tool_calls = react_state.get("tool_calls", 0)
            duplicates = react_state.get("duplicates", 0)
            reason = react_state.get("decision_reason", "未知")

            logger.info(
                f"{phase}: 迭代 {iterations} 次, 工具调用 {tool_calls} 次, "
                f"重复 {duplicates} 次, 原因: {reason}"
            )

        logger.info("-" * 80)
        total_calls = result.get("total_tool_calls", 0)
        total_duplicates = result.get("total_duplicates", 0)
        logger.info(f"总计: 调用 {total_calls} 次, 重复 {total_duplicates} 次")
        logger.info("=" * 80)