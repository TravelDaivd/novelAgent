import logging
import uuid
from datetime import datetime
from typing import Dict, List

from langchain_core.messages import BaseMessage, ToolMessage

from ai_agent.models.agent.agent_config import AgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class SessionManager:
    """会话管理器"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.sessions: Dict[str, Dict] = {}

    def get_or_create(self, session_id: str = None) -> str:
        """获取或创建会话"""
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "cache": {},
                "stats": {"total_calls": 0, "total_duplicates": 0}
            }
            logger.info(f"创建新会话: {session_id}")

        return session_id

    def get_session(self, session_id: str) -> Dict:
        """获取会话数据"""
        return self.sessions.get(session_id, {})

    def add_message(self, session_id: str, message: BaseMessage):
        """添加消息到会话"""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append(message)
            # 限制历史长度
            if len(self.sessions[session_id]["messages"]) > self.config.max_history_length:
                self.sessions[session_id]["messages"] = \
                    self.sessions[session_id]["messages"][-self.config.max_history_length:]

    def get_history(self, session_id: str, limit: int = None) -> List[BaseMessage]:
        """获取历史消息"""
        if session_id not in self.sessions:
            return []
        messages = self.sessions[session_id]["messages"]
        limit = limit or self.config.max_history_length
        return messages[-limit:] if messages else []

    def get_cache(self, session_id: str) -> Dict:
        """获取会话缓存"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("cache", {})
        return {}

    def set_cache(self, session_id: str, cache: Dict):
        """设置会话缓存"""
        if session_id in self.sessions:
            self.sessions[session_id]["cache"] = cache

    def update_stats(self, session_id: str, tool_calls: int, duplicates: int):
        """更新统计"""
        if session_id in self.sessions:
            self.sessions[session_id]["stats"]["total_calls"] += tool_calls
            self.sessions[session_id]["stats"]["total_duplicates"] += duplicates

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def set_phase_result(self, session_id: str, phase: str, result: List[ToolMessage]):
        if session_id in self.sessions:
            self.sessions[session_id][f"{phase}_result"] = result

    def get_phase_result(self, session_id: str, phase: str) -> List[ToolMessage]:
        if session_id in self.sessions:
            return self.sessions[session_id].get(f"{phase}_result", [])
        return []

