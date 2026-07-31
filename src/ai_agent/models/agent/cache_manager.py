import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, List

from langchain_core.messages import ToolMessage

from ai_agent.models.agent.agent_config import AgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """多级缓存管理器"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.answer_cache: Dict[str, Dict] = {}  # {hash: {answer, timestamp}}
        self.phase_cache: Dict[str, Dict] = {}  # {phase_hash: {result, timestamp}}
        self.tool_cache: Dict[str, Dict] = {}  # {signature: {result, timestamp}}

    def get_answer(self, question: str) -> Optional[str]:
        """获取缓存的完整答案"""
        key = hashlib.md5(question.encode()).hexdigest()
        if key in self.answer_cache:
            entry = self.answer_cache[key]
            if self.is_valid(entry, self.config.answer_ttl):
                logger.info(f"答案缓存命中: {question[:30]}...")
                return entry["answer"]
            del self.answer_cache[key]
        return None

    def set_answer(self, question: str, answer: str):
        """缓存完整答案"""
        key = hashlib.md5(question.encode()).hexdigest()
        self.answer_cache[key] = {
            "answer": answer,
            "timestamp": datetime.now().timestamp()
        }

    def get_phase_result(self, phase: str, question: str) -> Optional[List[ToolMessage]]:
        """获取阶段缓存"""
        key = hashlib.md5(f"{phase}_{question}".encode()).hexdigest()
        if key in self.phase_cache:
            entry = self.phase_cache[key]
            if self.is_valid(entry, self.config.phase_ttl):
                logger.info(f"阶段缓存命中: {phase}")
                return entry["result"]
            del self.phase_cache[key]
        return None

    def set_phase_result(self, phase: str, question: str, result: List[ToolMessage]):
        """缓存阶段结果"""
        key = hashlib.md5(f"{phase}_{question}".encode()).hexdigest()
        self.phase_cache[key] = {
            "result": result,
            "timestamp": datetime.now().timestamp()
        }

    def get_tool_result(self, signature: str) -> Optional[str]:
        """获取工具缓存"""
        if signature in self.tool_cache:
            entry = self.tool_cache[signature]
            if self.is_valid(entry, self.config.tool_ttl):
                logger.info(f"工具缓存命中: {signature}")
                return entry["result"]
            del self.tool_cache[signature]
        return None

    def set_tool_result(self, signature: str, result: str):
        """缓存工具结果"""
        self.tool_cache[signature] = {
            "result": result,
            "timestamp": datetime.now().timestamp()
        }

    @staticmethod
    def is_valid(entry: Dict, ttl: int) -> bool:
        """检查缓存是否有效"""
        return datetime.now().timestamp() - entry["timestamp"] < ttl

    def clear(self):
        """清空所有缓存"""
        self.answer_cache.clear()
        self.phase_cache.clear()
        self.tool_cache.clear()

