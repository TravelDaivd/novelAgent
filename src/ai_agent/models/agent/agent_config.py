from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent配置"""
    # 模型配置
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.0

    # 阶段配置（最大迭代次数 = 工具数量）
    graph_max_iterations: int = 5  # 图谱工具数
    segment_max_iterations: int = 3  # 片段工具数
    document_max_iterations: int = 2  # 原文工具数

    # 缓存配置
    enable_cache: bool = False
    answer_ttl: int = 3600  # 1小时
    phase_ttl: int = 1800  # 30分钟
    tool_ttl: int = 600  # 10分钟

    # 会话配置
    max_history_length: int = 50
    enable_context_compression: bool = True

    # 决策配置
    enable_auto_switch: bool = True  # 空结果自动切换

    # 系统提示词配置
    enable_system_prompt: bool = True  # 是否启用系统提示词

