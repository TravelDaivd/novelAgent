



from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChapterUpload(BaseModel):
    """上传章节请求"""
    chapter_id: int
    title: str
    content: str

class SegmentResult(BaseModel):
    """单个段落推理结果"""
    segment_id: str
    text: str
    label: str
    confidence: float
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]

class UploadResponse(BaseModel):
    """上传响应"""
    status: str
    chapter_id: int
    segment_count: int
    entity_count: int
    relation_count: int

class ChatRequest(BaseModel):
    """对话请求"""
    question: str 

class ChatResponse(BaseModel):
    """对话响应"""
    answer: str
    confidence: float