import logging
import os
import time
import threading
from typing import List, Optional
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

from utils.config import VECTOR_DATABASES_DATA_DIR, BGE_RERANKER_V2_M3_NAME

logger = logging.getLogger(__name__)

class ChromaReranker:

    _instance = None
    _model = None
    _tokenizer = None
    
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.max_batch_size = 5
        self.model_file_path = os.path.join(VECTOR_DATABASES_DATA_DIR, BGE_RERANKER_V2_M3_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_seconds = 30
        self.max_length = 512
        self._load_model()

    def _load_model(self):
        """加载模型（只执行一次）"""
        if ChromaReranker._model is  None:
            try:
                # 1. 加载分词器
                ChromaReranker._tokenizer = AutoTokenizer.from_pretrained( self.model_file_path,trust_remote_code=True)
                # 2. 加载模型
                ChromaReranker._model = AutoModelForSequenceClassification.from_pretrained(self.model_file_path,trust_remote_code=True)
                # 3. 移动到指定设备
                ChromaReranker._model.to(self.device)
                ChromaReranker._model.eval()  
                logger.info(f"模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise RuntimeError(f"Rerank 模型加载失败: {e}")

    def rerank(self,query: str,documents: List[str],top_k: int = 5,batch_size: Optional[int] = None,fallback: bool = True) :
        """
        对文档进行重排序
        Args:
            query: 用户查询
            documents: 待排序文档列表
            top_k: 返回前 K 个结果
            batch_size: 批次大小（自动分批处理）
            fallback: 失败时是否降级（返回原始顺序）
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 排序失败且 fallback=False
        """
        # 1. 输入校验
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        if not documents:
            return []
        if top_k <= 0:
            top_k = 1
        # 2. 确定批次大小
        if batch_size is None:
            batch_size = self.max_batch_size
        # 3. 分批处理
        all_scores = []
        try:
            
            timer = threading.Timer(self.timeout_seconds, lambda: None)
            timer.start()

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                try:
                    batch_scores = self.compute_batch(query,batch_docs)
                    all_scores.extend(batch_scores)
                except Exception as e:
                    logger.error(f"批次 {i // batch_size + 1} 排序失败: {e}")
                    # 该批次使用 0 分
                    all_scores.extend([0.0] * len(batch_docs))

            timer.cancel()

        except Exception as e:
            logger.error(f"Rerank 过程异常: {e}")
            if fallback:
                # 降级：返回原始顺序
                logger.warning(f"Rerank 降级到原始顺序")
                return [
                    {"document": doc, "score": float(0.0)}
                    for i, doc in enumerate(documents)
                ]
            else:
                raise RuntimeError(f"Rerank 失败: {e}")

        # 4. 排序并返回 Top-K
        results = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(documents, all_scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    
    
    def compute_batch(self,query,batch_docs):
        pairs = [[query, doc] for doc in batch_docs]
        # 分词
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length= 512,
            return_tensors='pt'
        )
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = ChromaReranker._model(**inputs)
            logits = outputs.logits
            scores = torch.sigmoid(logits).view(-1,).float()

        # 返回分数列表
        return scores.cpu().tolist()
        
