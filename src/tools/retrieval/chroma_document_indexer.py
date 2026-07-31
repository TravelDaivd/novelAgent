import logging
from typing import List, Optional, Dict, Any

import chromadb
import numpy as np
from chromadb import EmbeddingFunction

from tools.utils.tool_utils import ToolUtils
from utils.config import VECTOR_DATABASES_DATA_DIR, CHROMA_DOCUMENT_COLLECTION
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ChromaDocumentIndexer(EmbeddingFunction):

    def __init__(self):
        self.toolconfig = ToolUtils()
        self.chroma_db_client = chromadb.PersistentClient(path=VECTOR_DATABASES_DATA_DIR)
        self.model = self.toolconfig.chroma_db_model()
        self.data_coll = self.initialize()
        self._reset()

    def __call__(self, input: list[str]) -> List[List[float]]:
        if not input:
            return []
        embeddings = self.model.encode(
            input,
            show_progress_bar=False,
            normalize_embeddings=True  # 重要：归一化以便cosine相似度
        )
        # 转换为正确的格式
        if isinstance(embeddings, np.ndarray):
            result = embeddings.tolist()
        else:
            result = [list(e) for e in embeddings]

        return result

    def _reset(self):
        """重置所有查询条件"""
        self._question = None
        self._where = {}
        self._result = 10
        self._include = ["documents", "metadatas"]
        return self

    def initialize(self):
        return self.chroma_db_client.get_collection(
            name=CHROMA_DOCUMENT_COLLECTION,
            embedding_function=self  # 我们已经自己生成向量
        )

    """查询问题"""

    def semantic_search(self, question: str):
        self._question = question
        return self

    """查询单个章节id和多个章节ID"""

    def document_get_chapter(self, chapter_ids: list[int]):
        if len(chapter_ids) == 1:
            self._where["chapter_id"] = chapter_ids[0]
        else:
            self._where["chapter_id"] = {"$in": chapter_ids}
            self._result = 15
        return self

    """构建查询结果"""

    def build(self) -> Optional[Dict[str, Any]]:
        if self._question is None:
            logger.error("必须使用 question() 设置查询问题")
            return None
        try:
            logger.info(f"开始执行查询：{self._question[:50]}")
            if self._where:
                logger.info(f"过滤条件：{self._where},参数个数：{len(self._where)}")

            result = self.data_coll.query(
                query_texts=[self._question],
                where=self._where,
                n_results=self._result,
                include=self._include
            )

            # 记录结果统计
            doc_count = len(result['documents'][0]) if result['documents'] else 0
            logger.info(f"找到 {doc_count} 个相关段落")
            # 重置构建器（便于复用）
            self._reset()
            return result

        except Exception as e:
            logger.error(f"查询失败: {e}")
            self._reset()
            return None

    @log_and_catch
    def build_raw(self) -> Optional[Dict[str, Any]]:
        """
        执行查询并返回格式化的结果（便于Agent直接使用）
        """
        build_result_data = self.build()
        if not build_result_data:
            return None

        doc_context_list = build_result_data["documents"][0] if build_result_data['documents'] else []
        formatted = []

        for doc_context in doc_context_list:
            formatted.append({
                "text": doc_context,
            })

        return {
            "context": formatted
        }
     
if __name__ == "__main__":
    chromaVectorIndexer = ChromaDocumentIndexer()
    segment_ids = ['10','11']
    logger.info(f"参数个数：{len(segment_ids)}")
    chromaVectorIndexer.semantic_search("第10、11章发生了什么事情？")
    chromaVectorIndexer.get_chapter_by_document(segment_ids)
    logger.info(chromaVectorIndexer.build_raw())