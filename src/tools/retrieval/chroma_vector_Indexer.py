import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb import EmbeddingFunction

import numpy as np

from tools.utils.tool_utils import ToolUtils
from utils.config import *
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ChromaVectorIndexer(EmbeddingFunction):

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
        self._result = self.data_coll.count()
        self._include = ["documents","metadatas"]
        return self
    
    
    def initialize(self):
        return self.chroma_db_client.get_collection(
            name=CHROMA_SEGMENT_COLLECTION,
            embedding_function=self  # 我们已经自己生成向量
        )

    """查询问题"""

    def semantic_search(self, question: str):
        self._question = question
        return self

    """查询单个章节id和多个章节ID"""
    
    def vector_get_chapter(self, chapter_ids: list[int]):
        if len(chapter_ids) == 1:
            self._where["chapter_id"] = chapter_ids[0]
        else:
            self._where["chapter_id"] = {"$in": chapter_ids}
        return self

    def vector_get_segment_by_label(self, chapter_ids: list[int], labels: list[str]):
        self._where={
            "$and":[
                {"chapter_id":{"$in": chapter_ids}},
                {"label": {"$in": labels}}
            ]
        }
        return self

    def vector_get_segments(self, semgment_ids: list[str]):
        """
        从Neo4j查询到的片段ID集合去chromadb向量数据库
        查对应的片段内容
        :param semgment_ids: 片段ID集合
        :return:
        """
        if len(semgment_ids) == 1:
            self._where["segment_id"] = semgment_ids[0]
        else:
            self._where["segment_id"] = {"$in": semgment_ids}
        return self

    def vector_search_segments_by_keyword(self,chapter_ids: list[int]):
        self._where = {}
        if chapter_ids:
            self._where["chapter_id"] = {"$in": chapter_ids}
        return self
        
        
        
        
        
        

    """构建查询结果"""

    def build(self) -> Optional[Dict[str, Any]]:
        if self._question is None:
            logger.error("必须使用 question() 设置查询问题")
            return None
        try:
            logger.info(f"开始执行查询{self._question[:50]}")
            if self._where:
                logger.info(f"过滤条件：{self._where},参数个数：{len(self._where)}")
               
                
            result = self.data_coll.query(
                query_texts=[self._question],
                where=self._where ,
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
        metadatas_list = build_result_data["metadatas"][0] if build_result_data['metadatas'] else []
        formatted = []

        for doc_context,metadata in zip(doc_context_list,metadatas_list):
            formatted.append({
                "text": doc_context,
                "chapter_id": metadata.get("chapter_id", "unknown"),
                "label": metadata.get("label", "unknown")
            })

        return {
            "context": formatted
        }

if __name__ == "__main__":
    chromaVectorIndexer = ChromaVectorIndexer()
  #  segment_ids = ['seg_11_1', 'seg_10_2', 'seg_11_4', 'seg_10_4', 'seg_11_5', 'seg_10_5', 'seg_11_6', 'seg_10_8', 'seg_11_9', 'seg_10_9', 'seg_11_10', 'seg_11_11', 'seg_11_12', 'seg_10_12', 'seg_11_16', 'seg_10_16', 'seg_10_17', 'seg_11_18', 'seg_10_18', 'seg_10_19']
   # logger.info(f"参数个数：{len(segment_ids)}")
    chromaVectorIndexer.semantic_search("对方是怎么死的？..")
    chromaVectorIndexer.vector_get_segment_by_label([10, 11],['战斗'])
    logger.info(chromaVectorIndexer.build_raw())