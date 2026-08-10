import logging
import os
from typing import List

import chromadb
import numpy as np
from neo4j import GraphDatabase

from utils.config import *
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ToolUtils:

    

    def chroma_process_batch(self, collection, texts, metadatas, ids, batch_num):
        """处理单个批次"""
        try:
            # 存入数据库
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f" 批次处理完成: {batch_num}，本批{len(texts)}个文本块")
        except Exception as e:
            logger.error(f" 批次{batch_num}处理失败: {e}")
            # 尝试重新处理单个文档
            for i, text in enumerate(texts):
                try:
                    collection.add(
                        documents=[text],
                        metadatas=[metadatas[i]],
                        ids=[ids[i]]
                    )
                except Exception as single_error:
                    logger.error(f"  子文档{ids[i]}也失败: {single_error}")




    def chroma_db_model(self):
        # 1. 加载中文嵌入模型下载到本地了
        model_file_path = os.path.join(VECTOR_DATABASES_DATA_DIR, BGE_SMALL_ZH_NAME)
        logger.info(f"检查模型目录: {model_file_path}")
        if os.path.exists(model_file_path):
            logger.info("模型目录存在")
        logger.info("正在加载模型...")
        try:
            model = SentenceTransformer(model_file_path, device='cpu')  # 效果和速度的绝佳平衡
            logger.info(f"模型加载成功，向量维度: {model.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
        return model
    
    @staticmethod
    def chroma_db_call(model,input: list[str]) -> List[List[float]]:
        if not input:
            return []
        embeddings = model.encode(
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


    @staticmethod
    def get_graph_drive():
        uri = NEO4J_URL
        user = NEO4J_USER
        password = NEO4J_PASSWORD
        graph_drive = GraphDatabase.driver(uri, auth=(user, password))
        return graph_drive