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

    @staticmethod
    def load_bm25_stopwords() -> set:
        """加载停用词表"""
        return {
            # 1. 助词
            '的', '了', '着', '过', '得', '地', '所',

            # 2. 连词
            '和', '与', '或', '而', '且', '及', '并', '则', '以', '于',

            # 3. 介词
            '在', '把', '被', '从', '向', '给', '对', '为', '将', '靠',

            # 4. 代词
            '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
            '自己', '大家', '各位', '某', '每', '各', '本', '该',

            # 5. 语气词
            '吗', '呢', '吧', '啊', '呀', '哇', '哈', '呵', '嘿', '哦',
            '嗯', '唔', '唉', '哎', '喂', '咦', '哼', '呸',

            # 6. 判断词
            '是', '为', '有', '无', '非',

            # 7. 程度副词
            '很', '太', '非常', '十分', '更', '最', '极', '颇', '相当',
            '比较', '稍微', '略微', '更加', '越发',

            # 8. 范围副词
            '都', '全', '总', '共', '同', '一齐', '一起', '一同',

            # 9. 数量词
            '几', '数',

            # 10. 常用动词（信息量低）
            '要', '会', '能', '可以', '应该',
            '觉得', '认为', '以为',

            # 11. 常用副词
            '已经', '正在', '将要', '刚刚', '才', '就', '还', '也',
            '又', '再', '更', '越', '挺', '好', '坏',
            '多', '少', '高', '低', '长', '短', '快', '慢',

            # 12. 纯语气/填充词
            '吧', '哦', '嗯', '啊', '哈', '嘿', '呵',

            # 13.  疑问词（新增）
            '什么', '怎么', '如何', '怎样', '为什么', '为何',
            '哪', '哪些', '谁', '何时', '何地', '多少',
            '是否', '有没有', '是不是', '吗', '呢',
        }
    
    