import json
import logging
import os
from typing import List

import chromadb
import numpy as np
from chromadb import EmbeddingFunction

from tools.utils.tool_utils import ToolUtils
from utils.config import *
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore(EmbeddingFunction):

    def __init__(self):
        self.toolUtils = ToolUtils()
        self.model = self.toolUtils.chroma_db_model()
        chromadb_client = chromadb.PersistentClient(path=VECTOR_DATABASES_DATA_DIR)
        self.data_coll = chromadb_client.get_or_create_collection(
            name=CHROMA_SEGMENT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self
        )
    def name(self) -> str:
        return BGE_SMALL_ZH_NAME
    
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


    def chapter_vector_database(self,chapter_chroma_data_list):
        batch_size = 3
        document_batch = []
        metadata_batch = []
        ids_batch = []
        for idx, data in enumerate(chapter_chroma_data_list,1):
            self.chroma_data(data, document_batch, metadata_batch, ids_batch)
            if len(document_batch) >= batch_size:
                self.toolUtils.chroma_process_batch(self.data_coll, document_batch, metadata_batch, ids_batch, idx)
                # 清空批次
                document_batch = []
                metadata_batch = []
                ids_batch = []


    @log_and_catch
    def build_vector_database(self):
        # 初始化ChromaDB客户端  把数据持久化磁盘
        # 批量处理参数
        batch_size = 32
        document_batch = []
        metadata_batch = []
        ids_batch = []
        text_split_data_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)
        with open(text_split_data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                data = json.loads(line.strip())
                # 添加到批次
                self.chroma_data(data,document_batch, metadata_batch, ids_batch)
                # 批次处理
                if len(document_batch) >= batch_size:
                    self.toolUtils.chroma_process_batch(self.data_coll,document_batch, metadata_batch, ids_batch,line_num )
                    # 清空批次
                    document_batch = []
                    metadata_batch = []
                    ids_batch = []

    def chroma_data(self,data,document_batch,metadata_batch,ids_batch):
        document_batch.append(data.get("text"))
        metadata_batch.append({
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "chapter_id": data.get("chapter_id"),
            "segment_id": data.get("segment_id"),
            "order": data.get("order"),
            # 只存实体名称列表（用于过滤）
            "entities": ",".join([e['name'] for e in data.get("entities", [])]),
            "relations": ",".join([r['relation'] for r in data.get("relations", [])])

        })
        ids_batch.append(data.get("id"))

# 使用方式
if __name__ == "__main__":
    builder = ChromaVectorStore()
    builder.build_vector_database()