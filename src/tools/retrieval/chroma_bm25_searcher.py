import logging
import os
import re
import sqlite3
from typing import List, Dict

import jieba

from tools.retrieval.chroma_rrf_merger import RRFMerger
from tools.utils.tool_utils import ToolUtils
from utils.config import *

logger = logging.getLogger(__name__)
class BM25Searcher:
    
    def __init__(self):
        self.db_path = os.path.join(VECTOR_DATABASES_DATA_DIR,BM25_DB_NAME)
        self.load_data_path = os.path.join(SPLITS_DATA_DIR,CHROMA_VECTRO_SEGMENT_DATA)
        self.fts_table_name = "fts_documents"
        self.max_batch_size = 15
        self.default_top_k = 15
        self.conn = None
        self.init_connection()
        
    def init_connection(self):
        """初始化数据库连接"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
            # 启用 WAL 模式（提高并发性能）
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=-64000")  # 64MB
            # 注册自定义函数
            self.conn.create_function("fts5_score", 1, float(0.0))
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭数据库连接"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

    def search_for_rrf( self,query: str,top_k: int = 50, chapter_ids: List[int] = None, labels: List[str] = None) :
        """
        为 RRF 融合准备 BM25 检索结果
        与 search() 不同的是，这里返回更多结果且不做分数阈值过滤
        Returns:
            带 source='bm25' 标记的结果列表
        """
        results = self.search(
            query=query,
            top_k=top_k,
            chapter_ids=chapter_ids,
            labels=labels,
            score_threshold=None  # 不过滤分数
        )
        # 标记来源
        for doc in results:
            doc['source'] = 'bm25'
        return results

    def search_with_rrf(
            self,query: str,vector_results: List[Dict] = None,graph_results: List[Dict] = None,
            top_k: int = 20,
            chapter_ids: List[int] = None,
            labels: List[str] = None,
            system_weights: List[float] = None,
            rrf_k: int = 60
    ) :
        """
        使用 RRF 融合 BM25 + 其他检索系统
        Args:
            query: 搜索关键词
            vector_results: 向量检索结果（外部传入）
            graph_results: 知识图谱检索结果（外部传入）
            top_k: 最终返回结果数量
            chapter_ids: 章节过滤
            labels: 标签过滤
            system_weights: [bm25权重, vector权重, graph权重]
            rrf_k: RRF 平滑参数

        Returns:
            RRF 融合后的结果
        """
        # 1. BM25 检索（多召回一些）
        bm25_results = self.search_for_rrf(
            query=query,
            top_k=top_k * 2,
            chapter_ids=chapter_ids,
            labels=labels
        )

        # 2. 标记各系统来源
        for doc in bm25_results:
            doc['source'] = 'bm25'

        if vector_results:
            for doc in vector_results:
                doc['source'] = 'vector'

        if graph_results:
            for doc in graph_results:
                doc['source'] = 'graph'

        # 3. 收集所有结果
        result_lists = [bm25_results]
        default_weights = [1.0]

        if vector_results:
            result_lists.append(vector_results)
            default_weights.append(1.0)

        if graph_results:
            result_lists.append(graph_results)
            default_weights.append(1.0)

        # 4. 应用权重
        weights = system_weights if system_weights else default_weights
        if len(weights) != len(result_lists):
            logger.warning(f"权重数量 ({len(weights)}) 与系统数量 ({len(result_lists)}) 不匹配，使用默认权重")
            weights = default_weights

        # 5. RRF 融合
        merger = RRFMerger(k=rrf_k)

        if len(result_lists) == 1:
            # 只有一个系统，按原顺序返回
            return result_lists[0][:top_k]

        fused = merger.fuse_with_weights(result_lists, weights)
        logger.info(f"RRF 融合完成: {len(fused)} 条结果")
        return fused[:top_k]
    
        
        
    def search(self,query: str, top_k: int = None, chapter_ids: List[int] = None,labels: List[str] = None,score_threshold: float = None) :
        """
        执行 BM25 搜索
        策略：提取关键词 → OR 连接 → FTS5 搜索 → 降级 LIKE
        Args:
            query: 搜索关键词
            top_k: 返回结果数量
            chapter_ids: 可选的章节过滤列表
            labels: 可选的标签过滤列表
            score_threshold: 最低分数阈值（0-1）
        Returns:
            搜索结果列表
        """
        if not query or not query.strip():
            logger.warning("查询为空，返回空结果")
            return []
        top_k = top_k or self.default_top_k
        try:
            # 1. 提取关键词
            special_chars = r'[，。！？；：、·""''（）【】《》……\.,!?;:\'\"()\[\]{}<>]'
            cleaned = re.sub(special_chars, ' ', query).strip()
            words = jieba.cut(cleaned)
            jieba_keywords = [w for w in words if w not in ToolUtils.load_bm25_stopwords() and len(w) >= 2]
            keywords = list(dict.fromkeys(jieba_keywords))
            if not keywords:
                logger.warning("没有有效关键词")
                return []
            logger.info(f"查询: {query}... → 关键词: {keywords}")
    
            # 2. 构建查询（OR 连接，取前5个关键词）
            fts_query = " OR ".join(keywords)
    
            # 3. 执行 FTS 搜索
            results = self.search_fts(fts_query, top_k, chapter_ids, labels)
    
            # 4. 如果没结果，降级到 LIKE
            if not results:
                logger.info("FTS 无结果，降级 LIKE")
                results = self.search_like(keywords, top_k, chapter_ids, labels)
    
            # 5. 分数过滤
            if score_threshold is not None:
                results = [r for r in results if r['raw_score'] <= score_threshold]
    
            logger.info(f"返回 {len(results)} 条结果")
            return results[:top_k]
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def search_fts(self, fts_query: str, limit: int,chapter_ids: List[int] = None,labels: List[str] = None):
        """FTS5 搜索"""
        try:
            where_clauses = [f"{self.fts_table_name} MATCH ?"]
            params = [fts_query]
            if chapter_ids:
                placeholders = ','.join(['?'] * len(chapter_ids))
                where_clauses.append(f"chapter_id IN ({placeholders})")
                params.extend(chapter_ids)

            if labels:
                placeholders = ','.join(['?'] * len(labels))
                where_clauses.append(f"label IN ({placeholders})")
                params.extend(labels)

            sql = f"""
                SELECT 
                    id, content, chapter_id, order_num, label,
                    bm25({self.fts_table_name}) as bm25_score
                FROM {self.fts_table_name}
                WHERE {" AND ".join(where_clauses)}
                ORDER BY bm25_score ASC
                LIMIT ?
            """
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()

            return [{
                'id': row['id'],
                'text': row['content'],
                'chapter_id': row['chapter_id'],
                'order': row['order_num'],
                'label': row['label'],
                'raw_score': round(row['bm25_score'], 4) if row['bm25_score'] else 0.0,
                'source': 'fts'
            } for row in rows]

        except Exception as e:
            logger.debug(f"FTS 搜索失败: {e}")
            return []

    def search_like(self, keywords: List[str], limit: int,chapter_ids: List[int] = None, labels: List[str] = None) :
        """降级：LIKE 搜索"""
        try:
            if not keywords:
                return []

            conditions = []
            params = []

            for kw in keywords:
                conditions.append("content LIKE ?")
                params.append(f"%{kw}%")

            where_sql = " OR ".join(conditions)

            if chapter_ids:
                placeholders = ','.join(['?'] * len(chapter_ids))
                where_sql += f" AND chapter_id IN ({placeholders})"
                params.extend(chapter_ids)

            if labels:
                placeholders = ','.join(['?'] * len(labels))
                where_sql += f" AND label IN ({placeholders})"
                params.extend(labels)

            sql = f"""
                SELECT 
                    id, content, chapter_id, order_num, label,
                    1.0 as bm25_score
                FROM {self.fts_table_name}
                WHERE {where_sql}
                LIMIT ?
            """
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()
            return [{
                'id': row['id'],
                'text': row['content'],
                'chapter_id': row['chapter_id'],
                'order': row['order_num'],
                'label': row['label'],
                'raw_score': 1.0,
                'source': 'like'
            } for row in rows]

        except Exception as e:
            logger.debug(f"LIKE 搜索失败: {e}")
            return []

