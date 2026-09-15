import logging
import os
import re
import sqlite3
from typing import List, Dict, Optional

import jieba

from tools.retrieval.retrieval_rrf_coordinator import RetrievalRRFCoordinator
from tools.utils.tool_utils import ToolUtils
from utils.config import *
from utils.log_and_catch import log_and_catch

logger = logging.getLogger(__name__)
class BM25Searcher:
    """BM25 检索引擎，支持 FTS5 全文检索和 RRF 多系统融合"""
    def __init__(self):
        self.db_path = os.path.join(VECTOR_DATABASES_DATA_DIR, BM25_DB_NAME)
        self.load_data_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)
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
    
    @log_and_catch
    def search( self, query: str,top_k: Optional[int] = None,chapter_ids: Optional[List[int]] = None,labels: Optional[List[str]] = None,
        score_threshold: Optional[float] = None) :
        """
        执行 BM25 搜索
        策略：提取关键词 → OR 连接 → FTS5 搜索 → 降级 LIKE
        Args:
            query: 搜索关键词
            top_k: 返回结果数量，默认 15
            chapter_ids: 可选的章节过滤列表
            labels: 可选的标签过滤列表
            score_threshold: 最低分数阈值（BM25 分数越小越相关，设为 None 不过滤）

        Returns:
            搜索结果列表，每项包含：id, text, chapter_id, order, label, raw_score, source
        """
        if not query or not query.strip():
            logger.warning("查询为空，返回空结果")
            return []

        top_k = top_k or self.default_top_k

        try:
            # 1. 提取关键词
            keywords = self.extract_keywords(query)
            if not keywords:
                logger.warning("没有有效关键词")
                return []

            logger.info(f"查询: {query} → 关键词: {keywords}")

            # 2. 构建 FTS 查询（OR 连接）
            fts_query = " OR ".join(keywords)  # 取前5个关键词避免查询过长

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
        
    @log_and_catch
    def search_with_rrf(
        self,
        query: str,
        vector_results: Optional[List[Dict]] = None,
        graph_results: Optional[List[Dict]] = None,
        top_k: int = 20,
        chapter_ids: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        system_weights: Optional[List[float]] = None,
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
            system_weights: [bm25权重, vector权重, graph权重]，默认均为 1.0
            rrf_k: RRF 平滑参数，默认 60

        Returns:
            RRF 融合后的结果列表
        """

        """委托给协调器（保持兼容）"""
        retrievalRRFCoordinator = RetrievalRRFCoordinator(rrf_k=60)
        return retrievalRRFCoordinator.fuse(
            query=query,
            bm25_searcher=self,
            vector_results=vector_results,
            graph_results=graph_results,
            top_k=top_k,
            chapter_ids=chapter_ids,
            labels=labels,
            system_weights=system_weights,
        )

    # ==================== 私有方法 ====================

    def extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词
        Args:
            query: 原始查询字符串
        Returns:
            去重后的关键词列表
        """
        # 移除特殊字符
        special_chars = r'[，。！？；：、·""''（）【】《》……\.,!?;:\'\"()\[\]{}<>]'
        cleaned = re.sub(special_chars, ' ', query).strip()
        words = jieba.cut(cleaned)
        # 过滤停用词和短词
        stopwords = ToolUtils.load_bm25_stopwords()
        keywords = [w for w in words if w not in stopwords and len(w) >= 2]
        # 去重并保持顺序
        return list(dict.fromkeys(keywords))

    def search_fts(
        self,
        fts_query: str,
        limit: int,
        chapter_ids: Optional[List[int]] = None,
        labels: Optional[List[str]] = None
    ) -> List[Dict]:
        """FTS5 全文搜索"""
        try:
            where_clauses = [f"search_text MATCH ?"]
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
                    id, raw_text, chapter_id, order_num, label,
                    bm25({self.fts_table_name},2) as bm25_score
                FROM {self.fts_table_name}
                WHERE {" AND ".join(where_clauses)}
                ORDER BY bm25_score ASC
                LIMIT ?
            """
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()

            return [{
                'id': row['id'],
                'text': row['raw_text'],
                'chapter_id': row['chapter_id'],
                'order': row['order_num'],
                'label': row['label'],
                'raw_score': round(row['bm25_score'], 4) if row['bm25_score'] is not None else 0.0,
                'source': 'fts'
            } for row in rows]

        except Exception as e:
            logger.debug(f"FTS 搜索失败: {e}")
            return []

    def search_like(
        self,
        keywords: List[str],
        limit: int,
        chapter_ids: Optional[List[int]] = None,
        labels: Optional[List[str]] = None
    ) -> List[Dict]:
        """降级方案：LIKE 模糊搜索"""
        try:
            if not keywords:
                return []
            # 构建 LIKE 条件
            conditions = []
            params = []

            for kw in keywords:
                conditions.append("search_text LIKE ?")
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
                    id, raw_text, chapter_id, order_num, label,
                    1.0 as bm25_score
                FROM {self.fts_table_name}
                WHERE {where_sql}
                LIMIT ?
            """
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()
            return [{
                'id': row['id'],
                'text': row['raw_text'],
                'chapter_id': row['chapter_id'],
                'order': row['order_num'],
                'label': row['label'],
                'raw_score': 1.0,
                'source': 'like'
            } for row in rows]

        except Exception as e:
            logger.debug(f"LIKE 搜索失败: {e}")
            return []