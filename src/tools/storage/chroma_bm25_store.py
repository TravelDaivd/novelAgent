import json
import logging
import os

import sqlite3
from typing import List, Dict

from utils.config import *

logger = logging.getLogger(__name__)


class BM25Store:

    def __init__(self, force_reload=False):
        self.db_path = os.path.join(VECTOR_DATABASES_DATA_DIR, BM25_DB_NAME)
        self.load_data_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)
        self.fts_table_name = "fts_documents"
        self.max_batch_size = 15
        self.conn = None
        self.init_connection()
        self.init_fts_table()

        if self.is_empty():
            self.load_data_to_fts()
        elif force_reload:
            self.load_data_to_fts()
        else:
            logger.info(f"数据库已有数据，跳过加载")

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

    def init_fts_table(self):
        """创建 FTS5 虚拟表"""
        try:
            create_sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table_name}
                USING fts5(
                    id UNINDEXED,
                    content,
                    chapter_id ,
                    order_num ,
                    label ,
                    tokenize='unicode61'
                )
            """
            self.conn.execute(create_sql)
            self.conn.commit()
            logger.debug(f"FTS5 表创建/检查完成: {self.fts_table_name}")
        except Exception as e:
            logger.error(f"FTS5 表创建失败: {e}")
            raise

    def is_empty(self) -> bool:
        """检查索引是否为空"""
        try:
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {self.fts_table_name}"
            ).fetchone()[0]
            return count == 0
        except Exception:
            return True

    def load_data_to_fts(self):
        """加载数据到 FTS5"""
        # 检查是否已有数据
        if not os.path.exists(self.load_data_path):
            logger.error(f"JSONL 文件不存在: {self.load_data_path}")
            raise FileNotFoundError(f"JSONL 文件不存在: {self.load_data_path}")
        logger.info(f"开始从 JSONL 加载数据: {self.load_data_path}")
        total_count = 0
        batch = []

        with open(self.load_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    row = (
                        data.get('id', ''),
                        data.get('text', ''),
                        data.get('label', ''),
                        data.get('chapter_id', 0),
                        data.get('order', 0)
                    )
                    batch.append(row)
                    total_count += 1
                    if len(batch) >= self.max_batch_size:
                        self.insert_batch(batch)
                        logger.info(f"已加载 {total_count} 条数据")
                        batch = []
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败，跳过: {e}")
                    continue

        # 插入最后一批
        if batch:
            self.insert_batch(batch)

        logger.info(f" 数据加载完成")
        return total_count

    def insert_batch(self, batch: List[tuple]):
        """批量插入数据"""
        try:
            # 1. 提取所有 ID
            ids = [row[0] for row in batch]
            # 2. 删除已存在的记录
            self.delete_document(ids)

            insert_sql = f"""
                INSERT INTO {self.fts_table_name}
                (id, content,label, chapter_id, order_num)
                VALUES (?, ?, ?, ?, ?)
            """
            self.conn.executemany(insert_sql, batch)
            self.conn.commit()
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            self.conn.rollback()
            raise

    def batch_add_documents(self, docs: List[Dict]) -> int:
        """
        批量添加文档
        """
        if not docs:
            return 0
        try:
            batch = []
            for doc in docs:
                batch.append((
                    doc.get('id', ''),
                    doc.get('text', ''),
                    doc.get('label', ''),
                    doc.get('chapter_id', 0),
                    doc.get('order', 0)
                ))

            insert_sql = f"""
                INSERT OR REPLACE INTO {self.fts_table_name}
                (id, content, chapter_id, order_num, label)
                VALUES (?, ?, ?, ?, ?)
            """
            self.conn.executemany(insert_sql, batch)
            self.conn.commit()
            count = len(batch)
            logger.info(f"批量添加完成: {count} 个文档")
            return count

        except Exception as e:
            logger.error(f" 批量添加失败: {e}")
            self.conn.rollback()
            return 0

    def delete_document(self, doc_ids: list[str]) -> bool:
        """
        删除文档
        """
        try:
            placeholders = ','.join(['?'] * len(doc_ids))
            sql = f"DELETE FROM {self.fts_table_name} WHERE id IN ({placeholders}) "
            self.conn.execute(sql, doc_ids)
            self.conn.commit()
            logger.info(f"文档已删除: {doc_ids}")
            return True
        except Exception as e:
            logger.error(f"文档删除失败: {e}")
            self.conn.rollback()
            return False

    def optimize(self):
        """
        优化 FTS5 索引
        建议在大量数据变更后调用
        """
        try:
            self.conn.execute(
                f"INSERT INTO {self.fts_table_name}({self.fts_table_name}) VALUES('optimize')")
            self.conn.commit()
            logger.info("FTS5 索引优化完成")
        except Exception as e:
            logger.error(f"索引优化失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭数据库连接"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

