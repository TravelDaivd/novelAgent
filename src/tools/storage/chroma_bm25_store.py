import json
import logging
import os

import sqlite3
import subprocess
from typing import List, Dict

from utils.config import *
from utils.config import LIBSIMPLE_FILE_PATH, LIBSIMPLE_DIR
from utils.log_and_catch import log_and_catch

logger = logging.getLogger(__name__)


class BM25Store:

    def __init__(self, force_reload=False):
        self.db_path = os.path.join(VECTOR_DATABASES_DATA_DIR, BM25_DB_NAME)
        self.libsimple_file_path = os.path.join(LIBSIMPLE_DIR,LIBSIMPLE_FILE_PATH)
        self.load_data_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)
        self.fts_table_name = "fts_documents"
        self.max_batch_size = 15
        self.use_jieba = False
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
            
            # 尝试加载 simple 扩展
            self.load_simple_extension()
            
            # 启用 WAL 模式（提高并发性能）
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=-64000")  # 64MB
            # 注册自定义函数
            self.conn.create_function("fts5_score", 1, float(0.0))
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def allow_dylib_load(self):
        """移除 macOS 隔离属性"""
        try:
            subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", self.libsimple_file_path],
                check=False,
                capture_output=True
            )
            return True
        except Exception as e:
            logger.warning(f"移除隔离属性失败: {e}")
            return False

    @log_and_catch
    def load_simple_extension(self):
        """加载 simple 扩展（带降级）"""
        try:
            # 检查扩展文件是否存在
            if not os.path.exists(self.libsimple_file_path):
                logger.warning(f"simple 扩展文件不存在: {self.libsimple_file_path}")
                self.use_jieba = False
                return
            #移除 macOS 隔离属性
            self.allow_dylib_load()
            original_dir = os.getcwd()
            lib_dir = os.path.dirname(self.libsimple_file_path)
            logger.info(f"切换工作目录: {original_dir} → {lib_dir}")
            os.chdir(lib_dir)
            try:
                # 启用扩展加载
                self.conn.enable_load_extension(True)
                # 加载扩展
                self.conn.load_extension(self.libsimple_file_path)
    
                #验证扩展是否可用
                cursor = self.conn.cursor()
                cursor.execute("SELECT jieba_query('测试')")
                cursor.fetchone()
    
                self.use_jieba = True
                logger.info(f"✅ simple 扩展加载成功: {self.libsimple_file_path}")
            finally:
                # 切换回原目录
                os.chdir(original_dir)
                logger.info(f"恢复工作目录: {original_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ simple 扩展加载失败: {e}，将使用 unicode61 降级方案")
            self.use_jieba = False
    
    
    
    def init_fts_table(self):
        """创建 FTS5 虚拟表"""
        try:
            if self.use_jieba:
                tokenizer = "simple"
                logger.info("使用 jieba 分词器")
            else:
                tokenizer = "unicode61"
                logger.warning("使用 unicode61 分词器（降级方案）")
            
            create_sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table_name}
                USING fts5(
                    id UNINDEXED,
                    raw_text,
                    search_text,
                    label,
                    chapter_id ,
                    order_num, 
                    tokenize='{tokenizer}'
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
                    label = f"标签：{data.get('label', '')}"
                    chapter = f"第{data.get('chapter_id', '')}章"
                    if self.use_jieba: 
                        search_text = f"{data.get('text', '')} {chapter} {label} " 
                    else:
                        search_text = f"{data.get('text', '')} {chapter} {data.get('chapter_id', '')} {label} {data.get('label', '')}"
                    row = (
                        data.get('id', ''),
                        data.get('text', ''),
                        search_text,
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
                (id, raw_text,search_text,label, chapter_id, order_num)
                VALUES (?, ?, ?, ?, ?,?)
            """
            self.conn.executemany(insert_sql, batch)
            self.conn.commit()
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            self.conn.rollback()
            raise

    def batch_add_documents(self, doc_list: List[Dict]) -> int:
        """
        批量添加文档
        """
        if not doc_list:
            return 0
        try:
            batch = []
            for doc in doc_list:
                label = f"标签：{doc.get('label', '')}"
                chapter = f"第{doc.get('chapter_id', '')}章"
                if self.use_jieba:
                    search_text = f"{doc.get('text', '')} {chapter} {label} "
                else:
                    search_text = f"{doc.get('text', '')} {chapter} {doc.get('chapter_id', '')} {label} {doc.get('label', '')}"
                batch.append((
                    doc.get('id', ''),
                    doc.get('text', ''),
                    search_text,
                    doc.get('label', ''),
                    doc.get('chapter_id', 0),
                    doc.get('order', 0)
                ))

            insert_sql = f"""
                INSERT OR REPLACE INTO {self.fts_table_name}
                (id, raw_text,search_text,label, chapter_id, order_num)
                VALUES (?, ?, ?, ?, ?, ?)
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

