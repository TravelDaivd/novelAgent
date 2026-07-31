import logging
from typing import Dict, Any
import json

from tools.utils.tool_utils import ToolUtils
from utils.config import *
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphStore:

    def __init__(self):
        self.graph_drive = ToolUtils.get_graph_drive()
        self.setup_constraints()

    def drive_close(self):
        self.graph_drive.close()

    @log_and_catch
    def setup_constraints(self):
        """创建唯一约束（确保数据不会重复）"""
        with self.graph_drive.session() as session:
            try:
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS 
                    FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS 
                    FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE
                """)
                logger.info("数据库约束设置成功")
            except Exception as e:
                logger.error(f"设置约束失败: {e}")
                raise

    
    @staticmethod
    def merge_chapter_with_segments(tx, chapter_data: Dict[str, Any]):
        """事务内使用 MERGE 创建或更新章节、片段和关系"""
        chapter_id = str(chapter_data['chapter_id'])
        title = chapter_data['title']
        segments = chapter_data['segment_array']  # 注意字段名
        query = """
            // ============================================================
            // 第一部分：创建/更新 章节 + 片段（与之前完全一样）
            // ============================================================
            
            // 1. 创建或更新章节节点
            MERGE (c:Chapter {chapter_id: $chapter_id})
            ON CREATE SET 
                c.title = $title,
                c.created_at = datetime()
            ON MATCH SET 
                c.title = $title,
                c.updated_at = datetime()

            // 2. 处理片段节点
            WITH c
            UNWIND $segments AS seg_data
            MERGE (s:Segment {segment_id: seg_data.segment_id})
            ON CREATE SET 
                s.chapter_id = seg_data.chapter_id,
                s.segment_id = seg_data.segment_id,
                s.label = seg_data.label,
                s.order = seg_data.order,
                s.created_at = datetime()
            ON MATCH SET 
                s.chapter_id = seg_data.chapter_id,
                s.segment_id = seg_data.segment_id,
                s.label = seg_data.label,
                s.order = seg_data.order,
                s.updated_at = datetime()

            // 3. 创建或更新章节到片段的关系
            MERGE (c)-[:HAS_SEGMENT]->(s)
            
            // ============================================================
            // 第二部分：创建/更新 实体（追加）
            // 如果 seg_data.entities 不存在或为空，这段不执行任何操作
            // ============================================================
            
            WITH c,s, seg_data
            WHERE seg_data.entities IS NOT NULL
            UNWIND seg_data.entities AS entity_data
            MERGE (e:Entity {name: entity_data.name})
            ON CREATE SET
                e.confidence = entity_data.confidence
            ON MATCH SET
                e.confidence = entity_data.confidence
            
            // 片段 → 实体（片段包含实体）
            MERGE (s)-[:HAS_ENTITY]->(e)
            
            // ============================================================
            // 第三部分：创建/更新 实体关系（追加）
            // 如果 seg_data.relations 不存在或为空，这段不执行任何操作
            // ============================================================
            
            WITH c,s, seg_data
            WHERE seg_data.relations IS NOT NULL
            UNWIND seg_data.relations AS rel_data
            // 查找或创建关系节点
            MERGE (r:EntityRelation {
                relation: rel_data.relation,
                entity_one: rel_data.entity_one,
                entity_two: rel_data.entity_two
            })
            ON CREATE SET r.confidence = rel_data.confidence
            ON MATCH SET r.confidence = rel_data.confidence
            // 连接关系节点到两个实体
            WITH r, rel_data,c,s
            MATCH (e1:Entity {name: rel_data.entity_one})
            MATCH (e2:Entity {name: rel_data.entity_two})
            
            MERGE (e1)-[:FROM_ENTITY]->(r)
            MERGE (r)-[:TO_ENTITY]->(e2)
            
            // 关系 → 来源片段
            MERGE (r)-[:BASED_ON]->(s)
            
            
            // 4. 创建片段之间的顺序关系（按 order 排序）
            WITH c, collect(s) AS seg_list
            UNWIND seg_list AS seg
            WITH c, seg
            ORDER BY seg.order
            WITH c, collect(seg) AS sorted_list

            // 5. 创建 NEXT 关系
            WITH c, sorted_list
            UNWIND range(0, size(sorted_list)-2) AS i
            WITH c, sorted_list,sorted_list[i] AS current, sorted_list[i+1] AS next
            MERGE (current)-[:NEXT]->(next)

            
        """

        result = tx.run(
            query,
            chapter_id=chapter_id,
            title=title,
            segments=segments
        )
        return result.single()

    @log_and_catch
    def delete_chapter(self, chapter_id: str):
        """删除整个章节及其所有片段和关系（级联删除）"""
        with self.graph_drive.session() as session:
            query = """
                MATCH (c:Chapter {chapter_id: $chapter_id})
                OPTIONAL MATCH (c)-[:HAS_SEGMENT]->(s:Segment)
                OPTIONAL MATCH (s)-[r:NEXT]-()
                DELETE r, s, c
                RETURN count(s) AS deleted_segments
            """
            result = session.run(query, chapter_id=chapter_id)
            return result.single()

    def clear_all(self):
        """清空所有数据（谨慎使用）"""
        with self.graph_drive.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("所有数据已清空")

    @log_and_catch
    def up_chapter_context(self,context):
        try:
            with self.graph_drive.session() as session :
                result = session.execute_write(
                    self.merge_chapter_with_segments,
                    context
                )
            if result:
                logger.info(f"导入章节编号: {result['chapter_id']}, 片段数: {result['segment_count']}")
        
        except Exception as e:
            logger.error(f"数据入neo4j处理失败: {e}")



    def handle_data(self):
        """处理数据的主方法"""
        chapter_segment_data_path = os.path.join(SPLITS_DATA_DIR, GRAPH_SEGMENT_DATA)

        with open(chapter_segment_data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file,1):
                try:
                    # 解析JSON
                    data_array = json.loads(line.strip())
                    logger.info(f"处理第 {line_num} 行数据: {data_array.get('chapter_id', 'unknown')}")
                    # ✅ 修复：使用 execute_write 传递静态方法
                    with self.graph_drive.session() as session:
                        result = session.execute_write(
                            self.merge_chapter_with_segments,
                            data_array
                        )
                        if result:
                            logger.info(f"导入章节编号: {result['chapter_id']}, 片段数: {result['segment_count']}")

                except json.JSONDecodeError as e:
                    logger.error(f"第 {line_num + 1} 行 JSON 解析失败: {e}")
                except Exception as e:
                    logger.error(f"第 {line_num + 1} 行数据处理失败: {e}")
                    # 可选择继续或中断
                    # raise


if __name__ == "__main__":
    GraphStore().clear_all()
    GraphStore().handle_data()