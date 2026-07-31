
import logging

from tools.utils.tool_utils import ToolUtils
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
    1、把文本分类中的标签查询出来
    2、第10、11章发生了什么事情？主要描述谁的？有没有我喜欢的打斗环节
    3、他们之间是什么关系?为什么张元烛救她们
    4、对方是怎么死的？
    5、被救出她们后，张元烛离开后发生了什么事情？
    6、所有的事情是发生在哪个地方？
    
"""

class GraphSearch:
    def __init__(self):

        self.graph_drive = ToolUtils.get_graph_drive()

    def drive_close(self):
        try:
            self.graph_drive.close()
        except Exception as e:
            self.try_except_msg(f"数库关闭失败:{e}")
    
    def graph_find_segments_by_chapter(self, chapter_ids) -> list[str]:
        """
         根据章节ID集合，获取片段ID集合
        :param chapter_ids: 章节ID集合
        :return:  片段ID集合
        """
        with self.graph_drive.session() as session:
            query = """
                match (c:Chapter)-[:HAS_SEGMENT]->(s:Segment)
                where toInteger(c.chapter_id) in $chapter_id
                return s.segment_id AS segment_id,
                       s.chapter_id AS chapter_id
                order by s.order asc
             """
            result = session.run(query, {"chapter_id": chapter_ids})

            return result.data()

    def graph_find_entity_metadata_by_segment(self,segment_ids)->list[str]:
        """
         根据片段ID集合，获取实体人名列表
        :param segment_ids: 片段ID集合
        :return: 实体人物元数据
        """
        with self.graph_drive.session() as session:
            query = """
                match (s:Segment)-[:HAS_ENTITY]->(e:Entity)
                where s.segment_id in $segment_ids
                return distinct e.name as entityName
            """
            result =  session.run(query,{"segment_ids":segment_ids})
            entity_names = [record["entityName"] for record in result]

        with self.graph_drive.session() as session:
            query = """
                match (s:Segment)-[:HAS_ENTITY]->(e:Entity)
                where e.name in $entity_names
                return s.segment_id AS segmentId
                order by s.chapter_id asc

            """
            result = session.run(query, {"entity_names": entity_names})
            segment_ids_from_entities = [record["segmentId"] for record in result]
        
        with self.graph_drive.session() as session:
            query = """
                    match (c:Chapter)-[:HAS_SEGMENT]->(s:Segment)
                    where s.segment_id in $segment_ids
                    return c.chapter_id AS chapterId
                 """
            result = session.run(query, {"segment_ids": segment_ids})
            chapter_ids = [record["chapterId"] for record in result]

        return {
            "entities": entity_names,
            "segment_ids": list(set(segment_ids + segment_ids_from_entities)),
            "chapter_ids": chapter_ids
        }
        
    def graph_find_relations_by_entities(self,entity_names):
        """
         根据实体人物列表，获得实体人物之间的关系
        :param entity_names: 
        :return:  实体人物之间关系列表
        """
        with self.graph_drive.session() as  session:
            query ="""
                MATCH (e1:Entity)-[:FROM_ENTITY]->(r:EntityRelation)-[:TO_ENTITY]->(e2:Entity)
                WHERE e1.name IN $entity_names
                  AND e2.name IN $entity_names
                  AND e1.name < e2.name
                OPTIONAL MATCH (r)-[:BASED_ON]->(s:Segment)
                RETURN e1.name AS entity_one,
                       e2.name AS entity_two,
                       r.relation AS relation
            
            """
            result = session.run(query, {"entity_names": entity_names})
            return result.data()

   

    
    
      
        
    def get_segment_context(self, segment_id: str, prev: int = 3, next: int = 3):
        """
        获取某个片段的前 N 个和后 N 个片段（沿 NEXT 关系）
        :param segment_id:  片段ID
        :param prev: 向前取的数量
        :param next: 向后取的数量
        :return: 上下文信息
        """
        logger.info(f"get_segment_context -> segment_id: {segment_id}, prev: {prev}, next: {next}")
        with self.graph_drive.session() as session:
            query = """
                // 1. 找到当前片段和它的章节
                 MATCH (current:Segment {segment_id: $segment_id})
               // 2. 获取同一章节的所有片段（通过 chapter_id 属性）
                MATCH (s:Segment {chapter_id: current.chapter_id})

                // 3. 计算每个片段相对于当前片段的位置
                WITH current, s,
                     s.order AS seg_order,
                     current.order AS current_order
                ORDER BY s.order ASC

                // 4. 收集所有片段信息
                WITH current, 
                     collect({
                         segment_id: s.segment_id,
                         distance: s.order - current.order
                     }) AS all_segments

                // 5. 提取前后片段
                RETURN 
                    [seg IN all_segments 
                     WHERE seg.distance < 0 AND seg.distance >= -$prev 
                     | seg {.segment_id}
                    ] AS prev_segments,

                    [seg IN all_segments WHERE seg.distance = 0 
                     | seg {.segment_id}
                    ][0] AS current_segment,

                    [seg IN all_segments 
                     WHERE seg.distance > 0 AND seg.distance <= $next 
                     | seg {.segment_id}
                    ] AS next_segments

            """
            record = session.run(query, {"segment_id": segment_id, "prev": prev, "next": next}).single()

            if record:
                return {
                    "prev": record["prev_segments"],
                    "current": record["current_segment"],
                    "next": record["next_segments"]
                }
            else:
                return {"prev": [], "current": {}, "next": []}

    def search_by_label(self, label: str) -> list[str]:
        """
        根据标签查询片段ID
        :param label:
        :return:
        """
        logger.info(f"search_by_label -> label :{label}")
        with self.graph_drive.session() as session:
            query = """
                match (s:Segment)
                where s.label = $label
                return s.segment_id AS segment_id
                       s.chapter_id AS chapter_id
                       s.label AS label

            """
            reslut = session.run(query, label=label)
            return [record["segmentId"] for record in reslut]

    def search_by_chapter_label(self, chapter_id: int, label: str) -> list[str]:
        """
        根据章节ID指定查询标签片段
        :param label:
        :return:
        """
        logger.info(f"search_by_chapter_label -> chapter_id:{chapter_id}, label :{label}")
        with self.graph_drive.session() as session:
            query = """
                match (c:Chapter)-[:HAS_SEGMENT]->(s:Segment)
                where toInteger(c.chapter_id) = $chapter_id and s.label = $label
                return s.segment_id as segmentId

            """
            reslut = session.run(query, chapter_id=chapter_id, label=label, )
            return [record["segmentId"] for record in reslut]

    def get_segments_by_order_range(self, chapter_id: int, start_order: int, end_order: int) -> list[str]:
        """
        按顺序/范围查询
        :param chapter_id: 章节id
        :param start_order: 起始位置
        :param end_order: 结束位置
        :return:
        """
        logger.info(
            f"get_segments_by_order_range ->chapter_id:{chapter_id},start_order:{start_order}, end_order:{end_order}")
        if start_order > end_order:
            old_order = start_order
            start_order = end_order
            end_order = old_order
        with self.graph_drive.session() as session:
            query = """
                match (c:Chapter)-[:HAS_SEGMENT]->(s:Segment)
                where toInteger(c.chapter_id) = $chapter_id and s.order >= $start_order AND s.order <= $end_order
                return s.segment_id as segmentId
                order by s.order
            """
            result = session.run(query, chapter_id=chapter_id, start_order=start_order, end_order=end_order)
            return [record["segmentId"] for record in result]

    def graph_find_segments_by_label(self, chapter_ids: list[int], label: list[str]):
        """
        根据章节Id列表和标签，获取片段ID列表
        :param chapter_ids: 章节Id列表
        :param label: 标签
        :return: 
        """
        with self.graph_drive.session() as session:
            query = """
                 match (c:Chapter)-[:HAS_SEGMENT]->(s:Segment)
                 where toInteger(c.chapter_id) in $chapter_ids
                     and s.label in $label
                 return s.segment_id AS segment_id,
                        s.chapter_id AS chapter_id
                 order by s.chapter_id asc
             """
            result = session.run(query, {"chapter_ids": chapter_ids, "label": label})
            return result.data()

if __name__ == "__main__":
    textQuery = GraphSearch()
    sel_data = textQuery.graph_find_segments_by_chapter([36])
    logger.info(f"sel data :{sel_data}")
    sel_data = textQuery.graph_find_entity_metadata_by_segment(["seg_36_1", "seg_10_12","seg_36_3","seg_10_19"])
    logger.info(f"sel data :{sel_data}")
    sel_data = textQuery.graph_find_segments_by_label([10, 11], ["战斗","对话","内心"])
    logger.info(f"sel data :{sel_data}")
    
   
    
   
    """
    for segment_id in sel_data:
        sel_data = textQuery.get_segment_details(segment_id)
        logger.info(f"sel data :{sel_data}")
    sel_data = textQuery.get_segment_context('seg_14_7')
    logger.info(f"sel data :{sel_data}")
    sel_data = textQuery.search_by_label("探索")
    logger.info(f"sel data :{len(sel_data)},{sel_data}")
    sel_data = textQuery.search_by_chapter_label(14, "精彩")
    logger.info(f"sel data :{len(sel_data)},{sel_data}")
    sel_data = textQuery.get_segments_by_order_range(14, 12, 6)
    logger.info(f"sel data :{len(sel_data)},{sel_data}")
    """