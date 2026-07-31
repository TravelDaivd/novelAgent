import logging

from tools.retrieval.graph_search import GraphSearch
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class LlmGraph:
    
    def __init__(self):
        self.graphSearch = GraphSearch()
    @log_and_catch
    def handle_get_segments_ids(self, chapter_ids:list[int]):
        result = self.graphSearch.graph_find_segments_by_chapter(chapter_ids)
        return {
            "operation": "graph_find_segments_by_chapter",
            "chapter_ids": chapter_ids,
            "exists": result is not None and len(result) > 0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_entity_names(self, segment_ids):
        result = self.graphSearch.graph_find_entity_metadata_by_segment(segment_ids)
        return {
            "operation": "graph_find_entity_metadata_by_segment",
            "segment_ids": segment_ids,
            "exists": result is not None and len(result) > 0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_entity_relations(self, entity_names):
        result = self.graphSearch.graph_find_relations_by_entities(entity_names)
        return {
            "operation": "graph_find_relations_by_entities",
            "entity_names": entity_names,
            "exists": result is not None and len(result) > 0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_segment_by_label(self, chapter_ids,label):
        result = self.graphSearch.graph_find_segments_by_label(chapter_ids,label)
        return {
            "operation": "graph_find_segments_by_label",
            "chapter_ids": chapter_ids,
            "label": label,
            "exists": result is not None and len(result) > 0,
            "data": result if result else []
        }