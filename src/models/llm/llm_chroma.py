from tools.retrieval.chroma_document_indexer import ChromaDocumentIndexer
from tools.retrieval.chroma_vector_Indexer import ChromaVectorIndexer
from tools.utils.log_and_catch import log_and_catch


class LlmChroma:
    
    def __init__(self):
       self.chromaVectorIndexer = ChromaVectorIndexer()
       self.chromaDocumentIndexer = ChromaDocumentIndexer()

    @log_and_catch
    def handle_get_segments(self, question,segment_ids):
        self.chromaVectorIndexer.semantic_search(question)
        self.chromaVectorIndexer.vector_get_segments(segment_ids)
        result = self.chromaVectorIndexer.build_raw()
        return {
            "operation": "vector_get_segments",
            "segment_ids": segment_ids,
            "question": question,
            "exists": result is not None and len(result)>0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_segments_by_label(self, question,chapter_ids: list[int], labels: list[str]):
        self.chromaVectorIndexer.semantic_search(question)
        self.chromaVectorIndexer.vector_get_segment_by_label(chapter_ids,labels)
        result = self.chromaVectorIndexer.build_raw()
        return {
            "operation": "vector_get_segment_by_label",
            "chapter_ids": chapter_ids,
            "labels": labels,
            "question": question,
            "exists": result is not None and len(result)>0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_chapter(self, question,chapter_ids: list[int]):
        self.chromaVectorIndexer.semantic_search(question)
        self.chromaVectorIndexer.vector_get_chapter(chapter_ids)
        result = self.chromaVectorIndexer.build_raw()
        return ({
            "operation": "vector_get_chapter",
            "chapter_ids": chapter_ids,
            "question": question,
            "exists": result is not None and len(result)>0,
            "data": result if result else []
        })
    @log_and_catch
    def handle_search_segments_by_keyword(self, keywords,chapter_ids: list[int]):
        self.chromaVectorIndexer.vector_search_segments_by_keyword(chapter_ids)
        self.chromaVectorIndexer.semantic_search(keywords)
        result = self.chromaVectorIndexer.build_raw()
        return {
            "operation": "vector_search_segments_by_keyword",
            "chapter_ids": chapter_ids,
            "keywords": keywords,
            "exists": result is not None and len(result)>0,
            "data": result if result else []
        }

    @log_and_catch
    def handle_get_document(self, question,chapter_ids):
        self.chromaDocumentIndexer.semantic_search(question)
        self.chromaDocumentIndexer.document_get_chapter(chapter_ids)
        result = self.chromaDocumentIndexer.build_raw()
        return {
            "operation": "document_get_chapter",
            "segment_ids": chapter_ids,
            "question": question,
            "exists": result is not None and len(result)>0,
            "data": result if result else []
        }