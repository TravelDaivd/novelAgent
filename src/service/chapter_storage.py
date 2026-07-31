from tools.storage.chroma_vector_store import ChromaVectorStore
from tools.storage.graph_store import GraphStore


class ChapterStorage:
    
    def __init__(self):
        self.graphStore = GraphStore()
        self.chromaVectorStore = ChromaVectorStore()
       
    def chroma_graph_database(self,chroma_vector_data_list,graph_chapter_segment_list):
        self.graphStore.up_chapter_context(graph_chapter_segment_list)
        self.chromaVectorStore.chapter_vector_database(chroma_vector_data_list)
    