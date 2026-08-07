import glob
import json
import logging
import os

from models.inference_pipeline import InferencePipeline
from models.util.models_utils import  ModelsUtils
from service.chapter_storage import ChapterStorage
from tools.utils.log_and_catch import log_and_catch
from utils.config import *
from utils.file_utils import FileUtils

logger = logging.getLogger(__name__)
class ChapterProcessor:
    
    def __init__(self):
        self.txt_file_path_list = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
        self.inferencePipeline = InferencePipeline()
        self.chapterStorage = ChapterStorage()

    @log_and_catch
    def process_text_context(self,context,title):
        graph_chapter_segment_list = []
        chroma_vector_data_list = []
        sentence_list = ModelsUtils.handle_context(context)
        number = self.inferencePipeline.json_file_data_num()
        unique_chapters = number + 1
        chroma_vector_data_list,graph_segment_list =self.inferencePipeline.extract_segment(sentence_list, unique_chapters, chroma_vector_data_list)
        graph_chapter_segment_list.append({
            'chapter_id': unique_chapters,
            'title': f"{title}",
            'segment_array': graph_segment_list
        })
        self.chapterStorage.chroma_graph_database(chroma_vector_data_list, graph_chapter_segment_list[0])
        self.inferencePipeline.write_json_lines(chroma_vector_data_list, graph_chapter_segment_list,'a')
        return chroma_vector_data_list,len(sentence_list),unique_chapters
    
    
    
    
    
    @log_and_catch
    def process_directory(self):
        graph_chapter_segment_list = []
        chroma_vector_data_list = []
        for index, file_path in enumerate(self.txt_file_path_list):
            file_name = os.path.basename(file_path)
            unique_chapters = FileUtils.extract_chapter_id(file_name)
            sentence_list = ModelsUtils.split_into_sentences(file_path)
            chroma_vector_data_list,graph_segment_list = self.inferencePipeline.extract_segment(sentence_list,unique_chapters,chroma_vector_data_list)
            graph_chapter_segment_list.append({
                'chapter_id': unique_chapters,
                'title': file_name,
                'segment_array': graph_segment_list
            })
        self.inferencePipeline.write_json_lines(chroma_vector_data_list, graph_chapter_segment_list,'w')
        

if __name__ == "__main__":
    ChapterProcessor().process_directory()
