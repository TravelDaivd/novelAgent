import glob
import logging
import re
from typing import List

import chromadb
from chromadb import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
import  json
from src.utils.config import *
from tools.utils.tool_utils import ToolUtils
from tools.utils.log_and_catch import log_and_catch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class  ChromaDocumentStore(EmbeddingFunction) :

    def __init__(self):
        # 1. 加载中文嵌入模型下载到本地了
        self.toolUtils = ToolUtils()
        self.model = self.toolUtils.chroma_db_model()
        
    def name(self): 
        return BGE_SMALL_ZH_NAME
    
    def __call__(self, input: list[str]) -> List[List[float]]:
        return  ToolUtils.chroma_db_call(self.model,input)
    
    
    # 按段落分割，再处理长段落
    @log_and_catch
    def  document_splitter(self,context,chapter_name,unique_chapters,mode ):
        # 超简单分块：按段落+句子
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=256,
            separators=["\n\n", "\n", "。", "！", "？", "……"],
            keep_separator=True
        )

        # 直接分割
        chunks = splitter.split_text(context)
        write_split_file_path = os.path.join(SPLITS_DATA_DIR,CHROMA_DOCUMENT_DATA)
        with open(write_split_file_path, mode, encoding='utf-8') as file:
            for index, chunk in enumerate(chunks,1):
                chunk_handle_text = chunk.replace('\n', '').strip()
                file.write(json.dumps({
                    "id": f"seg_{unique_chapters}_{index}",
                    'text': chunk_handle_text,
                    'chapter_id': unique_chapters,
                    'title_name': chapter_name,
                    'order': index,
                    
                }, ensure_ascii=False) + '\n')



    @log_and_catch
    def handle_chapter_file(self):
        txt_file_path_list = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
        for index, txt_file_path in enumerate(txt_file_path_list):

            mode = 'w'
            if index >0 : mode ='a'
            file_name = os.path.basename(txt_file_path)
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                unique_chapters = re.findall(r'第(\d+)章', file_name)
                chapter_numbers = list(set(int(num) for num in unique_chapters))[0]
                self.document_splitter(file.read(), file_name,chapter_numbers,mode)

    def build_document_database(self):
        chroma_client = chromadb.PersistentClient(path=VECTOR_DATABASES_DATA_DIR)
        try:
            chroma_client.delete_collection(CHROMA_DOCUMENT_COLLECTION)
            logger.info(f"已删除旧集合: {CHROMA_DOCUMENT_COLLECTION}")
        except ValueError:
            logger.info(f"集合不存在，将创建: {CHROMA_DOCUMENT_COLLECTION}")
        except Exception as e:
            logger.warning(f"删除集合时出错: {e}")
        logger.info(f"开始给集合添加数据")
        data_coll = chroma_client.get_or_create_collection(
            name=CHROMA_DOCUMENT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self
        )
        
        # 批量处理参数
        batch_size = 32
        document_batch = []
        metadata_batch = []
        ids_batch = []
        text_split_data_path = os.path.join(SPLITS_DATA_DIR, CHROMA_DOCUMENT_DATA)
        with open(text_split_data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                data = json.loads(line.strip())
                # 添加到批次
                document_batch.append(data.get("text"))
                metadata_batch.append({
                    "title_name": data.get("title_name"),
                    "chapter_id": data.get("chapter_id"),
                    "order": data.get("order"),

                })
                ids_batch.append(data.get("id"))
                # 批次处理
                if len(document_batch) >= batch_size:
                    self.toolUtils.chroma_process_batch(data_coll, document_batch, metadata_batch, ids_batch, line_num)
                    # 清空批次
                    document_batch = []
                    metadata_batch = []
                    ids_batch = []
    
    
    
if __name__ == '__main__':
    chromaDocumentStore =  ChromaDocumentStore()
    chromaDocumentStore.handle_chapter_file()
    chromaDocumentStore.build_document_database()