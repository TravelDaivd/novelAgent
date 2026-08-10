import logging
import os
import re
from ast import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FileUtils:
    
    
    @staticmethod
    def get_txt_files(directory) :
        """获取目录下所有 txt 文件路径"""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

    
    @staticmethod
    def extract_chapter_id(file_name: str) -> int:
        """从文件名中提取章节号"""
        numbers = re.findall(r'\d+', file_name)
        return int(numbers[0]) if numbers else 0
    
    
    @staticmethod
    def get_file_context(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                article_context = file.read()
                return article_context
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return []