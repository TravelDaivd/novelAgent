import os

from dotenv import load_dotenv

# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DATA_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')




# 共享模型存放文件夹
SHARED_DATA_DIR = os.path.join(MODELS_DATA_DIR, 'shared')
# 基本模型
CHINESE_MAC_BERT_BASE_NAME = 'chinese_macBert_base'
BGE_SMALL_ZH_NAME = 'bge_small_zh_v1.5'

# 小说-人物识别模型
PERSON_RECOGNITION_CHINESE_MAC_BERT_NAME = 'person_recognition_macBert'
# 小说-关系抽取模型
RELATION_RECOGNITION_CHINESE_MAC_BERT_NAME = 'relation_recognition_macBert'
# 小说-文本分类模型
TEXT_RECOGNITION_CHINESE_MAC_BERT_NAME = 'text_recognition_macBert'


# 人工训练数据
MANUAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'manual')
PERSON_TRAIN_NAME='person_train.json'
RELATION_TRAIN_NAME='relation_train.json'
TEXT_TRAIN_NAME='text_train.json'

# 训练模型验证集数据
VALIDATION_DATA_DIR = os.path.join(MANUAL_DATA_DIR, 'validation')
PERSON_VALIDATION_NAME='person_validation.json'
RELATION_VALIDATION_NAME='relation_validation.json'
TEXT_VALIDATION_NAME='text_validation.json'



# 自动标注数据
AUTO_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'auto')
AUTO_PERSON_MARGINALIA_NAME='auto_person_marginalia.json'
AUTO_RELATION_MARGINALIA_NAME='auto_relation_marginalia.json'
AUTO_TEXT_MARGINALIA_NAME='auto_text_marginalia.json'

# 评估模型
APPRAISE_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'appraise')
APPRAISE_PERSON_DATA_NAME='appraise_person_data.json'
APPRAISE_RELATION_DATA_NAME='appraise_relation_data.json'
APPRAISE_TEXT_DATA_NAME='appraise_text_data.json'



#========Chroma/Graph系统配置信息=============
SPLITS_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'splits')
CHROMA_DOCUMENT_DATA = 'chroma_document_data.jsonl'  # 向量数据库文档 数据文件名称
CHROMA_VECTRO_SEGMENT_DATA = 'chroma_vectro_segment_data.jsonl'  # 向量数据库语义索引 数据文件名称
GRAPH_SEGMENT_DATA = 'graph_segment_data.jsonl'  # 知识图谱 数据文件名称
#向量数据库与集合名称
VECTOR_DATABASES_DATA_DIR = os.path.join(DATA_DIR, 'vector_databases')
CHROMA_DOCUMENT_COLLECTION = 'chroma_document_collection'        # 向量数据库文档 集合名称
CHROMA_SEGMENT_COLLECTION = 'chroma_segment_collection'   # 向量数据库语义索引 集合名称




"""
 小说文章采样基本配置
"""
NOVEL_ARTICLE_SPLITTER_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'novel_article_splitter')
#文本采样
CANDIDATE_SENTENCES = 'candidate_sentences.txt'
# 采样配置
SAMPLING_CONFIG = {
    'samples_per_chapter': 110,
    'score_threshold': 0.2
}

"""
 小说文章批注基本配置 （根据采样数据进行批注）
"""
NOVEL_MARGINALIA_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'novel_marginalia')
# 数据批注
NOVEL_AUTO_MARGINALIA_AFTER_NAME = 'novel_auto_annotated_data.json'


#大模型
load_dotenv()
DEEPSEEK_API_KEY= os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1'

