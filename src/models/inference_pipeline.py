import glob
import json
import logging
import os
import re

import torch

from models.inference.ner_predict import NerPredict
from models.inference.relation_predict import RelationPredict
from models.inference.text_predict import TextPredict
from models.registry.config_person import ConfigPerson
from models.registry.config_relation import ConfigRelation
from models.registry.config_text import ConfigText
from tools.storage.data_exporter import DataExporter
from tools.utils.log_and_catch import log_and_catch
from utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class InferencePipeline:
    """
    小说全文推理管道：
    1. 情节分类（判断属于哪类情节）
    2. 实体识别（提取人名）
    3. 关系抽取（实体间的关系）
    串了3个模型进行内容推理流程：
    文本分类模型->实体识别模型->关系抽取模型
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.txt_file_path_list = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
        self.text_predict = TextPredict(ConfigText.text_train_model_name,self.device)
        self.ner_predict = NerPredict(ConfigPerson.person_train_model_name,self.device)
        self.relationPredict = RelationPredict(ConfigRelation.relation_train_model_name)
        self.text_threshold = 0.67  # 置信度阈值
        self.ner_threshold = 0.97  # 置信度阈值
        self.relation_threshold = 0.57  # 置信度阈值
        self.write_graph_file_path = os.path.join(SPLITS_DATA_DIR, GRAPH_SEGMENT_DATA)
        self.write_chroma_file_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)

    def text_pipeline(self,segmnet,idx,unique_chapters):
        text_data = self.text_predict.predict_with_threshold(segmnet,ConfigText.id2label, self.text_threshold,idx, unique_chapters)
        
        return  text_data

    def entity_pipeline(self,segmnet):
        entity_array = self.ner_predict.predict_entities(segmnet, self.device, self.ner_threshold)
        return entity_array

    def relation_pipeline(self,segmnet,text_data,entity_array):
        tags = ['战斗', '对话', '探索']
        relation_data = []
        if text_data.get("label") in tags:
            entity_names = [entity["name"] for entity in entity_array]
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    if entity_names[i] == entity_names[j]: continue
                    label, confidence = self.relationPredict.predict_with_threshold(segmnet, entity_names[i],entity_names[j],self.relation_threshold)
                    if label == "未知" :continue
                    relation_data.append({
                        f"entity_one": entity_names[i],
                        f"entity_two": entity_names[j],
                        "relation": label,
                        "confidence": f"{float(confidence):.2f}"
                    })
        return relation_data

    @log_and_catch
    def extract_segment(self,sentence_list,unique_chapters,chroma_vector_data_list):
        graph_segment_list = []
        for idx, segment in enumerate(sentence_list, 1):
            text_data = self.text_pipeline(segment, idx, unique_chapters)
            if text_data is None: continue
            graph_segment = {
                'segment_id': text_data.get("segment_id"),
                'chapter_id': text_data.get("chapter_id"),
                'label': text_data.get("label"),
                'order': text_data.get("order")
            }
            entity_array = self.entity_pipeline(segment)
            if len(entity_array) == 0: continue
            text_data.update({"entities": entity_array})
            graph_segment.update({"entities": entity_array})
            relation_data = self.relation_pipeline(segment, text_data, entity_array)
            if len(relation_data) > 0:
                text_data.update({"relations": relation_data})
                graph_segment.update({"relations": relation_data})
            graph_segment_list.append(graph_segment)
            chroma_vector_data_list.append(text_data)
       
        return chroma_vector_data_list,graph_segment_list
    
    
    def write_json_lines(self,chroma_vector_data_list,graph_chapter_segment_list,mode='w'):
        DataExporter.export_vector_data(self.write_chroma_file_path, chroma_vector_data_list,mode)
        DataExporter.export_graph_data(self.write_graph_file_path, graph_chapter_segment_list,mode)
    
    def json_file_data_num(self):
        data = []
        with open(self.write_graph_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"跳过无效行: {e}")
        
        return len(data)
        