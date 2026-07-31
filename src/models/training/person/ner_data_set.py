import json
import logging
from abc import ABCMeta
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from models.registry.config_person import ConfigPerson
from models.util.ner_utils import NerUtils


class NerDataSet(Dataset, metaclass=ABCMeta):

    def __init__(self,data_path, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.dataArray = []

        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data,list):
                self.dataArray.extend(data)
            else:
                self.dataArray.append(data)
            logging.info(f"实体识别训练数据数量：{len(self.dataArray)}")
            entity_counts = [len(item.get('entities', [])) for item in self.dataArray]
            logging.info(f"实体分布:")
            logging.info(f"  0个实体: {sum(1 for c in entity_counts if c == 0)}")
            logging.info(f"  1个实体: {sum(1 for c in entity_counts if c == 1)}")
            logging.info(f"  2个实体: {sum(1 for c in entity_counts if c == 2)}")
            logging.info(f"  3+实体: {sum(1 for c in entity_counts if c >= 3)}")
    
    def __len__(self):
        return len(self.dataArray)

    
    def __getitem__(self, idx):
        data= self.dataArray[idx]
        text = data["text"]
        entities = data.get('entities', [])
        return NerDataSet.encode_for_training(
            self.tokenizer,self.label2id,
            text, entities
        )
    
    @staticmethod
    def encode_for_training(tokenizer,label2id, text, entities):
        encoding = NerUtils.ner_tokenize(tokenizer, text, ConfigPerson.person_max_length)
        char_labels = NerDataSet.create_char_labels( text, entities)
        # 对齐标签：将字符级标签对齐到 token 级
        aligned_labels = NerDataSet.align_labels(label2id,char_labels,encoding['offset_mapping'][0])
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'offset_mapping': encoding['offset_mapping'][0],
            'label': torch.tensor(aligned_labels, dtype=torch.long)
        }



    @staticmethod
    def align_labels(label2id, char_labels, offset_mapping):
        """
        将字符级标签对齐到 token 
        offset_mapping  长度是配置中设置的max_length
        """
        aligned = []
        for idx, (start, end) in enumerate(offset_mapping):
            # 1、处理特殊 token（[CLS], [SEP], [PAD]）
            if start == 0 and end == 0:
                aligned.append(-100) 
                continue
                
            # 2. 由于offset_mapping的长度=Max_length;检查 token 是否在有效范围内
            if end > len(char_labels) or start >= len(char_labels):
                # token 被截断或越界，标记为 "O"
                aligned.append(label2id.get("O", 0))
                continue
            label = char_labels[start]
            # 4. 安全查找 label id
            label_id = label2id.get(label)
            if label_id is None:
                # 如果标签不存在，使用 "O" 作为默认
                label_id = label2id.get("O", 0)

            aligned.append(label_id)
            
        return aligned
    
    @staticmethod
    def create_char_labels(text: str, entities: List[Dict]) -> List[str]:
        """
        创建字符级 BIO 标签
        :param text:  训练文本
        :param entities: 实体列表
        :return: 
        """
        char_labels = ["O"] * len(text)
        for ent in entities:
            start = ent['start']
            end = ent['end']
            # 验证实体位置
            if start < 0 or end > len(text) or start >= end:
                logging.warning(f"Invalid entity position: {ent}")
                continue

            # B-人物（第一个字符）
            char_labels[start] = "B-PER"
            # I-人物（后续字符）
            for i in range(start + 1, end):
                if i < len(char_labels):
                    char_labels[i] = "I-PER"

        return char_labels


