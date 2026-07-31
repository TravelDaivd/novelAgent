import json
import logging
from abc import ABCMeta

import torch
from torch.utils.data import Dataset

from models.registry.config_relation import ConfigRelation
from models.util.relation_utils import RelationUtils


class RelationDataSet(Dataset, metaclass=ABCMeta):
    
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
                
        self.relation_data =[]
        for item in self.dataArray :
            text =  item.get("text")
            for relation in item.get("relations",[]):
                # 直接获取所有 entity 开头的键的值
                entity_values = [v for k, v in relation.items() if k.startswith("entity")]
                if len(entity_values) >= 2:
                    self.relation_data.append({
                        "text": text,
                        "entity_one": entity_values[0],
                        "entity_two": entity_values[1],
                        "label": relation.get("relation", "")
                    })
        logging.info(f"关系抽取训练数据数量：{len(self.relation_data)}")
        

    def __len__(self):
        return len(self.relation_data)

    def encode_for_training(self, text, entity_one, entity_two, label):
        """
            训练时编码
        """
        combined_text = RelationUtils.build_input(text, entity_one, entity_two)
        max_length = ConfigRelation.relation_max_length
        encoding = RelationUtils.relation_tokenize(self.tokenizer,combined_text,max_length)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label2id.get(label, 0), dtype=torch.long)
        }
    
    def __getitem__(self, idx):
        item = self.relation_data[idx]
        return self.encode_for_training(
            text=item.get('text', ''),
            entity_one=item.get('entity_one', ''),
            entity_two=item.get('entity_two', ''),
            label=item.get('label', '')
        )
        