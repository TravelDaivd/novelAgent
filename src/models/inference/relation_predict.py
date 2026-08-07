import json
import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.registry.config_relation import ConfigRelation
from models.training.relation.relation_classifier import RelationClassifier
from models.training.relation.relation_data_set import RelationDataSet
from models.util.models_utils import ModelsUtils
from models.util.relation_utils import RelationUtils
from utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RelationPredict:
    
    
    
    def __init__(self,relation_recognition_dir):
        # 1. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(relation_recognition_dir)
        self.model = self.load_model(relation_recognition_dir)
    
    @staticmethod
    def load_model(relation_recognition_dir):
        """
         加载新训练的模型
        """
        files = os.listdir(relation_recognition_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'adapter_config.json' in files and 'adapter_model.safetensors' in files:
            model = ModelsUtils.load_lora_model(
                base_model_dir=ConfigRelation.relation_model_name,
                train_after_model_dir=relation_recognition_dir,
                label_num=len(ConfigRelation.label2id),
                model_class=RelationClassifier
            )
            model.to(device)
            model.eval()
            return model
        else:
            model = ModelsUtils.load_all_model(
                train_after_model_dir=relation_recognition_dir,
                model_class=RelationClassifier
            )
            model.to(device)
            model.eval()
            return model

    def encode_for_predict(self, text, entity_one, entity_two):
        """
            推理时编码
        """
        combined_text = RelationUtils.build_input(text, entity_one, entity_two)
        max_length = ConfigRelation.relation_max_length
        encoding = RelationUtils.relation_tokenize(self.tokenizer, combined_text,max_length )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    
    def decode_prediction(self, logits, threshold=0.5):
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_id = torch.max(probs, dim=1)
        max_prob = max_prob.item()
        pred_id = pred_id.item()
        # 低于阈值返回"未知"
        if max_prob < threshold:
            return "未知", max_prob

        return ConfigRelation.id2label.get(pred_id, "认识"), max_prob

    def predict(self, text, entity_one, entity_two, threshold=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = self.encode_for_predict(text, entity_one, entity_two)
        input_ids = encoded['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)

        # 2. 模型推理
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # 处理可能的返回值格式
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs
        # 3. 解码
        label, confidence = self.decode_prediction(logits, threshold)
        return {
            'label': label,
            'confidence': confidence,
            'entity_one': entity_one,
            'entity_two': entity_two
        }

    def predict_with_threshold(self, text, entity_one, entity_two, threshold=0.5):
        """
        带阈值预测（兼容你原来的方法名）
        """
        result = self.predict(text, entity_one, entity_two, threshold)
        return result['label'],result['confidence']

    def apraise_model_after_data(self):
        appraise_data = ConfigRelation.appraise_relation_data_path
        max_length = ConfigRelation.relation_max_length
        label2id = ConfigRelation.label2id
        person_data_set = RelationDataSet(appraise_data, self.tokenizer, max_length, label2id)
        train_loader = DataLoader(
            person_data_set,
            batch_size=ConfigRelation.relation_batch_size,
            shuffle=True  # 每个批次将数据打乱
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = RelationUtils.appraise_relation_model(self.model, train_loader, ConfigRelation.id2label,device)
        RelationUtils.print_error_analysis(result)
        
        
        
if __name__ == "__main__":
    threshold = 0.60
    
    
    relationPredict = RelationPredict(ConfigRelation.relation_train_model_name)

    """
    relationPredict.apraise_model_after_data()
    """
    auto_person_marginalia_name = os.path.join(AUTO_DATA_DIR, AUTO_PERSON_MARGINALIA_NAME)
    with open(auto_person_marginalia_name, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        relation_train_data = []
        for index,item in enumerate(data_list):
            text  = item.get("text")
            relation_data = []
            entity_names = [entity["name"] for entity in item.get("entities")]
            if len(entity_names) == 1 :continue
            logger.info(text)
            
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    if entity_names[i] == entity_names[j]: continue
                    label, confidence = relationPredict.predict_with_threshold(text, entity_names[i],entity_names[j],threshold)
                    if label == "未知" :continue
                    relation_data.append({
                        f"entity1": entity_names[i],
                        f"entity2": entity_names[j],
                        "relation": label,
                        "confidence": f"{float(confidence):.2f}"
                    })

            relation_train_data.append({
                "text": text,
                "relations": relation_data,
            })
        output_file = os.path.join(AUTO_DATA_DIR, AUTO_RELATION_MARGINALIA_NAME)
        # 保存去重后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relation_train_data, f, ensure_ascii=False, indent=2)
        
     