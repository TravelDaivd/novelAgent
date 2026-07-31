import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.registry.config_relation import ConfigRelation
from models.training.relation.relation_classifier import RelationClassifier
from models.training.relation.relation_data_set import RelationDataSet
from models.util.inference_utils import InferenceUtils
from models.util.relation_utils import RelationUtils
from utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RelationPredict:
    
    
    
    def __init__(self,relation_recognition_dir,device):
        # 1. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(relation_recognition_dir)
        self.model = self.load_model(relation_recognition_dir,device)
    
    @staticmethod
    def load_model(relation_recognition_dir,device):
        """
         加载新训练的模型
        """
        # 1. 加载配置
        with open(os.path.join(relation_recognition_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        # 2. 重建模型
        model = RelationClassifier(
            model_name=config['model_name'],
            num_labels=len(config['id2label'])
        )
        # 3. 加载权重
        model_path = os.path.join(relation_recognition_dir, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=device))
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
    threshold = 0.40
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relationPredict = RelationPredict(ConfigRelation.relation_train_model_name, device)
    
    """
    relationPredict.apraise_model_after_data()
    """
    auto_relation_marginalia_name = os.path.join(AUTO_DATA_DIR, AUTO_RELATION_MARGINALIA_NAME)
    with open(auto_relation_marginalia_name, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        relation_train_data = []
        for item in data_list:
            text  = item.get("text")
            relation_data = []
            for relation in item.get("relations", []):
                entity_values = [v for k, v in relation.items() if k.startswith("entity")]
                entity_key = [k for k, v in relation.items() if k.startswith("entity")]
                if len(entity_values) >= 2:
                    entity_one =  entity_values[0]
                    entity_two = entity_values[1]
                    label,confidence = relationPredict.predict_with_threshold(text,entity_one,entity_two,threshold)
                    relation_data.append({
                        f"{entity_key[0]}": entity_one,
                        f"{entity_key[1]}": entity_two,
                        "relation": label,
                        "confidence": f"{float(confidence):.2f}" 
                    })

            relation_train_data.append({
                "text": text,
                "relations": relation_data,
            })
        output_file = os.path.join(SPLITS_DATA_DIR, "relation_train_data.json")
        # 保存去重后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relation_train_data, f, ensure_ascii=False, indent=2)
        
        