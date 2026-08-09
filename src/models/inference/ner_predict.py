import glob
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.registry.config_person import ConfigPerson
from models.training.person.ner_classifier import NerClassifier
from models.training.person.ner_data_set import NerDataSet
from models.util.models_utils import ModelsUtils
from models.util.ner_utils import NerUtils
from utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class NerPredict:
    
    def __init__(self,ner_recognition_dir,device):
        self.tokenizer = AutoTokenizer.from_pretrained(ner_recognition_dir)
        self.model = self.load_model(ner_recognition_dir,device)
        
    @staticmethod   
    def load_model(ner_recognition_dir, device):
        """
        加载训练好的模型
        :param ner_recognition_dir: 
        :param device: 
        :return: 
        """
        # 1. 加载配置
        with open(os.path.join(ConfigPerson.person_train_model_name, 'config.json'), 'r') as f:
            config = json.load(f)

        # 2. 重建模型
        model = NerClassifier(
            model_name=ConfigPerson.person_train_model_name,
            num_labels=ConfigPerson.num_labels
        )
        # 3. 加载权重
        model_path = os.path.join(ner_recognition_dir, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model

    def encode_for_inference(self, text):
        
        """
        推理时编码
        :param text: 推理文本
        :return: 
        """
        encoding = NerUtils.ner_tokenize(self.tokenizer,text,ConfigPerson.person_max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'offset_mapping': encoding['offset_mapping'][0].cpu().numpy(),
            'text': text
        }
    
    def predict_entities(self, text, device, threshold=0.5):
        # 1. 编码
        encoded = self.encode_for_inference(text)
        input_ids = encoded['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)
        offset_mapping = encoded['offset_mapping']

        # 2. 模型推理
        with torch.no_grad():
            logit = self.model(input_ids, attention_mask)[0]
            probabilities = torch.softmax(logit, dim=-1)
            predictions = torch.argmax(logit, dim=-1)
        seq_len = attention_mask[0].sum().item()
        offset_mapping = offset_mapping[1:seq_len - 1]
        # 4. 解码实体
        return self.decode_predictions(
            text=text,
            pred_ids=predictions,
            probs=probabilities,
            offset_mapping=offset_mapping,
            threshold=threshold
        )
    def decode_predictions(self, text, pred_ids, probs, offset_mapping, threshold = 0.5) :
        """
        解码预测结果：将模型输出转换为实体列表
        :param text: 推理文本
        :param pred_ids: 预测标签 ID [seq_len]
        :param probs: 预测概率 [seq_len, num_labels]
        :param offset_mapping: 到字符的偏移映射
        :param threshold:  置信度阈值，默认使用
        :return: 
        """
        entities = []
        i = 0
        seq_len = len(pred_ids)
        while i < seq_len:
            label_id = pred_ids[i].item()
            if label_id == -100:
                i += 1
                continue
            # 获取标签名称
            label_name = ConfigPerson.id2label.get(label_id, "O")

            # 检查是否是实体开始 (B-*)
            if label_name.startswith("B-PER"):
                # 记录起始位置和置信度
                start_char = offset_mapping[i][0]
                confidence_list = [probs[i][label_id].item()]
                # 向后收集连续的 I-* 标签
                j = i + 1
                while j < seq_len:
                    next_label_id = pred_ids[j].item()
                    if next_label_id == -100:
                        j += 1
                        continue
                    next_label_name = ConfigPerson.id2label.get(next_label_id, "O")
                    # 检查是否是同一个实体的延续
                    if next_label_name == f"I-PER":
                        confidence_list.append(probs[j][next_label_id].item())
                        j += 1
                    else:
                        break
                # 计算平均置信度
                avg_confidence = sum(confidence_list) / len(confidence_list) if confidence_list else 0.0

                # 【修改点】只有平均置信度 >= threshold 才加入实体
                if avg_confidence >= threshold:
                    # 获取实体结束位置
                    end_char = offset_mapping[j - 1][1]
                    # 提取实体文本
                    entity_text = text[start_char:end_char].strip()
                    # 过滤空实体
                    if len(entity_text) > 1 :
                        entities.append({
                            "name": entity_text,
                            "start": int(start_char),
                            "end": int(end_char),
                            "confidence": f"{float(avg_confidence):.2f}"
                        })
                i = j
            else:
                i += 1
        # 去重逻辑保持不变
        entities.sort(key=lambda x: len(x['name']), reverse=True)
        unique_entities = {}
        for ent in entities:
            name = ent['name']
            confidence = float(ent['confidence'])
            if name not in unique_entities or confidence > float(unique_entities[name]['confidence']):
                unique_entities[name] = ent

        return list(unique_entities.values())
        
    
    
    
    
    def apraise_model_after_data(self):
        appraise_data = ConfigPerson.appraise_person_data_path
        max_length = ConfigPerson.person_max_length
        label2id = ConfigPerson.label2id
        person_data_set = NerDataSet(appraise_data, self.tokenizer, max_length, label2id)
        train_loader = DataLoader(
            person_data_set,
            batch_size=ConfigPerson.person_batch_size,
            shuffle=True  # 每个批次将数据打乱
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_labels, pred_labels,person_text = NerUtils.appraise_ner_model(self.model,train_loader,ConfigPerson.id2label,device)
        NerUtils.appraise_model_result(true_labels,pred_labels,person_text)
    
    
    
    
    

# 使用示例
if __name__ == "__main__":
    threshold  = 0.97
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerPredict =  NerPredict(ConfigPerson.person_train_model_name,device)
    """
    nerPredict.apraise_model_after_data()
    
    """
    txt_file_path_list = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
    train_data_array = []
    filter_train_data_array = []
    for index, file_path in enumerate(txt_file_path_list,1):
        sentence_list = ModelsUtils.split_into_sentences(file_path)
        file_name = os.path.basename(file_path)
        
        logger.info(f"{index}+{file_name}")
        for text in sentence_list:
            entities = nerPredict.predict_entities(text,device,threshold)
            if len(entities) == 0:continue
            train_data ={
                "text":text,
                "entities":entities
            }
            train_data_array.append(train_data)
    
    logger.info(f"训练条数：{len(train_data_array)}")
        # 保存到文件
    auto_person_marginalia_name = os.path.join(AUTO_DATA_DIR, AUTO_PERSON_MARGINALIA_NAME)
    with open(auto_person_marginalia_name, 'w', encoding='utf-8') as file:
        json.dump(train_data_array, file, ensure_ascii=False, indent=2)

   