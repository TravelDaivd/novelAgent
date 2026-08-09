import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from models.registry.config_person import ConfigPerson
from models.util.models_utils import ModelsUtils


class NerClassifier(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(ConfigPerson.person_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        
    
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 2. 获取序列输出
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # 3. 分类
        logits = self.classifier(sequence_output)  # [batch, seq_len, num_labels]
        
        if labels is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            class_weights = ModelsUtils.get_class_weights(ConfigPerson.person_train_model_name,ConfigPerson.label2id)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device), ignore_index=-100)
            loss = loss_fn(logits.view(-1, ConfigPerson.num_labels), labels.view(-1))
            return loss, logits

        batch_size = logits.size(0)
        valid_logits = []
        for i in range(batch_size):
            # 计算有效长度：去掉 [CLS] (位置0) 和 [SEP] (最后一个)
            seq_len = attention_mask[i].sum().item()  # 包含 [CLS] 和 [SEP]
            if seq_len > 2:
                # 只保留位置 1 到 seq_len-2（去掉 [CLS] 和 [SEP]）
                valid_logits.append(logits[i, 1:seq_len - 1, :])
            else:
                # 如果序列太短，保留全部（但理论上不会发生）
                valid_logits.append(logits[i])
        
        return valid_logits
    
    
    def model_predict (self, input_ids, attention_mask):
        """
        推理专用：返回 logits
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        
        return logits

    def predict_with_probs(self, input_ids, attention_mask):
        """
        推理方法：返回概率分布
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            return probs, logits

    def load_trained_model(self, persion_train_model_dir):
        # 1. 加载配置
        with open(os.path.join(persion_train_model_dir, 'config.json'), 'r') as file:
            config_info = json.load(file)
            # 2. 重建模型（必须和训练时结构一致）
            model = NerClassifier(
                model_name=ConfigPerson.person_train_model_name,
                num_labels=ConfigPerson.num_labels
            )

            # 3. 加载训练好的权重
            model_path = os.path.join(persion_train_model_dir, 'pytorch_model.bin')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model