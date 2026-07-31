import json
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from models.registry.config_relation import ConfigRelation
from models.util.relation_utils import RelationUtils


class RelationClassifier(nn.Module):
    
    def __init__(self,model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(ConfigRelation.relation_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
           前向传播 - 适配数据格式
        """
        # 1. BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 2. 获取句子表示（优先使用 pooler_output，否则取 last_hidden_state 的 [CLS]）
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # 回退方案：取 last_hidden_state 的 [CLS] 位置
            pooled = outputs.last_hidden_state[:, 0, :]
        # 3. Dropout + 分类
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        # 4. 计算损失（如果提供了标签）
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss,logits

        return logits

    def predict(self, input_ids, attention_mask):
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
            probs = torch.softmax(logits, dim=1)
            return probs, logits
    
    
    
    @staticmethod
    def load_trained_model(relation_train_model_dir):
        # 1. 加载配置
        with open(os.path.join(relation_train_model_dir, 'config.json'), 'r') as file:
            config_info = json.load(file)
            # 2. 重建模型（必须和训练时结构一致）
            
            model = RelationClassifier(
                model_name=config_info['model_name'],
                num_labels=len(config_info['label2id'])
            )

            # 3. 加载训练好的权重
            model_path = os.path.join(relation_train_model_dir, 'pytorch_model.bin')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model