import json
import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from models.registry.config_text import ConfigText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(ConfigText.text_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]标记的表示
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits

    def load_trained_model(self,text_train_model_dir):
        # 1. 加载配置
        with open(os.path.join(text_train_model_dir, 'config.json'), 'r') as file:
            config_info = json.load(file)
            # 2. 重建模型（必须和训练时结构一致）
            model = TextClassifier(
                model_name=ConfigText.text_train_model_name,
                num_labels=ConfigText.num_labels
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 3. 加载训练好的权重
            model_path = os.path.join(text_train_model_dir, 'pytorch_model.bin')
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model
