import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score



class TextUtil:
    
    @staticmethod
    def set_seed(seed):
        """设置随机种子保证可复现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def text_tokenize(tokenizer, text, max_length):
        """
        统一的 tokenize 方法
        """
        return tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    @staticmethod
    def compute_metrics(preds, labels):
        """计算评估指标"""
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
    
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

    @staticmethod
    def get_class_weights(train_data_path, label2id):
        """自动计算类别权重"""
        import json
        label_counts = {label_id: 0 for label_id in label2id.values()}
        dataArray = []
        with open(train_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                dataArray.extend(data)
            else:
                dataArray.append(data)
            for item in dataArray:
                label_id = label2id[item['label']]
                label_counts[label_id] += 1
    
        total = sum(label_counts.values())
        weights = [total / count for count in label_counts.values()]
        # 归一化
        weights = [w / sum(weights) for w in weights]
        return weights