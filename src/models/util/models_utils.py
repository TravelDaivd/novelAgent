import json
import logging
import os
from typing import Union, Any
import numpy as np
import torch
from collections import Counter

from peft import PeftModel
from transformers import AutoTokenizer

from utils.file_utils import FileUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ModelsUtils:
    
    @staticmethod
    def split_into_sentences(file_path:str) -> list[str]:
        article_context = FileUtils.get_file_context(file_path)
        return  ModelsUtils.handle_context(article_context)
        
    @staticmethod    
    def handle_context(article_context):
        sentence_num: int = 5
        raw_sentences = article_context.split('。')
        cleaned_sentences = []
        for raw_s in raw_sentences:
            # 清理：移除换行、首尾空格，但保留内部空格
            cleaned = raw_s.replace('\n', ' ').replace('\r', ' ').strip()
            # 关键：只添加非空句子
            if cleaned:
                cleaned_sentences.append(cleaned)
        #  如果没有有效句子，返回空列表（不是返回包含句号的列表）
        if not cleaned_sentences:
            logger.warning(f"没有有效句子: ")
            return []
        # 分组
        sentence_list = []
        for i in range(0, len(cleaned_sentences), sentence_num):
            group = cleaned_sentences[i:i + sentence_num]
            combined = '。'.join(group) + '。'

            #  额外检查：确保组合后不只是句号
            if combined.strip() != '。':
                sentence_list.append(combined)
            else:
                logger.warning(f"跳过空块: {combined}")

        return sentence_list
    
    @staticmethod
    def load_all_model(train_after_model_dir:str,model_class:Union[type, Any]):
        """
        加载训练好的模型->采用全量参数训练
        :param train_after_model_dir: 训练模型地址
        :param model_class: 模型类 (RelationClassifier/NerClassifier/TextClassifier)
        :return: 
        """
        # 1. 加载配置
        with open(os.path.join(train_after_model_dir, 'config.json'), 'r') as file:
            config_info = json.load(file)
            # 2. 重建模型（必须和训练时结构一致）

            model = model_class(
                model_name=train_after_model_dir,
                num_labels=len(config_info['label2id'])
            )

        # 3. 加载训练好的权重
        model_path = os.path.join(train_after_model_dir, 'pytorch_model.bin')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    @staticmethod
    def load_lora_model(base_model_dir:str,train_after_model_dir:str,label_num:int, model_class:Union[type, Any]):
        """
        加载训练好的模型->采用LoRA训练滴
        :param base_model_dir:  基础模型路径
        :param train_after_model_dir:  训练模型路径
        :param label_num:  标签数量
        :param model_class: 模型类 (RelationClassifier/NerClassifier/TextClassifier)
        :return: 
        """
        files = os.listdir(train_after_model_dir)
        model = model_class(
            model_name=base_model_dir,
            num_labels=label_num
        )
        if 'adapter_config.json' in files and 'adapter_model.safetensors' in files:
            model = PeftModel.from_pretrained(model, train_after_model_dir)
            logger.info("加载 LoRA 适配器成功")
        else:
            raise FileNotFoundError(f"找不到模型文件: {train_after_model_dir}")
        return  model
    
    @staticmethod
    def train_load_model(base_model_dir:str,train_after_model_dir:str,label_num:int, model_class:Union[type, Any]):
        """
        训练时加载模型
        :param base_model_dir:  基础模型路径
        :param train_after_model_dir:  训练模型路径
        :param label_num:  标签数量
        :param model_class: 模型类 (RelationClassifier/NerClassifier/TextClassifier)
        :return: 
      """
        
        files = os.listdir(train_after_model_dir)
        if 'adapter_config.json' in files and 'adapter_model.safetensors' in files:
           model = ModelsUtils.load_lora_model(
                base_model_dir=base_model_dir,
                train_after_model_dir=train_after_model_dir,
                label_num=label_num,
                model_class=model_class
            )
           # 3. 合并适配器到基础模型
           model = model.merge_and_unload()  # 合并后变成普通模型
           tokenizer = AutoTokenizer.from_pretrained(train_after_model_dir)
           return model,tokenizer
        elif 'pytorch_model.bin'in files and 'config.json' in files:
            model = ModelsUtils.load_all_model(
                train_after_model_dir=train_after_model_dir,
                model_class=model_class
            )
            tokenizer = AutoTokenizer.from_pretrained(train_after_model_dir)
            return model, tokenizer
        else:
            model = model_class(model_name=base_model_dir,num_labels=label_num)
            tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
            return model,tokenizer       
        
    
    

    @staticmethod
    def validation_class_weights(logits, labels):
        """
        新增验证代码 logits
        :param logits: 
        :param labels: 
        :return: 
        """
        logging.info(f"raw logits未经过softmax : {logits[0].detach().cpu().numpy()}")
        # 计算 softmax 概率
        probs = torch.softmax(logits[0], dim=-1)
        logging.info(f"softmax 概率: {probs.detach().cpu().numpy()}")
        # 假设真实标签是第一个样本的标签（如果有 labels 的话）
        if labels is not None:
            # 获取第一个样本的真实标签
            true_label = labels[0].item()
            logging.info(f"真实标签: {true_label}")
            # 计算交叉熵损失（手动计算，验证 loss_fn 是否正确）
            loss_manual = -torch.log(probs[true_label])
            logging.info(f"手动计算的 Loss: {loss_manual.item():.4f}")

    @staticmethod
    def get_class_weights(data_path, label2id, method='balanced',min_weight=float(0.5),max_weight=float(3.0)):
        """
           自动计算类别权重 - 适配数据结构
           Args:
                data_path: 数据文件路径
                label2id: 标签映射
                method: 'balanced' 或 'inverse' 或 'log'
                min_weight: 最小权重
                max_weight: 最大权重
           Returns:
               weights: 类别权重列表
        """

        all_labels = []

        # 初始化计数器
        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            items = data if isinstance(data, list) else [data]

            for item in items:
                for relation in item.get('relations', []):
                    label_str = relation.get('relation', '')
                    if label_str in label2id:
                        all_labels.append(label2id[label_str])

        # 统计各类别数量
        label_counts = Counter(all_labels)
        num_classes = len(label2id)
        # 确保所有类别都有计数
        for label_id in label2id.values():
            if label_id not in label_counts:
                label_counts[label_id] = 0

        # 根据不同方法计算权重
        if method == 'balanced':
            # 手动计算 balanced 权重
            # 公式：weight = n_samples / (n_classes * n_samples_per_class)
            total = len(all_labels)
            n_classes = len(label2id)

            weights = []
            for label_id in range(n_classes):
                count = label_counts.get(label_id, 0)
                if count == 0:
                    weight = 1.0  # 如果没有样本，给默认权重
                else:
                    weight = total / (n_classes * count)
                weights.append(weight)

            max_w = max(weights)
            min_w = min(weights)
            if max_w > min_w:
                normalized = [(w - min_w) / (max_w - min_w) for w in weights]
                # 缩放到 [min_weight, max_weight]
                weights = [min_weight + n * (max_weight - min_weight) for n in normalized]
            else:
                # 如果所有权重相等，直接设为 1.0
                weights = [1.0] * len(weights)
        elif method == 'inverse':
            # 方法2: 逆频率
            total = sum(label_counts.values())
            weights = [total / (count + 1e-6) for count in label_counts.values()]
            # 归一化
            weights = [w / sum(weights) for w in weights]

        elif method == 'log':
            # 方法3: 对数逆频率（平滑）
            total = sum(label_counts.values())
            weights = [np.log(total / (count + 1e-6)) for count in label_counts.values()]
            # 归一化
            weights = [w / sum(weights) for w in weights]

        else:
            raise ValueError(f"Unknown method: {method}")

        return weights

