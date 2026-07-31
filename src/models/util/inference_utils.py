import json
import logging
import os

import torch
from seqeval.metrics import classification_report, accuracy_score
from seqeval.scheme import IOB2
from transformers import AutoTokenizer

from utils.file_utils import FileUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class InferenceUtils:
    
    @staticmethod
    def split_into_sentences(file_path:str) -> list[str]:
        article_context = FileUtils.get_file_context(file_path)
        return  InferenceUtils.handle_context(article_context)
        
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
    def load_model(model, inference_train_recognition_dir, device):
        """
        从多文件格式加载训练好的模型
        """
        # 1. 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(inference_train_recognition_dir)
        # 4. 加载权重
        model_path = os.path.join(inference_train_recognition_dir, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, tokenizer
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
        
    