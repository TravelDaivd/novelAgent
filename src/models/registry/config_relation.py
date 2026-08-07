import os

from utils.config import *


class ConfigRelation:
    # 原始模型路径
    relation_model_name = os.path.join(SHARED_DATA_DIR, CHINESE_MAC_BERT_BASE_NAME)
    # 训练模型路径
    relation_train_model_name = os.path.join(MODELS_DATA_DIR, RELATION_RECOGNITION_CHINESE_MAC_BERT_NAME)

    # 训练数据路径
    relation_train_data_path = os.path.join(MANUAL_DATA_DIR, RELATION_TRAIN_NAME)
    # 验证集数据
    relation_validation_data_path = os.path.join(VALIDATION_DATA_DIR, RELATION_VALIDATION_NAME)
    # 评估模型数据
    appraise_relation_data_path = os.path.join(APPRAISE_DATA_DIR, APPRAISE_RELATION_DATA_NAME)

   
    label2id = {"师徒": 0, "同门": 1, "仇敌": 2, "血缘至亲": 3,"认识":4}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)

    relation_max_length = 256
    relation_batch_size = 7
    relation_epochs = 15  # 训练轮数
    bert_lr = 1e-4    
    classifier_lr = 3e-4
    weight_decay = 0.01  # 权重衰减
    relation_dropout = 0.1
    seed = 42
    relation_warmup_ratio = 0.1  # 预热比例
    