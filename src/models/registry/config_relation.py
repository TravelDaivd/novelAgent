import os

from utils.config import *


class ConfigRelation:
    # 原始模型路径
    relation_model_name = os.path.join(SHARED_DATA_DIR, CHINESE_MAC_BERT_BASE_NAME)

    # 训练数据路径
    relation_train_data_path = os.path.join(MANUAL_DATA_DIR, RELATION_TRAIN_NAME)
    # 评估模型数据
    appraise_relation_data_path = os.path.join(APPRAISE_DATA_DIR, APPRAISE_RELATION_DATA_NAME)

    # 训练模型路径
    relation_train_model_name = os.path.join(MODELS_DATA_DIR, RELATION_RECOGNITION_CHINESE_MAC_BERT_NAME)

    label2id = {"师徒": 0, "同门": 1, "道侣": 2, "仇敌": 3, "朋友": 4, "血缘至亲": 5,"认识":6}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)

    relation_max_length = 256
    relation_batch_size = 17
    relation_epochs = 2  # 训练轮数
    bert_lr = 1e-5
    classifier_lr = 3e-5
    relation_dropout = 0.2
    weight_decay = 0.03  # 权重衰减
    seed = 42
    relation_warmup_ratio = 0.1  # 预热比例
    