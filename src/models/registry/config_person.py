import os

from utils.config import *


class ConfigPerson:
    # 模型路径
    person_model_name = os.path.join(SHARED_DATA_DIR, CHINESE_MAC_BERT_BASE_NAME)
    # 训练模型保存路径
    person_train_model_name = os.path.join(MODELS_DATA_DIR, PERSON_RECOGNITION_CHINESE_MAC_BERT_NAME)

    # 训练数据路径
    person_train_data_path = os.path.join(MANUAL_DATA_DIR, PERSON_TRAIN_NAME)
    # 评估模型数据
    appraise_person_data_path = os.path.join(APPRAISE_DATA_DIR, APPRAISE_PERSON_DATA_NAME)

    
    label2id = {"O": 0,"B-PER": 1,"I-PER": 2}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)

    person_max_length = 256
    person_batch_size =17          #批次
    person_epochs = 2              # 训练轮数
    bert_lr = 5e-6                 #bert学习率
    classifier_lr = 1e-5           #分层类学习率 是bert的3~5倍
    person_dropout = 0.2           # 强迫模型"举一反三"  0.0 ~ 0.5  在训练时，随机"关掉"一部分神经元，强迫模型不依赖某几个特定的神经元
    weight_decay = 0.03            # 权重衰减 防止模型"死记硬背"  0.03~0.1 给模型的"学习"增加一点"阻力"，防止它学得太"用力"而记住训练数据中的噪声。
    seed = 42
    person_warmup_ratio = 0.1      # 预热比例
    