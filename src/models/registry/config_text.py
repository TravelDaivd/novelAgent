import os

from utils.config import *


# 配置文件
class ConfigText:
    # 模型路径
    text_model_name = os.path.join(SHARED_DATA_DIR, CHINESE_MAC_BERT_BASE_NAME)

    # 数据路径
    text_train_path = os.path.join(MANUAL_DATA_DIR, TEXT_TRAIN_NAME)

    # 新模型路径
    text_train_model_name = os.path.join(MODELS_DATA_DIR, TEXT_RECOGNITION_CHINESE_MAC_BERT_NAME)

    # 训练参数
    text_max_length = 256          # 最大序列长度
    text_batch_size =5              # 批次大小
    text_epochs = 2                # 训练轮数
    text_bert_lr = 1e-5            # BERT层学习率
    text_classifier_lr = 3e-5      # 分类层学习率
    text_weight_decay = 0.01       # 权重衰减
    text_dropout = 0.3             # 防止过拟合

    label2id = {"战斗": 0,"修炼": 1,"探索": 2,"对话": 3, "内心": 4}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    
    # 标签平衡（根据实际数据分布调整）
    text_class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # 战斗,修炼,探索,对话,内心
    # 如果日常样本很少，给更高权重

    

    
    # 其他
    text_seed = 42                 # 随机种子
    text_warmup_ratio = 0.1        # 预热比例
    text_eval_steps = 100          # 评估间隔
    text_save_steps = 500          # 保存间隔