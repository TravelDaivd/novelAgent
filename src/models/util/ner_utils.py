import logging
import random
import numpy as np
import torch

from seqeval.metrics import classification_report, accuracy_score
from seqeval.scheme import IOB2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class NerUtils:
    
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def ner_tokenize(tokenizer, text,max_length):
        """统一的 tokenize 方法"""
        return tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
    
    
   
    @staticmethod
    def appraise_ner_model(model, dataloader, id2label, device):
        """
        使用 attention_mask 对齐评估 NER 模型
        """
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()
                # 获取 logits 并取 argmax
                logit_list = model.model_predict(input_ids, attention_mask)
                for index,logit in enumerate(logit_list) :
                    preds = torch.argmax(logit, dim=-1).cpu().numpy()
                    seq_len = attention_mask[index].sum().item()
                    valid_label_list = labels[index][1:seq_len-1]
                    true_seq = []
                    pred_seq_clean = []
                    for pred, label in zip(preds, valid_label_list):
                        if label != -100:
                            true_seq.append(id2label.get(int(label), "O"))
                            pred_seq_clean.append(id2label.get(int(pred), "O"))
                    if true_seq and pred_seq_clean:
                        true_labels.append(true_seq)
                        pred_labels.append(pred_seq_clean)

        return true_labels, pred_labels 
    
    

    @staticmethod
    def appraise_model_result(true_labels, pred_labels):
        # 输出分类报告（每个实体类型的精确率/召回率/F1）
        logger.info("\n" + "=" * 60)
        logger.info("分类报告")
        logger.info("=" * 60)
        report = classification_report(
            true_labels,
            pred_labels,
            mode='strict',
            scheme=IOB2,
            output_dict=True,
            zero_division=0
        )

        # 打印人类可读版本
        logger.info(classification_report(
            true_labels,
            pred_labels,
            mode='strict',
            scheme=IOB2,
            zero_division=0
        ))

        # 提取整体指标
        logger.info("\n" + "=" * 60)
        logger.info("整体指标")
        logger.info("=" * 60)
        logger.info(f"精确率 (Precision): {report['micro avg']['precision']:.4f}")
        logger.info(f"召回率 (Recall):    {report['micro avg']['recall']:.4f}")
        logger.info(f"F1 值 (F1-score):   {report['micro avg']['f1-score']:.4f}")
        logger.info(f"准确率 (Accuracy):  {accuracy_score(true_labels, pred_labels):.4f}")

        # 只关心"人物"类别
        if "PER" in report:
            logger.info("\n" + "=" * 60)
            logger.info("人物类别指标")
            logger.info("=" * 60)
            logger.info(f"精确率 (Precision): {report['PER']['precision']:.4f}")
            logger.info(f"召回率 (Recall):    {report['PER']['recall']:.4f}")
            logger.info(f"F1 值 (F1-score):   {report['PER']['f1-score']:.4f}")
            logger.info(f"支持样本数:          {report['PER']['support']}")

        # 找出预测错误的样本（用于分析）
        logger.info("\n" + "=" * 60)
        logger.info("错误分析（前5个错误样本）")
        logger.info("=" * 60)
        error_count = 0
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
            if true != pred:
                error_count += 1
                if error_count <= 5:
                    logger.info(f"\n样本 {i + 1}:")
                    logger.info(f"  真实: {true[:50]}...")
                    logger.info(f"  预测: {pred[:50]}...")
        logger.info(f"\n总错误样本数: {error_count} / {len(true_labels)}")
        logger.info(f"准确率 (Accuracy): {(len(true_labels) - error_count) / len(true_labels):.4f}")


    
    
    