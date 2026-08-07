import logging
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score,classification_report
from models.registry.config_text import ConfigText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
    def appraise_relation_model(model, dataloader, id2label, device):
        """

        """
        # 存储所有预测和真实标签
        all_predictions = []
        all_true_labels = []
        all_probs = []
        all_texts = []
        all_entity_pairs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                # 获取文本和实体信息（用于错误分析）
                texts = batch.get('text', [''] * len(input_ids))
                logits = model(input_ids, attention_mask)
                # 获取预测结果
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # 存储结果
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_texts.extend(texts)

        # 转换为标签名称
        pred_labels = [id2label.get(int(pred), "UNKNOWN") for pred in all_predictions]
        true_labels = [id2label.get(int(true), "UNKNOWN") for true in all_true_labels]
        # 计算评估指标
        results = TextUtil.appraise_model_result(true_labels, pred_labels)

        # 添加详细结果用于错误分析
        results['detailed_results'] = TextUtil.get_detailed_results(
            all_texts, true_labels, pred_labels, all_probs
        )

        return results

    @staticmethod
    def appraise_model_result(true_labels, pred_labels):
        """
        计算评估指标（保持与实体识别评估一致的风格）
        """
        # 获取所有唯一的标签
        unique_labels = list(ConfigText.label2id.keys())

        # 输出分类报告
        logger.info("\n" + "=" * 60)
        logger.info("关系抽取分类报告")
        logger.info("=" * 60)

        # 生成分类报告
        report = classification_report(
            true_labels,
            pred_labels,
            labels=unique_labels,
            output_dict=True,
            zero_division=0
        )

        # 打印人类可读版本
        logger.info(classification_report(
            true_labels,
            pred_labels,
            labels=unique_labels,
            zero_division=0
        ))

        # 提取整体指标
        logger.info("\n" + "=" * 60)
        logger.info("整体指标")
        logger.info("=" * 60)

        # 计算准确率
        accuracy = accuracy_score(true_labels, pred_labels)

        # 提取微平均指标
        if 'micro avg' in report:
            logger.info(f"精确率 (Precision): {report['micro avg']['precision']:.4f}")
            logger.info(f"召回率 (Recall):    {report['micro avg']['recall']:.4f}")
            logger.info(f"F1 值 (F1-score):   {report['micro avg']['f1-score']:.4f}")
        else:
            # 如果没有micro avg，使用macro avg
            logger.info(f"精确率 (Precision): {report['macro avg']['precision']:.4f}")
            logger.info(f"召回率 (Recall):    {report['macro avg']['recall']:.4f}")
            logger.info(f"F1 值 (F1-score):   {report['macro avg']['f1-score']:.4f}")

        logger.info(f"准确率 (Accuracy):  {accuracy:.4f}")

        # 检查是否有 "UNKNOWN" 类别（可能表示预测失败）
        if "UNKNOWN" in report:
            logger.info("\n" + "=" * 60)
            logger.info("UNKNOWN 类别指标（预测失败）")
            logger.info("=" * 60)
            logger.info(f"精确率 (Precision): {report['UNKNOWN']['precision']:.4f}")
            logger.info(f"召回率 (Recall):    {report['UNKNOWN']['recall']:.4f}")
            logger.info(f"F1 值 (F1-score):   {report['UNKNOWN']['f1-score']:.4f}")
            logger.info(f"支持样本数:          {report['UNKNOWN']['support']}")
        return {
            'report': report,
            'accuracy': accuracy,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }

    @staticmethod
    def get_detailed_results(texts, true_labels, pred_labels, probs):
        """
        获取详细的预测结果（用于错误分析）
        """
        detailed = []
        for i, (text, true, pred, prob) in enumerate( zip(texts, true_labels, pred_labels, probs)):
            detailed.append({
                'index': i,
                'text': text,
                'true_label': true,
                'pred_label': pred,
                'is_correct': true == pred,
                'confidence': float(max(prob))
            })
        return detailed

    @staticmethod
    def print_error_analysis(results, top_k=5):
        """
        打印错误分析（与实体识别评估风格一致）

        Args:
            results: evaluate方法的返回结果
            top_k: 显示的错误样本数量
        """
        detailed_results = results.get('detailed_results', [])
        true_labels = results.get('true_labels', [])
        pred_labels = results.get('pred_labels', [])

        logger.info("\n" + "=" * 60)
        logger.info("错误分析（前{}个错误样本）".format(top_k))
        logger.info("=" * 60)

        # 找出预测错误的样本
        error_count = 0
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
            if true != pred:
                error_count += 1
                if error_count <= top_k:
                    # 获取详细样本信息
                    detail = detailed_results[i] if i < len(detailed_results) else None

                    logger.info(f"\n样本 {i + 1}:")
                    if detail:
                        logger.info(f"  文本: {detail['text']}")
                        logger.info(f"  真实: {true}")
                        logger.info(f"  预测: {pred}")
                        logger.info(f"  置信度: {detail['confidence']:.4f}")
                    else:
                        logger.info(f"  真实: {true}")
                        logger.info(f"  预测: {pred}")

        logger.info(f"\n总错误样本数: {error_count} / {len(true_labels)}")
        logger.info(f"准确率 (Accuracy): {(len(true_labels) - error_count) / len(true_labels):.4f}")
