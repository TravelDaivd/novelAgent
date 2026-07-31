import json
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from tqdm import tqdm

from models.registry.config_text import ConfigText
from models.training.text.text_classifier import  TextClassifier
from models.training.text.text_data_set import TextDataset
from models.util.text_utils import TextUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TextTrain:
    def __init__(self):
        TextUtil.set_seed(ConfigText.text_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_train_model_path = os.path.join(ConfigText.text_train_model_name, 'pytorch_model.bin')
        if os.path.exists(text_train_model_path):
            self.model = TextClassifier.load_trained_model(self,ConfigText.text_train_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigText.text_train_model_name)
        else:
            self.model = TextClassifier(ConfigText.text_model_name, ConfigText.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigText.text_model_name)
        self.model.to(device)
    
        
    def handle_train_loader(self):
        """
        创建数据集
        :return: 
        """
        train_data = ConfigText.text_train_path
        max_length = ConfigText.text_max_length
        label2id = ConfigText.label2id
        train_dataset = TextDataset(train_data, self.tokenizer,max_length,label2id)
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=ConfigText.text_batch_size,
            shuffle=True  # 每个批次将数据打乱
        )
        return train_loader

    def handle_learn_late(self):
        """
        分层学习率
        :return: 
        """
        bert_params = list(self.model.bert.parameters())
        classifier_params = list(self.model.classifier.parameters())
        optimizer_grouped_parameters = [
            {'params': bert_params, 'lr': ConfigText.text_bert_lr},
            {'params': classifier_params, 'lr': ConfigText.text_classifier_lr}
        ]
        return optimizer_grouped_parameters

    def handle_scheduler(self):
        """
        学习率调度
        :return: 
        """
        optimizer_grouped_parameters = self.handle_learn_late()
        train_loader = self.handle_train_loader()
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=ConfigText.text_weight_decay)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = TextUtil.get_class_weights(ConfigText.text_train_path, ConfigText.label2id)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
        # 学习率调度
        total_steps = len(train_loader) * ConfigText.text_epochs
        warmup_steps = int(total_steps * ConfigText.text_warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler,loss_fn,optimizer,train_loader

    def start_train(self):

        scheduler,loss_fn,optimizer,train_loader = self.handle_scheduler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 训练循环
        for epoch in range(ConfigText.text_epochs):
            logger.info(f"\n========== Epoch {epoch + 1}/{ConfigText.text_epochs} ==========")
            # 训练阶段
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc="Training")
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                textArray = batch['text']

                # 清空梯度（防止累积）
                optimizer.zero_grad()
                # 前向传播（模型预测）
                logits = self.model(input_ids, attention_mask)

                # 计算损失率  （预测 VS 真实）
                loss = loss_fn(logits, labels)
                # 反向传播 （计算梯度）
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 更新参数（优化器）
                optimizer.step()
                # 更新学习率（调度器）
                scheduler.step()

                total_loss += loss.item()
                # 【新增】记录异常样本
                if grad_norm > 70:
                    logger.info(f" 高梯度样本检测到text: {textArray}")

                    # 保存这个 batch 的索引，训练结束后去查
                logger.info(
                    f"Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Epoch: {epoch + step / len(train_loader):.4f} | Grad: {grad_norm:.2f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        logger.info("\n训练完成！")

        os.makedirs(ConfigText.text_train_model_name, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ConfigText.text_train_model_name, 'pytorch_model.bin'))
        # 2. 保存模型配置（JSON）
        autoconfig = AutoConfig.from_pretrained(ConfigText.text_model_name)
        autoconfig.model_name = ConfigText.text_train_model_name
        autoconfig.label2id = ConfigText.label2id
        autoconfig.id2label = ConfigText.id2label
        autoconfig.dropout = ConfigText.text_dropout
        autoconfig.save_pretrained(ConfigText.text_train_model_name)
        # 3. 保存标签映射（JSON）
        with open(os.path.join(ConfigText.text_train_model_name, 'label2id.json'), 'w', encoding='utf-8') as f:
            json.dump(ConfigText.label2id, f, ensure_ascii=False, indent=2)

        # 4. 保存 tokenizer
        self.tokenizer.save_pretrained(ConfigText.text_train_model_name)
        logger.info(f"模型已保存到: {ConfigText.text_train_model_name}")
        logger.info(f"文件列表: {os.listdir(ConfigText.text_train_model_name)}")

   

if __name__ == "__main__":
    TextTrain().start_train()