import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig

from models.registry.config_person import ConfigPerson
from models.training.person.ner_classifier import NerClassifier
from models.training.person.ner_data_set import NerDataSet
from models.util.ner_utils import NerUtils


# 用于计算准确率
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class NerTrain:
    
    def __init__(self):
        NerUtils.set_seed(ConfigPerson.seed)
        model_path = os.path.join(ConfigPerson.person_train_model_name, 'pytorch_model.bin')
        config_path = os.path.join(ConfigPerson.person_train_model_name, 'config.json')
        if os.path.exists(model_path) and os.path.exists(config_path) :
            self.model = NerClassifier.load_trained_model(self,
                persion_train_model_dir=ConfigPerson.person_train_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigPerson.person_train_model_name)
        else:
            self.model = NerClassifier(model_name=ConfigPerson.person_model_name,
                                       num_labels=ConfigPerson.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigPerson.person_model_name)
    
    
    
    def handle_train_loader(self):
        """
        创建数据加载器
        :return: 
        """
        train_data  = ConfigPerson.person_train_data_path
        max_length = ConfigPerson.person_max_length
        label2id = ConfigPerson.label2id
        person_data_set =  NerDataSet(train_data,self.tokenizer,max_length,label2id)
        train_loader = DataLoader(
            person_data_set,
            batch_size=ConfigPerson.person_batch_size,
            shuffle=True  # 每个批次将数据打乱
        )
        return train_loader
    def handle_learn_late(self):
        """
        # 分层学习率
        :return: 
        """
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        bert_params = list(self.model.bert.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        optimizer_grouped_parameters = [
            {'params': bert_params, 'lr': ConfigPerson.bert_lr},
            {'params': classifier_params, 'lr': ConfigPerson.classifier_lr}
        ]
        return optimizer_grouped_parameters
            
        
    def handle_scheduler(self):
        """
        学习率调度
        :return: 
        """
        optimizer_grouped_parameters = self.handle_learn_late()
        train_loader = self.handle_train_loader()
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=ConfigPerson.weight_decay)
        total_steps = len(train_loader) * ConfigPerson.person_epochs
        warmup_steps = int(total_steps * ConfigPerson.person_warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler,train_loader,optimizer
    
    
    def start_train(self):
        """
        开始训练
        :return: 
        """
        scheduler,train_loader,optimizer = self.handle_scheduler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练循环
        for epoch in range(ConfigPerson.person_epochs):
            logger.info(f"\n========== Epoch {epoch + 1}/{ConfigPerson.person_epochs} ==========")
            # 训练阶段
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_tokens = 0
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # 清空梯度（防止累积）
                optimizer.zero_grad()

                # 前向传播 （预测 VS 真实）
                loss, logits = self.model(input_ids, attention_mask,labels)
                
                # 反向传播 （计算梯度）
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 更新参数（优化器）
                optimizer.step()
                # 更新学习率（调度器）
                scheduler.step()

                total_loss += loss.item()
                # 9. 计算准确率（可选）
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100  # 忽略特殊 token
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

                # 10. 打印进度（每 10 步）
                if step % 3 == 0:
                    avg_loss = total_loss / (step + 1)
                    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
                    logger.debug(
                        f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, "
                        f"Acc: {accuracy:.4f}, Grad: {grad_norm:.4f}"
                    )
                
                # 保存这个 batch 的索引，训练结束后去查
                logger.info(f"Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Epoch: {epoch + step / len(train_loader):.4f} | Grad: {grad_norm:.2f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        logger.info("\n训练完成！")

        os.makedirs(ConfigPerson.person_train_model_name, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ConfigPerson.person_train_model_name, 'pytorch_model.bin'))
        # 2. 保存模型配置（JSON）
        autoconfig = AutoConfig.from_pretrained(ConfigPerson.person_model_name)
        autoconfig.model_name = ConfigPerson.person_train_model_name
        autoconfig.label2id = ConfigPerson.label2id
        autoconfig.id2label = ConfigPerson.id2label
        autoconfig.dropout = ConfigPerson.person_dropout
        autoconfig.save_pretrained(ConfigPerson.person_train_model_name)
        # 3. 保存标签映射（JSON）
        with open(os.path.join(ConfigPerson.person_train_model_name, 'label2id.json'), 'w', encoding='utf-8') as f:
            json.dump(ConfigPerson.label2id, f, ensure_ascii=False, indent=2)

        # 4. 保存 tokenizer
        self.tokenizer.save_pretrained(ConfigPerson.person_train_model_name)
        logger.info(f"模型已保存到: {ConfigPerson.person_train_model_name}")
        logger.info(f"文件列表: {os.listdir(ConfigPerson.person_train_model_name)}")
        
    
    
if __name__ == "__main__":
    NerTrain().start_train()
    




