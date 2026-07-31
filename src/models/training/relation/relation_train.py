import json
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig

from models.registry.config_relation import ConfigRelation
from models.training.relation.relation_classifier import RelationClassifier
from models.training.relation.relation_data_set import RelationDataSet
from models.util.relation_utils import RelationUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RelationTrain:
    
    def __init__(self):
        RelationUtils.set_seed(ConfigRelation.seed)

        model_path = os.path.join(ConfigRelation.relation_train_model_name, 'pytorch_model.bin')
        config_path = os.path.join(ConfigRelation.relation_train_model_name, 'config.json')
        if os.path.exists(model_path) and os.path.exists(config_path):
            self.model = RelationClassifier.load_trained_model(relation_train_model_dir= ConfigRelation.relation_train_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigRelation.relation_train_model_name)
        else:
            self.model = RelationClassifier(model_name=ConfigRelation.relation_model_name,
                                       num_labels=ConfigRelation.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(ConfigRelation.relation_model_name)
            
            
    def handle_train_loader(self):
        """
        创建数据加载器
        :return: 
        """
        train_data  = ConfigRelation.relation_train_data_path
        max_length = ConfigRelation.relation_max_length
        label2id = ConfigRelation.label2id
        person_data_set =  RelationDataSet(train_data,self.tokenizer,max_length,label2id)
        train_loader = DataLoader(
            person_data_set,
            batch_size=ConfigRelation.relation_batch_size,
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
            {'params': bert_params, 'lr': ConfigRelation.bert_lr},
            {'params': classifier_params, 'lr': ConfigRelation.classifier_lr}
        ]
        return optimizer_grouped_parameters
    
    
    def handle_scheduler(self):
        """
        学习率调度
        :return: 
        """
        optimizer_grouped_parameters = self.handle_learn_late()
        train_loader = self.handle_train_loader()
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=ConfigRelation.weight_decay)
        total_steps = len(train_loader) * ConfigRelation.relation_epochs
        warmup_steps = int(total_steps * ConfigRelation.relation_warmup_ratio)
        class_weights = RelationUtils.get_class_weights(
            ConfigRelation.relation_train_data_path, ConfigRelation.label2id, "balanced")
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            loss_fn = nn.CrossEntropyLoss()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler,train_loader,optimizer,loss_fn

    def start_train(self):
        """
        开始训练
        :return: 
        """
        scheduler, train_loader, optimizer,loss_fn = self.handle_scheduler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 训练循环
        for epoch in range(ConfigRelation.relation_epochs):
            logger.info(f"\n========== Epoch {epoch + 1}/{ConfigRelation.relation_epochs} ==========")
            # 训练阶段
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # 清空梯度（防止累积）
                optimizer.zero_grad()  
                
                # 前向传播（模型预测））
                loss,logits = self.model(input_ids, attention_mask,labels)
                
                # 反向传播 （计算梯度）
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 更新参数（优化器）
                optimizer.step()
                # 更新学习率（调度器）
                scheduler.step()

                total_loss += loss.item()

                # 保存这个 batch 的索引，训练结束后去查
                logger.info(f"Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Epoch: {epoch + step / len(train_loader):.4f} | Grad: {grad_norm:.2f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        logger.info("\n训练完成！")

        os.makedirs(ConfigRelation.relation_train_model_name, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(ConfigRelation.relation_train_model_name, 'pytorch_model.bin'))
        # 2. 保存模型配置（JSON）
        autoconfig = AutoConfig.from_pretrained(ConfigRelation.relation_model_name)
        autoconfig.model_name = ConfigRelation.relation_train_model_name
        autoconfig.label2id = ConfigRelation.label2id
        autoconfig.id2label = ConfigRelation.id2label
        autoconfig.dropout = ConfigRelation.relation_dropout
        autoconfig.save_pretrained(ConfigRelation.relation_train_model_name)
        # 3. 保存标签映射（JSON）
        with open(os.path.join(ConfigRelation.relation_train_model_name, 'label2id.json'), 'w', encoding='utf-8') as f:
            json.dump(ConfigRelation.label2id, f, ensure_ascii=False, indent=2)

        # 4. 保存 tokenizer
        self.tokenizer.save_pretrained(ConfigRelation.relation_train_model_name)
        logger.info(f"模型已保存到: {ConfigRelation.relation_train_model_name}")
        logger.info(f"文件列表: {os.listdir(ConfigRelation.relation_train_model_name)}")


if __name__ == "__main__":
    RelationTrain().start_train()