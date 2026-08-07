import json
import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig

from models.registry.config_relation import ConfigRelation
from models.training.relation.relation_classifier import RelationClassifier
from models.training.relation.relation_data_set import RelationDataSet
from models.util.models_utils import ModelsUtils
from models.util.relation_utils import RelationUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RelationTrain:
    
    def __init__(self):
        RelationUtils.set_seed(ConfigRelation.seed)
        model,tokenizer = ModelsUtils.train_load_model(
            base_model_dir=ConfigRelation.relation_model_name,
            train_after_model_dir=ConfigRelation.relation_train_model_name,
            label_num=len(ConfigRelation.label2id),
            model_class=RelationClassifier
        )
        self.tokenizer = tokenizer
        self.model = model
        
        self.build_lora_config()
       

    def build_lora_config(self):
       # for name,module in self.model.named_modules():
          #  logger.info(f"模型中真实层：{name}")
        lora_config =LoraConfig(
            r = 16,
            lora_alpha=32,
            target_modules=['query','key','value'],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        self.model = get_peft_model(self.model,lora_config)
        """打印模型可训练参数信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("=" * 50)
        logger.info("【LoRA 参数统计】")
        logger.info(f"模型总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")
        logger.info(f"可训练参数占比: {trainable_params / total_params * 100:.4f}%")
        logger.info("=" * 50)
        
    
        
        
    def handle_train_loader(self):
        """
        创建数据加载器
        :return: 
        """
        train_data  = ConfigRelation.relation_train_data_path
        
        max_length = ConfigRelation.relation_max_length
        label2id = ConfigRelation.label2id
        relation_data_set =  RelationDataSet(train_data, self.tokenizer, max_length, label2id)
        train_loader = DataLoader(
            relation_data_set,
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
        class_weights = ModelsUtils.get_class_weights(
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

    def validation_relation_loader(self):
        validation_data = ConfigRelation.relation_validation_data_path
        max_length = ConfigRelation.relation_max_length
        label2id = ConfigRelation.label2id
        validation_relation_data_set = RelationDataSet(validation_data, self.tokenizer, max_length, label2id)
        validation_train_loader = DataLoader(
            validation_relation_data_set,
            batch_size=ConfigRelation.relation_batch_size,
            shuffle=False  # 每个批次将数据打乱
        )
        return validation_train_loader
    
    def validation_data(self, validation_loader, device, loss_fn, epoch):
        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():  # 验证时不需要计算梯度
            for val_batch in tqdm(validation_loader, desc="Validating"):
                val_input_ids = val_batch['input_ids'].to(device)
                val_attention_mask = val_batch['attention_mask'].to(device)
                val_labels = val_batch['label'].to(device)

                # 前向传播
                val_logits = self.model(val_input_ids, val_attention_mask)

                # 计算验证集Loss
                val_loss = loss_fn(val_logits, val_labels)
                total_val_loss += val_loss.item()

                # 计算准确率（可选）
                preds = torch.argmax(val_logits, dim=1)
                correct_predictions += (preds == val_labels).sum().item()
                total_predictions += val_labels.size(0)
                # ... 计算loss和预测 ...
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(validation_loader)
        val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        logger.info(f"Epoch {epoch + 1} validation average loss: {avg_val_loss:.4f}")
        logger.info(f"Epoch  {epoch + 1} validation accuracy: {val_accuracy:.4f}")
        report = classification_report(
            all_labels, all_preds,
            labels=list(ConfigRelation.id2label.keys()),
            target_names=list(ConfigRelation.id2label.values()),
            zero_division=0
        )
        logger.info(f"\n分类报告:\n{report}")


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
                outputs = self.model(input_ids=input_ids,attention_mask =attention_mask,labels=labels)
                loss = outputs["loss"]
                """
                这是因为模型进行过lora封装后，有原先的loss=张量，现在变成loss=向量
                不能直接调用backward，用下面的loss.mean()方法，再把它变成标量
                """
                if loss.dim()>0:
                   loss =  loss.mean()
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
            validation_train_loader = self.validation_relation_loader()
            if validation_train_loader:
                self.validation_data(validation_train_loader,device,loss_fn,epoch)
        
        logger.info("\n训练完成！")

        os.makedirs(ConfigRelation.relation_train_model_name, exist_ok=True)
        """是否使用了LoRA"""
        if hasattr(self.model, 'peft_config'):
            self.model.save_pretrained(ConfigRelation.relation_train_model_name)
        else:
            torch.save(self.model.state_dict(),os.path.join(ConfigRelation.relation_train_model_name, 'pytorch_model.bin'))
        
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