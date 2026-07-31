
import torch
from torch.utils.data import Dataset
import json

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.dataArray = []

        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.texts = []

        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data,list):
                self.dataArray.extend(data)
            else:
                self.dataArray.append(data)

        for item in self.dataArray:
            encoding = self.tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            self.labels.append(label2id[item['label']])
            self.texts.append(item['text'])




    def __len__(self):
        return len(self.dataArray)

    def __getitem__(self, idx):

        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }