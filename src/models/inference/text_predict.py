import glob
import json
import logging
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.registry.config_text import ConfigText
from models.training.text.text_classifier import TextClassifier
from models.training.text.text_data_set import TextDataSet
from models.util.models_utils import ModelsUtils
from models.util.text_utils import TextUtil
from utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TextPredict:
    def __init__(self,text_recognition_dir,device):
        self.tokenizer = AutoTokenizer.from_pretrained(text_recognition_dir)
        self.model = self.load_mode(text_recognition_dir,device)
        
        
    @staticmethod 
    def load_mode(text_recognition_dir,device):
        with open(os.path.join(ConfigText.text_train_model_name, 'config.json'), 'r') as file:
              config = json.load(file)
        model = TextClassifier(config['model_name'],len(config['id2label']))
        model_path = os.path.join(text_recognition_dir, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
        

    def predict_with_threshold(self,text, id2label, threshold=0.5, idx=1,chapter_id=0 ):
        """
        预测文本类别，如果置信度低于阈值，返回 None（表示不属于任何类别）
        """
        # 编码
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=ConfigText.text_max_length,
            return_tensors='pt'
        )

        # 推理
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(outputs, dim=1)  # 转换成概率
            max_prob, pred_id = torch.max(probs, dim=1)

        max_prob = max_prob.item()
        pred_id = pred_id.item()
        # 判断是否低于阈值
        if max_prob < threshold:
            return None  # 不属于任何类别
        
        return {
            "id":f"seg_{chapter_id}_{idx}",
            'text':text,
            'label': id2label[pred_id],
            'confidence': f"{max_prob:.2f}",
            'segment_id': f"seg_{chapter_id}_{idx}",
            'chapter_id':chapter_id,
            'order':idx
        }
    


    def batch_predict_with_filter(self,texts, id2label, threshold=0.5, chapter_id=0):
        """
        批量预测，自动过滤低于阈值的结果
        返回：有效结果列表 + 被过滤的文本列表
        """
        valid_result_array = []
        filtered_text_array = []

        for idx,text in enumerate(texts,1):
            result = self.predict_with_threshold(text,id2label,threshold,idx,chapter_id)
            if result is None:
                filtered_text_array.append(text)
            else:
                valid_result_array.append({
                    'text': text,
                    **result
                })

        return valid_result_array, filtered_text_array


    #设置chromDB 主检索
    def chromDBTwo(self,dataList,unique_chapters:int,file_name:str ):
        chrom_db_data = []
        segment_node_data = []
        for idx, data in enumerate(dataList,1):
            chrom_db_data.append({
                'id':data.get("segment_id"),
                'text':data.get("text"),
                'metadata':{
                    'label': data.get("label"),
                    'confidence': data.get("confidence"),
                    'chapter_id': data.get("chapter_id"),
                    'order': data.get("order")
                }
            })
            segment_node_data.append({
                'segment_id':data.get("segment_id"),
                'chapter_id': data.get("chapter_id"),
                'label': data.get("label"),
                'order': data.get("order")
            })
        chapter_segment={
            'chapter_id': unique_chapters,
            'title': file_name,
            'segment_array': segment_node_data
        }
        return  chrom_db_data,chapter_segment

    def predict_after_ata(self,chromadb_data_array):
        write_split_file_path = os.path.join(SPLITS_DATA_DIR, CHROMA_VECTRO_SEGMENT_DATA)
        with open(write_split_file_path, 'w', encoding='utf-8') as file:
            for data in chromadb_data_array :
                metadata = data.get("metadata")
                file.write(json.dumps({
                    'id': data.get("id"),
                    'text': data.get("text"),
                    "label": metadata.get("label"),
                    "confidence": metadata.get("confidence"),
                    'chapter_id': metadata.get("chapter_id"),
                    'order': metadata.get("order")
                }, ensure_ascii=False) + '\n')

    def chapter_segment_data(self,chapter_segment_node_data):
        chapter_segment_data_path = os.path.join(SPLITS_DATA_DIR, GRAPH_SEGMENT_DATA)
        with open(chapter_segment_data_path, 'w', encoding='utf-8') as file:
            for chapter_segment_data in chapter_segment_node_data:
                file.write(json.dumps(chapter_segment_data, ensure_ascii=False) + '\n')

    def apraise_model_after_data(self):
        appraise_data = ConfigText.appraise_text_data_path
        max_length = ConfigText.text_max_length
        label2id = ConfigText.label2id
        person_data_set = TextDataSet(appraise_data, self.tokenizer, max_length, label2id)
        train_loader = DataLoader(
            person_data_set,
            batch_size=ConfigText.text_batch_size,
            shuffle=True  # 每个批次将数据打乱
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = TextUtil.appraise_relation_model(self.model, train_loader, ConfigText.id2label, device)
        TextUtil.print_error_analysis(result)

if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD = 0.67  # 置信度阈值
    textPredict = TextPredict(ConfigText.text_train_model_name,device)
    """
    textPredict.apraise_model_after_data()
    
    """
    
    txt_file_path_list = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
    autoPlotMarginalia_list = []
    chrom_db_data_list = []
    chapter_segment_list = []
    for index,file_path in enumerate (txt_file_path_list):
        file_name = os.path.basename(file_path)
        chapter_numbers = re.findall(r'\d+', file_name)
        unique_chapters = list(set(int(num) for num in chapter_numbers))[0]
        logger.info(file_name)
        sentence_list = ModelsUtils.split_into_sentences(file_path)
        # 3. 批量预测并过滤
        valid_result_array, filtered_text_array = textPredict.batch_predict_with_filter(
            texts=sentence_list,
            id2label=ConfigText.id2label,
            threshold=THRESHOLD,
            chapter_id = unique_chapters
        )
        logger.info(f"valid data sum : {len(valid_result_array)},filtered data sum : {len(filtered_text_array)},")
        autoPlotMarginalia_list.extend(valid_result_array)
        chrom_db_data,text_node_data = textPredict.chromDBTwo(valid_result_array,unique_chapters,file_name)
        chapter_segment_list.append(text_node_data)
        chrom_db_data_list.extend(chrom_db_data)
    #json.dumps(autoPlotMarginalia_list, ensure_ascii=False, indent=2)
    #textPredict.predict_after_ata(chrom_db_data_list)
   # textPredict.chapter_segment_data(chapter_segment_list)


    # 保存到文件
    auto_plot_marginalia = os.path.join(AUTO_DATA_DIR, AUTO_TEXT_MARGINALIA_NAME)
    # 直接写入文件
    with open(auto_plot_marginalia, 'w', encoding='utf-8') as file:
        json.dump(autoPlotMarginalia_list, file, ensure_ascii=False, indent=2)
        #for autoTextMarginalia in autoPlotMarginalia_list :
         #   file.write(json.dumps(autoTextMarginalia, ensure_ascii=False) + '\n')





