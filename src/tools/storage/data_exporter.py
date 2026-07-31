import json
import os
from typing import List, Dict, Any


class DataExporter:
    @staticmethod
    def export_vector_data(file_path, data,mode="w"):
        """写出 ChromaDB 用的 JSON Lines"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def export_graph_data(file_path, data,mode="w"):
        """写出 Neo4j 用的 JSON Lines"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')