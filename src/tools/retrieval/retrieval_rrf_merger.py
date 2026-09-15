from collections import defaultdict
from typing import List, Dict


class RetrievalRRFMerger:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, result_lists: List[List[Dict]]) -> List[Dict]:
        """融合多个检索结果（等权重）"""
        weights = [1.0] * len(result_lists)
        return self.fuse_with_weights(result_lists, weights)

    def fuse_with_weights(self, result_lists: List[List[Dict]], weights: List[float]) -> List[Dict]:
        """带权重的 RRF 融合"""
        if len(result_lists) != len(weights):
            raise ValueError("result_lists 和 weights 长度必须一致")

        rrf_scores = defaultdict(float)
        doc_data = {}

        for sys_idx, system_results in enumerate(result_lists):
            weight = weights[sys_idx]
            for rank, doc in enumerate(system_results, start=1):
                doc_id = doc.get('id')
                if not doc_id:
                    continue

                rrf_scores[doc_id] += weight / (rank + self.k)
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc.copy()
                    doc_data[doc_id]['rrf_score'] = 0.0
                    doc_data[doc_id]['hit_systems'] = []
                    doc_data[doc_id]['weighted_score'] = 0.0

                system_name = doc.get('source', 'unknown')
                if system_name not in doc_data[doc_id]['hit_systems']:
                    doc_data[doc_id]['hit_systems'].append(system_name)

        for doc_id, score in rrf_scores.items():
            if doc_id in doc_data:
                doc_data[doc_id]['rrf_score'] = round(score, 6)
                doc_data[doc_id]['weighted_score'] = round(score, 6)

        return sorted(
            doc_data.values(),
            key=lambda x: x.get('rrf_score', 0),
            reverse=True
        )