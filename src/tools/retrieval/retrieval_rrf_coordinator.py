import logging
from typing import List, Dict, Optional

from tools.retrieval.retrieval_rrf_merger import RetrievalRRFMerger

logger = logging.getLogger(__name__)


class RetrievalRRFCoordinator:
    """
    RRF 多系统检索协调器
    负责调用各检索系统并用 RRF 融合结果
    """

    def __init__(self, rrf_k: int = 60):
        self.merger = RetrievalRRFMerger(k=rrf_k)
        self.default_weights = [1.0, 1.0]  # [bm25, vector]

    def fuse(
        self,
        query: str,
        bm25_searcher,
        vector_results: Optional[List[Dict]] = None,
        graph_results: Optional[List[Dict]] = None,
        top_k: int = 20,
        chapter_ids: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        system_weights: Optional[List[float]] = None,
        bm25_top_k_multiplier: int = 2,
    ) -> List[Dict]:
        """
        执行多系统检索 + RRF 融合

        Args:
            query: 查询词
            bm25_searcher: BM25Searcher 实例
            vector_results: 向量检索结果（外部传入）
            graph_results: 知识图谱检索结果（外部传入）
            top_k: 最终返回数量
            chapter_ids: 章节过滤
            labels: 标签过滤
            system_weights: [bm25权重, vector权重, graph权重]
            bm25_top_k_multiplier: BM25 多召回倍数

        Returns:
            RRF 融合后的结果
        """
        # 1. BM25 检索
        bm25_results = bm25_searcher.search(
            query=query,
            top_k=top_k * bm25_top_k_multiplier,
            chapter_ids=chapter_ids,
            labels=labels,
            score_threshold=None
        )

        # 2. 标记来源
        for doc in bm25_results:
            doc['source'] = 'bm25'

        if vector_results:
            for doc in vector_results:
                doc['source'] = 'vector'

        if graph_results:
            for doc in graph_results:
                doc['source'] = 'graph'

        # 3. 组装结果列表和权重
        result_lists = [bm25_results]
        weights = [system_weights[0] if system_weights else self.default_weights[0]]

        if vector_results:
            result_lists.append(vector_results)
            weights.append(system_weights[1] if system_weights and len(system_weights) > 1 else self.default_weights[1])

        # 4. 单个系统直接返回
        if len(result_lists) == 1:
            return result_lists[0][:top_k]

        # 5. RRF 融合
        fused = self.merger.fuse_with_weights(result_lists, weights)
        logger.info(f"RRF 融合完成: {len(fused)} 条结果")
        return fused[:top_k]