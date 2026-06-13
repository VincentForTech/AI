import logging
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import pinecone
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VectorStore:
    """Pinecone 向量存储管理器"""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 384,
    ):
        """初始化向量存储
        
        Args:
            api_key: Pinecone API 密钥
            environment: Pinecone 环境
            index_name: 索引名称
            dimension: 向量维度
        """
        logger.info('Initializing Pinecone...')
        
        # 初始化 Pinecone
        pinecone.init(
            api_key=api_key,
            environment=environment,
        )
        
        self.index_name = index_name
        self.dimension = dimension
        
        # 创建或获取索引
        self._init_index()
        
        self.index = pinecone.Index(index_name)
        logger.info(f'Connected to index: {index_name}')
    
    def _init_index(self):
        """初始化或获取索引"""
        if self.index_name not in pinecone.list_indexes():
            logger.info(f'Creating index: {self.index_name}')
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
        else:
            logger.info(f'Index already exists: {self.index_name}')
    
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
    ) -> List[str]:
        """添加文本和向量到存储
        
        Args:
            texts: 文本列表
            embeddings: 向量列表
            metadatas: 元数据列表
        
        Returns:
            ID 列表
        """
        logger.info(f'Adding {len(texts)} texts to Pinecone...')
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 准备向量数据
        vectors = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            vector_id = f"doc_{i}_{hash(text) % 10000}"
            metadata['text'] = text
            vectors.append((vector_id, embedding, metadata))
        
        # 批量上传
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading vectors"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f'Successfully added {len(texts)} texts to Pinecone')
        return [v[0] for v in vectors]
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter: Dict = None,
    ) -> List[Tuple[str, float, Dict]]:
        """搜索相似向量
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter: 过滤条件
        
        Returns:
            (ID, 相似度, 元数据) 元组列表
        """
        logger.debug(f'Searching for top {top_k} similar vectors')
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        
        search_results = []
        for match in results['matches']:
            search_results.append((
                match['id'],
                match['score'],
                match['metadata']
            ))
        
        return search_results
    
    def delete_by_id(self, ids: List[str]):
        """删除向量
        
        Args:
            ids: 向量 ID 列表
        """
        logger.info(f'Deleting {len(ids)} vectors')
        self.index.delete(ids=ids)
    
    def delete_all(self):
        """删除所有向量"""
        logger.warning('Deleting all vectors from index')
        # 这会删除整个索引并重新创建
        pinecone.delete_index(self.index_name)
        self._init_index()
    
    def get_stats(self) -> Dict:
        """获取索引统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.get('total_vector_count', 0),
            'dimension': self.dimension,
            'index_name': self.index_name,
        }
