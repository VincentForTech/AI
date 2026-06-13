import logging
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Embedding 管理器"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """初始化 Embedding 管理器
        
        Args:
            model_name: 模型名称
        """
        logger.info(f'Loading embedding model: {model_name}')
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f'Model loaded. Dimension: {self.dimension}')
    
    def embed_text(self, text: str) -> List[float]:
        """将文本转换为向量
        
        Args:
            text: 输入文本
        
        Returns:
            向量列表
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量将文本转换为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            向量列表
        """
        logger.info(f'Embedding {len(texts)} texts with batch size {batch_size}')
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=False)
        logger.info(f'Embedding complete')
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """获取向量维度
        
        Returns:
            向量维度
        """
        return self.dimension
    
    def get_model_name(self) -> str:
        """获取模型名称
        
        Returns:
            模型名称
        """
        return self.model_name
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个向量的相似度
        
        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量
        
        Returns:
            相似度得分（0-1）
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
