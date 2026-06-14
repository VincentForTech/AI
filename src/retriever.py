import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class Retriever:
    """检索器"""
    
    def __init__(self, vector_store, embedding_manager, top_k: int = 3):
        """初始化检索器
        
        Args:
            vector_store: 向量存储实例
            embedding_manager: Embedding 管理器实例
            top_k: 返回的结果数量
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k
        logger.info(f'Retriever initialized with top_k={top_k}')
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Document]:
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（可选，默认使用初始化时的值）
        
        Returns:
            Document 列表
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f'Retrieving top {top_k} documents for query: {query[:50]}...')
        
        query_embedding = self.embedding_manager.embed_text(query)
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        documents = []
        for doc_id, score, metadata in search_results:
            text = metadata.pop('text', '')
            doc = Document(
                page_content=text,
                metadata={
                    **metadata,
                    'document_id': doc_id,
                    'similarity_score': score,
                }
            )
            documents.append(doc)
        
        logger.info(f'Retrieved {len(documents)} documents')
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = None,
    ) -> List[tuple]:
        """检索相关文档及其得分
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            (Document, 得分) 元组列表
        """
        if top_k is None:
            top_k = self.top_k
        
        query_embedding = self.embedding_manager.embed_text(query)
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        results = []
        for doc_id, score, metadata in search_results:
            text = metadata.pop('text', '')
            doc = Document(
                page_content=text,
                metadata={
                    **metadata,
                    'document_id': doc_id,
                }
            )
            results.append((doc, score))
        
        return results
