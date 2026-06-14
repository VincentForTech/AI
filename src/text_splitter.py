import logging
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextSplitter:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """初始化文本分块器
        
        Args:
            chunk_size: 每个块的大小（字符数）
            chunk_overlap: 块之间的重叠（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(
            f'TextSplitter initialized: '
            f'chunk_size={chunk_size}, chunk_overlap={chunk_overlap}'
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """分割文档
        
        Args:
            documents: Document 列表
        
        Returns:
            分割后的 Document 列表
        """
        logger.info(f'Splitting {len(documents)} documents...')
        split_docs = self.splitter.split_documents(documents)
        logger.info(f'Split into {len(split_docs)} chunks')
        
        # 添加块索引元数据
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_index'] = i
        
        return split_docs
    
    def split_text(self, text: str) -> List[str]:
        """分割文本
        
        Args:
            text: 原始文本
        
        Returns:
            分割后的文本列表
        """
        logger.debug(f'Splitting text of length {len(text)}')
        chunks = self.splitter.split_text(text)
        logger.debug(f'Split into {len(chunks)} chunks')
        return chunks
    
    def get_stats(self, documents: List[Document]) -> dict:
        """获取分割统计信息
        
        Args:
            documents: Document 列表
        
        Returns:
            统计信息字典
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
        return {
            'num_chunks': len(documents),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(documents) if documents else 0,
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap,
        }
