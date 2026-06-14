import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import markdown

logger = logging.getLogger(__name__)

class DocumentLoader:
    """文档加载器，支持多种格式"""
    
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.md', '.markdown'}
    
    def __init__(self):
        """初始化文档加载器"""
        pass
    
    def load(self, file_path: str) -> List[Document]:
        """加载文档
        
        Args:
            file_path: 文件路径
        
        Returns:
            Document 列表
        
        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.txt':
            return self.load_txt(file_path)
        elif suffix in {'.md', '.markdown'}:
            return self.load_markdown(file_path)
        else:
            raise ValueError(f'Unsupported file format: {suffix}')
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文件
        
        Args:
            file_path: PDF 文件路径
        
        Returns:
            Document 列表
        """
        logger.info(f'Loading PDF: {file_path}')
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f'Loaded {len(documents)} pages from PDF')
            return documents
        except Exception as e:
            logger.error(f'Error loading PDF: {e}')
            raise
    
    def load_txt(self, file_path: str) -> List[Document]:
        """加载文本文件
        
        Args:
            file_path: 文本文件路径
        
        Returns:
            Document 列表
        """
        logger.info(f'Loading TXT: {file_path}')
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f'Loaded text file successfully')
            return documents
        except Exception as e:
            logger.error(f'Error loading TXT: {e}')
            raise
    
    def load_markdown(self, file_path: str) -> List[Document]:
        """加载 Markdown 文件
        
        Args:
            file_path: Markdown 文件路径
        
        Returns:
            Document 列表
        """
        logger.info(f'Loading Markdown: {file_path}')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={'source': file_path, 'format': 'markdown'}
            )
            logger.info(f'Loaded markdown file successfully')
            return [doc]
        except Exception as e:
            logger.error(f'Error loading Markdown: {e}')
            raise
    
    def load_directory(self, dir_path: str, pattern: str = '*') -> List[Document]:
        """加载目录下的所有支持的文件
        
        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
        
        Returns:
            Document 列表
        """
        logger.info(f'Loading directory: {dir_path}')
        path = Path(dir_path)
        documents = []
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    docs = self.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f'Failed to load {file_path}: {e}')
        
        logger.info(f'Loaded {len(documents)} documents from directory')
        return documents
