import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

class Config:
    """应用配置管理"""
    
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent
    
    # Pinecone 配置
    PINECONE_API_KEY: str = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT: str = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
    PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'rag-index')
    PINECONE_DIMENSION: int = int(os.getenv('PINECONE_DIMENSION', 384))
    
    # Ollama 配置
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama2')
    OLLAMA_TEMPERATURE: float = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))
    
    # Embedding 配置
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_BATCH_SIZE: int = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
    
    # 文本处理配置
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', 500))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', 50))
    TOP_K: int = int(os.getenv('TOP_K', 3))
    
    # 应用配置
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # 目录路径
    DATA_DIR = BASE_DIR / 'data'
    DOCUMENTS_DIR = DATA_DIR / 'documents'
    LOGS_DIR = BASE_DIR / 'logs'
    
    def __init__(self):
        """初始化配置，创建必要的目录"""
        self._create_directories()
        self._setup_logging()
        self._validate_config()
    
    def _create_directories(self):
        """创建必要的目录"""
        for directory in [self.DATA_DIR, self.DOCUMENTS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOGS_DIR / 'app.log'),
                logging.StreamHandler()
            ]
        )
    
    def _validate_config(self):
        """验证配置"""
        logger = logging.getLogger(__name__)
        
        if not self.PINECONE_API_KEY:
            logger.warning('PINECONE_API_KEY not set')
        
        if not self._check_ollama_connection():
            logger.warning(f'Cannot connect to Ollama at {self.OLLAMA_BASE_URL}')
    
    def _check_ollama_connection(self) -> bool:
        """检查 Ollama 连接"""
        try:
            import requests
            response = requests.get(f'{self.OLLAMA_BASE_URL}/api/tags', timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_config_dict(self) -> dict:
        """获取配置字典"""
        return {
            'pinecone_api_key': self.PINECONE_API_KEY,
            'pinecone_environment': self.PINECONE_ENVIRONMENT,
            'pinecone_index_name': self.PINECONE_INDEX_NAME,
            'ollama_base_url': self.OLLAMA_BASE_URL,
            'ollama_model': self.OLLAMA_MODEL,
            'embedding_model': self.EMBEDDING_MODEL,
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'top_k': self.TOP_K,
        }
    
    def __repr__(self) -> str:
        return f'<Config: {self.get_config_dict()}>'
