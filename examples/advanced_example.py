"""Advanced RAG example with sources"""

import logging
from pathlib import Path
from src.rag_chain import RAGChain
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    高级示例：展示如何使用带源文档的 RAG 系统
    """
    logger.info('Starting advanced RAG example...')
    
    # 初始化配置
    config = Config()
    
    # 创建 RAG 链
    rag = RAGChain(config)
    logger.info('RAG Chain created successfully')
    
    # 检查数据目录
    doc_dir = config.DOCUMENTS_DIR
    if not list(doc_dir.glob('*')):
        logger.warning(f'No documents found in {doc_dir}')
        return
    
    # 添加文档（带自定义参数）
    doc_files = list(doc_dir.glob('*.pdf')) + list(doc_dir.glob('*.txt')) + list(doc_dir.glob('*.md'))
    if doc_files:
        logger.info(f'Adding {len(doc_files)} documents with custom parameters...')
        rag.add_documents(
            [str(f) for f in doc_files],
            chunk_size=800,
            chunk_overlap=100
        )
    
    # 查询与源
    question = "What are the main topics discussed?"
    logger.info(f'\nQuestion: {question}')
    
    try:
        result = rag.query_with_sources(
            question,
            top_k=5,
            return_sources=True
        )
        
        logger.info(f'\nAnswer: {result["answer"]}')
        
        logger.info(f'\nSources ({len(result["sources"])} found):')
        for i, source in enumerate(result['sources'], 1):
            logger.info(f'\n  Source {i}:')
            logger.info(f'    Content: {source["content"]}')
            logger.info(f'    Similarity: {source["similarity_score"]:.4f}')
            logger.info(f'    Metadata: {source["metadata"]}')
    
    except Exception as e:
        logger.error(f'Error: {e}')
    
    logger.info('\nAdvanced example completed!')

if __name__ == '__main__':
    main()
