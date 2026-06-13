"""Basic RAG example"""

import logging
from pathlib import Path
from src.rag_chain import RAGChain
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    基础示例：展示如何使用 RAG 系统
    """
    logger.info('Starting basic RAG example...')
    
    # 初始化配置
    config = Config()
    logger.info(f'Configuration: {config}')
    
    # 创建 RAG 链
    rag = RAGChain(config)
    logger.info('RAG Chain created successfully')
    
    # 检查数据目录
    doc_dir = config.DOCUMENTS_DIR
    if not list(doc_dir.glob('*')):
        logger.warning(f'No documents found in {doc_dir}')
        logger.info('Please add some PDF, TXT, or Markdown files to proceed')
        return
    
    # 添加文档
    doc_files = list(doc_dir.glob('*.pdf')) + list(doc_dir.glob('*.txt')) + list(doc_dir.glob('*.md'))
    if doc_files:
        logger.info(f'Adding {len(doc_files)} documents...')
        rag.add_documents([str(f) for f in doc_files])
    
    # 获取系统统计信息
    stats = rag.get_stats()
    logger.info(f'System stats: {stats}')
    
    # 示例查询
    questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "What are the important conclusions?",
    ]
    
    for question in questions:
        logger.info(f'\nQuestion: {question}')
        try:
            answer = rag.query(question)
            logger.info(f'Answer: {answer}')
        except Exception as e:
            logger.error(f'Error: {e}')
    
    logger.info('\nBasic example completed!')

if __name__ == '__main__':
    main()
