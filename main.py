#!/usr/bin/env python3
"""Main entry point for RAG system"""

import argparse
import logging
from pathlib import Path
from src.rag_chain import RAGChain
from config.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description='RAG (Retrieval-Augmented Generation) System'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query text to search for'
    )
    
    parser.add_argument(
        '--documents',
        type=str,
        nargs='+',
        help='Document paths to add to the system'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all data from the vector store'
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Launch Streamlit web interface'
    )
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    logger.info(f'Config: {config}')
    
    # Initialize RAG
    rag = RAGChain(config)
    logger.info('RAG Chain initialized')
    
    # Handle clear operation
    if args.clear:
        logger.warning('Clearing all data...')
        rag.delete_all_data()
        logger.info('Data cleared')
        return
    
    # Handle document addition
    if args.documents:
        logger.info(f'Adding {len(args.documents)} documents...')
        rag.add_documents(args.documents)
        logger.info('Documents added')
    
    # Handle statistics
    if args.stats:
        stats = rag.get_stats()
        logger.info(f'Statistics: {stats}')
    
    # Handle query
    if args.query:
        logger.info(f'Processing query: {args.query}')
        
        result = rag.query_with_sources(
            args.query,
            return_sources=True
        )
        
        print(f"\n{'='*50}")
        print(f"Question: {args.query}")
        print(f"{'='*50}")
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['sources']:
            print(f"\n{'='*50}")
            print(f"Sources ({len(result['sources'])} found)")
            print(f"{'='*50}")
            for i, source in enumerate(result['sources'], 1):
                print(f"\nSource {i}:")
                print(f"  Content: {source['content']}")
                print(f"  Similarity: {source['similarity_score']:.4f}")
    
    # Handle web interface
    if args.web:
        logger.info('Launching Streamlit web interface...')
        import subprocess
        subprocess.run(['streamlit', 'run', 'app/streamlit_app.py'])

if __name__ == '__main__':
    main()
