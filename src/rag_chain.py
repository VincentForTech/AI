import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embedding import EmbeddingManager
from .vector_store import VectorStore
from .retriever import Retriever
from config.config import Config

logger = logging.getLogger(__name__)

class RAGChain:
    """RAG 链的主类"""
    
    def __init__(self, config: Config = None):
        """初始化 RAG 链
        
        Args:
            config: 配置对象
        """
        if config is None:
            config = Config()
        
        self.config = config
        logger.info('Initializing RAG Chain...')
        
        # 初始化各个组件
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.embedding_manager = EmbeddingManager(
            model_name=config.EMBEDDING_MODEL
        )
        self.vector_store = VectorStore(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT,
            index_name=config.PINECONE_INDEX_NAME,
            dimension=self.embedding_manager.get_dimension(),
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            top_k=config.TOP_K,
        )
        
        # 初始化 Ollama LLM
        self.llm = Ollama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=config.OLLAMA_TEMPERATURE,
        )
        
        # 初始化 QA 链
        self._init_qa_chain()
        
        logger.info('RAG Chain initialized successfully')
    
    def _init_qa_chain(self):
        """初始化 QA 链"""
        prompt_template = """You are a helpful assistant answering questions based on the provided documents.

Context from documents:
{context}

Question: {question}

Provide a comprehensive answer based on the context above. If the answer is not in the context, say so.

Answer:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=self.prompt,
        )
    
    def add_documents(
        self,
        doc_paths: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """添加文档到 RAG 系统
        
        Args:
            doc_paths: 文档路径列表
            chunk_size: 分块大小（可选）
            chunk_overlap: 分块重叠（可选）
        """
        logger.info(f'Adding {len(doc_paths)} documents...')
        
        # 如果提供了自定义参数，更新分割器
        if chunk_size is not None or chunk_overlap is not None:
            cs = chunk_size if chunk_size is not None else self.config.CHUNK_SIZE
            co = chunk_overlap if chunk_overlap is not None else self.config.CHUNK_OVERLAP
            self.splitter = TextSplitter(chunk_size=cs, chunk_overlap=co)
        
        # 加载文档
        all_documents = []
        for doc_path in doc_paths:
            docs = self.loader.load(doc_path)
            all_documents.extend(docs)
        
        logger.info(f'Loaded {len(all_documents)} documents')
        
        # 分割文本
        chunks = self.splitter.split(all_documents)
        logger.info(f'Split into {len(chunks)} chunks')
        
        # 生成 embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)
        
        # 准备元数据
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 添加到向量存储
        self.vector_store.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info('Documents added successfully')
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> str:
        """查询问答
        
        Args:
            question: 问题
            top_k: 返回文档数量
        
        Returns:
            答案
        """
        logger.info(f'Processing query: {question}')
        
        # 检索相关文档
        if top_k is None:
            top_k = self.config.TOP_K
        
        documents = self.retriever.retrieve(question, top_k=top_k)
        
        if not documents:
            logger.warning('No relevant documents found')
            return 'No relevant information found in the knowledge base.'
        
        # 生成答案
        logger.info('Generating answer from Ollama...')
        answer = self.qa_chain.run(
            input_documents=documents,
            question=question,
        )
        
        logger.info('Answer generated successfully')
        return answer
    
    def query_with_sources(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """带源文档的查询
        
        Args:
            question: 问题
            top_k: 返回文档数量
            return_sources: 是否返回源文档
        
        Returns:
            包含答案和源的字典
        """
        logger.info(f'Processing query with sources: {question}')
        
        if top_k is None:
            top_k = self.config.TOP_K
        
        # 检索相关文档
        docs_with_scores = self.retriever.retrieve_with_scores(question, top_k=top_k)
        
        if not docs_with_scores:
            return {
                'answer': 'No relevant information found.',
                'sources': [],
            }
        
        documents = [doc for doc, _ in docs_with_scores]
        
        # 生成答案
        answer = self.qa_chain.run(
            input_documents=documents,
            question=question,
        )
        
        # 准备源信息
        sources = []
        if return_sources:
            for doc, score in docs_with_scores:
                sources.append({
                    'content': doc.page_content[:200] + '...',
                    'metadata': doc.metadata,
                    'similarity_score': score,
                })
        
        return {
            'answer': answer,
            'sources': sources,
        }
    
    def delete_all_data(self) -> None:
        """删除所有数据"""
        logger.warning('Deleting all data from vector store')
        self.vector_store.delete_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.vector_store.get_stats()
        stats.update({
            'embedding_model': self.embedding_manager.get_model_name(),
            'llm_model': self.config.OLLAMA_MODEL,
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
        })
        return stats
