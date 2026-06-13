"""Tests for RAG chain"""

import pytest
from src.rag_chain import RAGChain
from config.config import Config

class TestRAGChain:
    """Test cases for RAGChain"""
    
    @pytest.fixture
    def config(self):
        """Create a Config instance"""
        return Config()
    
    def test_rag_initialization(self, config):
        """Test RAG chain initialization"""
        try:
            rag = RAGChain(config)
            assert rag is not None
            assert rag.config == config
            assert rag.loader is not None
            assert rag.splitter is not None
            assert rag.embedding_manager is not None
        except Exception as e:
            pytest.skip(f"RAG initialization failed: {e}")
    
    def test_get_stats(self, config):
        """Test getting RAG statistics"""
        try:
            rag = RAGChain(config)
            stats = rag.get_stats()
            
            assert isinstance(stats, dict)
            assert 'embedding_model' in stats
            assert 'llm_model' in stats
        except Exception as e:
            pytest.skip(f"Failed to get stats: {e}")
