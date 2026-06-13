"""Tests for text splitter"""

import pytest
from langchain.schema import Document
from src.text_splitter import TextSplitter

class TestTextSplitter:
    """Test cases for TextSplitter"""
    
    @pytest.fixture
    def splitter(self):
        """Create a TextSplitter instance"""
        return TextSplitter(chunk_size=100, chunk_overlap=10)
    
    def test_initialization(self, splitter):
        """Test TextSplitter initialization"""
        assert splitter.chunk_size == 100
        assert splitter.chunk_overlap == 10
    
    def test_split_text(self, splitter):
        """Test text splitting"""
        text = "This is a long text. " * 20
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= splitter.chunk_size + 10 for chunk in chunks)
    
    def test_split_documents(self, splitter):
        """Test document splitting"""
        documents = [
            Document(page_content="This is document 1. " * 10),
            Document(page_content="This is document 2. " * 10),
        ]
        
        split_docs = splitter.split(documents)
        
        assert len(split_docs) > len(documents)
        assert all(isinstance(doc, Document) for doc in split_docs)
    
    def test_get_stats(self, splitter):
        """Test getting statistics"""
        documents = [
            Document(page_content="Test " * 20),
        ]
        
        split_docs = splitter.split(documents)
        stats = splitter.get_stats(split_docs)
        
        assert 'num_chunks' in stats
        assert 'total_characters' in stats
        assert 'average_chunk_size' in stats
