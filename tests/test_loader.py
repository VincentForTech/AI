"""Tests for document loader"""

import pytest
from pathlib import Path
from src.document_loader import DocumentLoader
from langchain.schema import Document

class TestDocumentLoader:
    """Test cases for DocumentLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create a DocumentLoader instance"""
        return DocumentLoader()
    
    def test_supported_formats(self, loader):
        """Test supported file formats"""
        assert '.pdf' in loader.SUPPORTED_FORMATS
        assert '.txt' in loader.SUPPORTED_FORMATS
        assert '.md' in loader.SUPPORTED_FORMATS
        assert '.markdown' in loader.SUPPORTED_FORMATS
    
    def test_load_nonexistent_file(self, loader):
        """Test loading a non-existent file"""
        with pytest.raises(FileNotFoundError):
            loader.load('nonexistent_file.pdf')
    
    def test_load_unsupported_format(self, loader, tmp_path):
        """Test loading an unsupported file format"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")
        
        with pytest.raises(ValueError):
            loader.load(str(test_file))
    
    def test_load_txt_file(self, loader, tmp_path):
        """Test loading a text file"""
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document."
        test_file.write_text(test_content)
        
        documents = loader.load(str(test_file))
        
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
        assert test_content in documents[0].page_content
