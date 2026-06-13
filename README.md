# RAG (检索增强生成) 项目

一个完整的 RAG 系统实现，支持 Pinecone 向量数据库和 Ollama 本地大模型。

## 🌟 功能特性

- 📄 **多格式文档支持**：PDF、TXT、Markdown
- 🔍 **智能分块**：递归字符分割，支持重叠
- 🧠 **向量化**：使用开源 embedding 模型
- 📚 **Pinecone 集成**：云端向量数据库，支持实时更新
- 🤖 **Ollama 本地模型**：支持 Llama2、Mistral 等开源模型
- 🌐 **Web UI**：Streamlit 交互界面
- 🧪 **单元测试**：完整的测试覆盖
- 📊 **日志记录**：详细的操作日志

## 📋 项目结构

```
RAG-Project/
├── config/
│   └── config.py              # 配置管理
├── src/
│   ├── document_loader.py     # 文档加载器
│   ├── text_splitter.py       # 文本分块器
│   ├── embedding.py           # Embedding 处理
│   ├── vector_store.py        # Pinecone 集成
│   ├── retriever.py           # 检索器
│   └── rag_chain.py           # RAG 链
├── app/
│   └── streamlit_app.py       # Web UI
├── examples/
│   ├── basic_example.py       # 基础示例
│   └── advanced_example.py    # 高级示例
├── tests/
│   ├── test_loader.py         # 测试文档加载
│   ├── test_splitter.py       # 测试分块
│   └── test_rag.py            # 测试 RAG
├── data/
│   ├── documents/             # 知识库文件
│   └── sample.pdf             # 示例文件
├── logs/                       # 日志目录
├── .env                        # 环境变量
├── .env.example                # 环境变量示例
├── requirements.txt            # Python 依赖
└── main.py                     # 主程序入口
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Ollama 已安装并运行
- Pinecone 账户

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置：

```env
# Pinecone 配置
PINCONE_API_KEY=your_pinecone_api_key
PINCONE_ENVIRONMENT=us-west1-gcp
PINCONE_INDEX_NAME=rag-index

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Embedding 配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 应用配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
LOG_LEVEL=INFO
```

### 4. 启动 Ollama

```bash
# 首先安装 Ollama（如果还没有）
# https://ollama.ai

# 启动 Ollama 服务
ollama serve

# 在另一个终端拉取模型
ollama pull llama2
```

### 5. 运行基础示例

```bash
python examples/basic_example.py
```

### 6. 启动 Web UI

```bash
streamlit run app/streamlit_app.py
```

然后访问 `http://localhost:8501`

## 📖 使用示例

### 基础用法

```python
from src.rag_chain import RAGChain
from config.config import Config

# 初始化配置
config = Config()

# 创建 RAG 链
rag = RAGChain(config)

# 添加文档
rag.add_documents(["path/to/document.pdf"])

# 提问
response = rag.query("什么是机器学习？")
print(response)
```

### 高级用法

```python
from src.rag_chain import RAGChain
from config.config import Config

config = Config()
rag = RAGChain(config)

# 自定义分块参数
rag.add_documents(
    ["path/to/document.pdf"],
    chunk_size=1000,
    chunk_overlap=100
)

# 获取检索结果和答案
results = rag.query_with_sources(
    "你的问题",
    return_sources=True,
    top_k=5
)

print(f"答案: {results['answer']}")
print(f"来源: {results['sources']}")
```

## 🔧 核心 API

### RAGChain

```python
class RAGChain:
    def __init__(self, config: Config)
    def add_documents(self, doc_paths: List[str], **kwargs) -> None
    def query(self, question: str, top_k: int = 3) -> str
    def query_with_sources(self, question: str, **kwargs) -> Dict
    def delete_all_data(self) -> None
    def get_stats(self) -> Dict
```

### DocumentLoader

```python
class DocumentLoader:
    def load_pdf(self, file_path: str) -> List[Document]
    def load_txt(self, file_path: str) -> List[Document]
    def load_markdown(self, file_path: str) -> List[Document]
    def load(self, file_path: str) -> List[Document]
    def load_directory(self, dir_path: str) -> List[Document]
```

## 🧪 运行测试

```bash
python -m pytest tests/ -v
```

## 🎯 性能调优

### 1. 优化分块大小

- 太小（<200）：检索精度高但数据库大
- 太大（>1000）：检索速度快但精度低
- 推荐：500 字符

### 2. 优化 Top-K 值

- Top-3：快速响应，可能信息不足
- Top-5：平衡性能和质量
- Top-10：更多上下文，但可能引入噪音

### 3. 更换 Embedding 模型

```python
# 更快的模型
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 更准确的模型（更慢）
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

## 🐛 常见问题

### Q: Ollama 连接不上

```bash
# 确保 Ollama 已启动
ollama serve

# 检查是否可以连接
curl http://localhost:11434/api/tags
```

### Q: Pinecone 超配额

- 检查 Pinecone 免费层限制
- 删除旧索引：`rag.delete_all_data()`
- 考虑本地向量库（FAISS）作为替代

### Q: 内存占用过高

- 减小 `CHUNK_SIZE`
- 使用更小的 embedding 模型
- 分批处理大型文档

## 📦 依赖说明

| 包名 | 版本 | 用途 |
|------|------|------|
| langchain | >=0.1.0 | LLM 框架 |
| pinecone-client | >=3.0.0 | Pinecone 集成 |
| ollama | >=0.1.0 | Ollama 集成 |
| sentence-transformers | >=2.2.0 | Embedding 模型 |
| pypdf | >=3.0.0 | PDF 解析 |
| streamlit | >=1.28.0 | Web UI |
| python-dotenv | >=1.0.0 | 环境变量管理 |

## 📝 日志示例

```
2024-06-13 10:30:00 INFO - 初始化 RAG 系统
2024-06-13 10:30:01 INFO - 加载文档: sample.pdf (2500 字符)
2024-06-13 10:30:02 INFO - 文本分块完成: 5 个块
2024-06-13 10:30:05 INFO - 向量化完成: 使用模型 all-MiniLM-L6-v2
2024-06-13 10:30:10 INFO - 已上传到 Pinecone: 5 个向量
2024-06-13 10:30:15 INFO - 查询: "什么是机器学习？"
2024-06-13 10:30:17 INFO - 检索到 3 个相关文档
2024-06-13 10:30:22 INFO - 从 Ollama 生成答案 (耗时 5.2s)
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交 Issue 或联系维护者。
