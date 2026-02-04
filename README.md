# 多模态 VLM + RAG 安全分析系统

本项目实现了一个智能安全分析系统，结合了**视觉语言模型 (VLM)** 与 **检索增强生成 (RAG)** 技术。系统利用 **Chinese-CLIP** 进行图像特征提取，使用 **MiniMind2-V** 进行视觉理解，通过 **Milvus** 向量数据库检索相似的历史事故案例和相关安全法规，提供全面的安全分析和解释。

## 🚀 核心功能

- **以图搜图 (Image-to-Image Retrieval)**: 上传图片，利用高维视觉嵌入 (Chinese-CLIP ViT-B-16) 检索历史上相似的事故案例。
- **多模态 RAG 分析**: 通过检索相关上下文（相似案例 + 法规）增强 VLM 的生成能力，提供准确、基于事实的分析结果。
- **法规匹配**: 根据上下文自动检索相关的安全法规和法律标准。
- **高性能向量检索**: 由 **Milvus** 支持的大规模向量相似度搜索。
- **现代化 API**: 基于 **FastAPI** 构建，提供健壮且可扩展的 RESTful 接口。

## 🛠️ 技术栈

- **核心框架**: Python 3.12+, FastAPI
-   **向量数据库**: Milvus 2.x
-   **模型**:
    -   **Embedding**: Chinese-CLIP (ViT-B-16)
    -   **VLM**: MiniMind2-V
-   **ML 框架**: PyTorch, Transformers
-   **依赖管理**: uv

## 📋 前置条件

开始之前，请确保已安装以下环境：

-   **Python 3.12+**
-   **Milvus 服务**: 运行中的 Milvus 实例（通常通过 Docker 运行）。
    -   [Milvus 安装指南](https://milvus.io/docs/install_standalone-docker.md)
-   **CUDA GPU**: 强烈建议用于模型推理 (Chinese-CLIP & MiniMind2-V)。

## 📦 安装指南

1.  **克隆仓库**
    ```bash
    git clone <repository_url>
    cd my_project
    ```

2.  **安装依赖**
    使用 `uv` (推荐):
    ```bash
    uv sync
    ```
    或者使用标准 `pip`:
    ```bash
    pip install .
    ```

3.  **模型准备**
    确保模型放置在 `config.yaml` 中配置的 `model/` 目录下：
    -   `model/vision_model/chinese-clip-vit-base-patch16`
    -   `model/MiniMind2-V`

## ⚙️ 配置说明

系统通过 `config.yaml` 进行配置。关键设置包括：

-   **Milvus 连接**: `host`, `port`
-   **Embedding 模型**: `model_name`, `dim` (512), `device`
-   **检索设置**: `top_k`, `score_threshold`
-   **VLM 设置**: `model_path`, `generation` 参数

## 🗄️ 数据库初始化

初始化 Milvus 集合并导入数据：

1.  **初始化集合**:
    ```bash
    python scripts/init_milvus_collections.py
    ```

2.  **导入数据 (可选)**:
    如果有数据在 `dataset/` 中，运行：
    ```bash
    python scripts/init_milvus_collections.py --ingest --ingest-kb
    ```

## 🚀 使用方法

### 启动服务器

运行 FastAPI 应用：

```bash
python run.py
```
服务器将在 `http://0.0.0.0:8000` 启动。

### API 文档

服务器启动后，可以访问交互式 API 文档：

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 关键接口

-   **`GET /v1/info`**: 检查系统状态。
-   **`POST /v1/search/image`**: 使用图片搜索相似案例。
-   **`POST /v1/search/image-with-regulations`**: 搜索案例及语义相关的法规。
-   **`POST /v1/analyze`**: 对上传图片执行完整的 RAG + VLM 分析。

## 📂 项目结构

```
my_project/
├── config.yaml          # 系统配置文件
├── run.py               # 应用启动入口
├── pyproject.toml       # Python 依赖管理 (uv)
├── src/                 # 源代码目录
│   ├── api/             # API 路由与 schemas
│   ├── milvus/          # Milvus 客户端与工具
│   ├── services/        # 核心业务逻辑 (AI, Search, System)
│   ├── inference/       # 模型推理逻辑
│   └── main.py          # FastAPI 应用工厂
├── scripts/             # 工具脚本 (初始化, 数据清洗)
├── model/               # 模型权重目录
├── dataset/             # 数据目录
└── tests/               # 单元测试
```
