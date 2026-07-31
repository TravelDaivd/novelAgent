novelAgent/                          # 项目根目录
├── scrapy.cfg                      # ✅ Scrapy配置（放在根目录）
├── requirements.txt                # 项目依赖
├── README.md                       # 项目说明
├── src/                            # 源代码目录
│   ├── __init__.py
│   ├── crawlers/                   # 爬虫模块
│   │   ├── __init__.py
│   │   ├── spiders/                # Scrapy spiders
│   │   │   ├── __init__.py
│   │   │   ├── novel_spider.py     # 小说爬虫
│   │   │   └── base_spider.py      # 基础爬虫类
│   │   ├── pipelines.py            # 数据管道
│   │   └── middlewares.py          # 中间件
│   ├── ai_agent/                   # AI智能体模块
│   │   ├── __init__.py
│   │   ├── models/                 # 模型相关
│   │   ├── training/               # 训练代码
│   │   └── inference.py            # 推理代码
│   ├── knowledge/                  # 知识管理模块
│   │   ├── __init__.py
│   │   ├── graph/                  # 知识图谱
│   │   ├── vector_db.py            # 向量数据库主文件 
│   │   ├── document_loader.py      # 文档加载器
│   │   ├── text_splitter.py        # 文本分割器 
│   │   ├── embedding_manager.py    # 嵌入管理
│   │   └── chroma_manager.py       # ChromaDB 管理器 
│   ├── text_processing/            # 文本处理模块
│   │   ├── __init__.py
│   │   ├── text_sampler.py         # 文本采样器
│   │   ├── entity_extractor.py     # 实体提取器
│   │   ├── chapter_parser.py       # 章节解析器
│   │   └── sentence_splitter.py    # 句子分割器
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── data_cleaner.py         # 数据清洗
│       └── config.py               # 配置管理
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后的数据
│   │   ├── splits/                 # 分割后的文本文件 
│   │   │   ├── novel_01/
│   │   │   │   ├── chapter_01_chunks.json
│   │   │   │   └── ...
│   │   │   └── novel_02/
│   ├── vector_databases/           # 向量数据库存储 
│   │   ├── novel_01_chromadb/      # 每部小说一个数据库
│   │   ├── novel_02_chromadb/
│   │   └── central_chromadb/       # 或一个中央数据库
│   └── models/                     # 训练好的模型
├── tests/                          # 测试代码
├── docs/                           # 文档
└── scripts/                        # 脚本文件
    ├── run_crawler.py              # 运行爬虫
    ├── train_model.py              # 训练模型
    └── start_agent.py              # 启动智能体
    └── run_text_sampling.py        # 战略性采样