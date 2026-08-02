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
│   │   ├── api/                    # 前端API
│   │   ├── models/                 # 模型应用
│   │   └── agent_system.py         # agent启动
│   ├── frontend/                   # 前端页面
│   │   ├── __init__.py
│   │   ├── static/                 # 静态文件
│   │   ├── index.html              # 前端页面 
│   ├── models/                     # 模型训练
│   │   ├── __init__.py
│   │   ├── inference/              # 模型推理
│   │   ├── LLM/                    # LLM工具调用
│   │   ├── registry/               # 模型配置
│   │   └── training/               # 模型推理
│   │   └── util/                   # 模型评估
│   ├── service/                    # 章节服务
│   │   ├── __init__.py
│   │   ├── chapter_processor.py    # 上传章节推理
│   │   └── chapter_storage.py      # 上传章节入库
│   ├── tools/                      # 模型工具
│   │   ├── __init__.py
│   │   ├── langchain_tools/        # 模型工具
│   │   ├── retrieval/              # 数据检索
│   │   ├── storage/                # 数据入库
│   │   └── utils.py                # 数据工具
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── file_utils.py           # 文件管理
│       └── config.py               # 配置管理
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后的数据
│   │   ├── splits/                 # 模型推理后数据 
│   ├── vector_databases/           # 向量数据库存储 
│   └── models/                     # 训练好的模型
└──     ├── person_recognition_macBert/            # 实体识别模型
        ├── relation_recognition_macBert/          # 关系抽取模型
        └── shared/                                 # 模型基座
        └── text_recognition_macBert/              # 文本分类模型

## 系统架构图

系统采用五层架构设计，包含离线内容上传链路与在线问答链路两条主要数据流：

```mermaid
flowchart TD
    subgraph 用户交互层
        User[用户提问<br>自然语言问题]
        Upload[上传内容<br>新章节 / 新文档]
    end

    subgraph 上层_Agent调度层
        Agent[Agent调度引擎<br>LangGraph + ReAct]
        Flow[固定流程编排<br>图谱优先 → 片段检索 → 原文兜底]
        State[多轮对话状态管理<br>Human/AI/Tool Message 配对]
        Prompt[阶段提示词<br>控制每个阶段执行行为]
    end

    subgraph 中层_工具与存储层
        Tools[Tools 封装层]
        Tool1[图谱查询工具<br>Cypher 查询]
        Tool2[片段检索工具<br>向量相似度检索]
        Tool3[原文检索工具<br>兜底检索]
        Storage[存储层]
        Neo4j[(知识图谱<br>实体 + 关系 + 片段ID)]
        Chroma1[(片段向量库<br>实体 + 关系 + 片段内容)]
        Chroma2[(原文向量库<br>章节分块 + 原文)]
        UploadTools[入库工具<br>向量写入 / 图谱写入]
    end

    subgraph 底层_模型层
        Base[基座模型<br>chinese_macbert_base]
        Model1[实体识别模型<br>识别人物/地点等实体]
        Model2[关系抽取模型<br>抽取实体间关系]
        Model3[文本分类模型<br>识别文本类型]
        Struct[结构化数据<br>实体 + 关系 + 分类标签]
    end

    subgraph 输出层
        Output[最终回答<br>含证据链 / 可追溯]
    end

    %% 内容上传链路
    Upload --> Model1
    Upload --> Model2
    Upload --> Model3
    Model1 --> Struct
    Model2 --> Struct
    Model3 --> Struct
    Struct --> UploadTools
    UploadTools --> Neo4j
    UploadTools --> Chroma1
    UploadTools --> Chroma2

    %% 问答链路
    User --> Agent
    Agent --> Flow
    Agent --> State
    Agent --> Prompt
    Flow --> Tools
    Tools --> Tool1
    Tools --> Tool2
    Tools --> Tool3
    Tool1 --> Neo4j
    Tool2 --> Chroma1
    Tool3 --> Chroma2
    Neo4j -->|返回实体/关系| Tools
    Chroma1 -->|返回片段内容| Tools
    Chroma2 -->|返回原文内容| Tools
    Tools -->|整合结果| Agent
    Agent --> Output
```