## 系统架构图

本系统采用自下而上的五层架构设计，覆盖从非结构化文本到结构化知识再到可控问答的完整工程链路，
支持在线问答与离线内容入库两条并行数据流。

![垂直领域知识型Agent系统-小说版](./src/frontend/static/images/novel_agent_system.svg?raw=true)
### 系统架构分层描述
第1层：基座与自训练模型层

选用 chinese_macbert_base 作为预训练基座
在基座基础上自训练三个垂直领域NLP模型：

实体识别模型：识别小说中的人物实体
关系抽取模型：抽取实体间的语义关系
文本分类模型：识别输入文本的类型与场景归属
训练策略：LoRA参数高效微调、bert学习率，分层学习率、数据飞轮迭代、过拟合诊断
输入：小说原文（非结构化文本）
输出：结构化数据（实体、关系、分类标签）

第2层：存储层

三层存储架构设计：

知识图谱（Neo4j）：存储实体、关系及片段ID，支持Cypher查询
片段向量库（Chroma）：存储模型推理后的结构化片段及向量表示
原文向量库（Chroma）：存储章节分块及原文内容，用于兜底检索
存储内容来源：自训练模型推理后自动写入 + 用户上传内容后自动写入
入库方式：通过统一的入库工具完成向量写入与图谱写入

第3层：工具封装层

将存储层能力封装为Agent可调用的标准Tools：

图谱查询工具（graph_）：执行Cypher查询，返回实体与关系
片段检索工具（segment_）：执行向量相似度检索，返回片段内容
原文检索工具（document_）：执行原文兜底检索
工具阶段隔离机制：采用阶段前缀命名 + 提示词约束，防止跨阶段调用

第4层：Agent调度层

基于 LangGraph 构建固定流程编排：

执行顺序：知识图谱优先 → 片段检索 → 原文兜底
每个阶段集成 ReAct循环（思考→行动→观察→判断信息充足→进入下一阶段或结束）
每个阶段配置独立 阶段提示词，控制LLM在该阶段的执行行为
多轮对话 状态管理：通过HumanMessage、AIMessage、ToolMessage三种消息类型配对约束，确保上下文不丢失、不混淆

第5层：用户交互层

在线问答链路：用户输入自然语言问题 → Agent调度 → 工具调用 → 检索存储层 → 整合结果 → 返回含证据链的最终回答
离线内容入库链路：用户上传新内容 → 经三个模型推理 → 生成结构化数据 → 自动写入知识图谱与向量库
入库方式：支持主动上传触发入库 + 被动检测自动入库

### 数据流转
#### 1. 离线内容入库流程
```
flowchart TD
    Upload[用户上传新内容] --> Clean[文本清洗]
    Clean --> Classify[文本分类<br>判断内容分类]
    Classify --> NER[实体识别<br>抽取人物实体]
    NER --> RE[关系抽取<br>抽取实体间关系]
    RE --> Struct[结构化数据<br>实体 + 关系 + 分类标签]
    Struct --> Tools[入库工具]
    Tools --> Neo4j[(知识图谱<br>Neo4j)]
    Tools --> Chroma1[(片段向量库<br>Chroma)]
    Tools --> Chroma2[(原文向量库<br>Chroma)]
```
说明：用户上传新内容后，系统自动完成文本清洗，依次经过文本分类、实体识别、关系抽取三个自训练模型推理，生成结构化数据，最终通过入库工具自动写入知识图谱（实体+关系+片段ID）和向量库（片段向量+原文向量），实现知识的持续扩展。入库支持主动上传触发与被动检测自动触发两种模式。
#### 2.在线问答检索流程
```
flowchart TD
    Query[用户提问] --> Agent[Agent调度引擎<br>LangGraph]
    Agent --> Stage1[阶段一：知识图谱查询]
    Stage1 --> Tool1[调用图谱查询工具<br>graph_]
    Tool1 --> Result1[返回实体 + 关系 + 片段ID]
    Result1 --> Judge1{信息是否充足？}
    Judge1 -->|是| Stage2[阶段二：片段向量检索]
    Judge1 -->|否| ReAct1[ReAct循环<br>调整参数迭代]
    ReAct1 --> Judge1
    
    Stage2 --> Tool2[调用片段检索工具<br>segment_]
    Tool2 --> Result2[返回片段内容 + 向量相似度]
    Result2 --> Judge2{信息是否充足？}
    Judge2 -->|是| Stage3[阶段三：原文向量检索]
    Judge2 -->|否| ReAct2[ReAct循环<br>调整参数迭代]
    ReAct2 --> Judge2
    
    Stage3 --> Tool3[调用原文检索工具<br>document_]
    Tool3 --> Result3[返回原文内容]
    Result3 --> Merge[信息整合<br>图谱 + 片段 + 原文]
    Merge --> Output[LLM生成最终回答<br>含证据链]
```
说明：用户自然语言提问后，Agent调度引擎按照“知识图谱优先→片段检索→原文兜底”的固定流程依次执行，每层检索后通过ReAct循环判断信息是否充足，充足则进入下一阶段，不足则重试或降级，最终将三级检索结果整合后交给LLM生成含证据链的最终回答

#### 3.降级兜底机制
```
flowchart TD
    Query[用户提问] --> L1[L1：知识图谱查询]
    L1 --> Check1{有结果？}
    Check1 -->|是| Done[进入下一阶段]
    Check1 -->|否| L2[L2：片段向量检索]
    
    L2 --> Check2{有结果？}
    Check2 -->|是| Done
    Check2 -->|否| L3[L3：原文向量检索]
    
    L3 --> Check3{有结果？}
    Check3 -->|是| Done
    Check3 -->|否| Fallback[返回：<br>未找到相关信息]
    
    Done --> Merge[整合结果]
    Merge --> Output[LLM生成回答]
```
说明：系统采用三级降级策略：优先查询知识图谱获取精确的实体关系；图谱无结果时降级至片段向量库检索相关内容；片段检索仍无结果时，触发原文向量库兜底检索，确保答案有据可查。如果三级检索全部无结果，系统返回“未找到相关信息”，避免大模型产生幻觉回答。

#### 4.ReAct循环控制（每个阶段内）
```
flowchart TD
    Start[进入当前检索阶段] --> Act[执行检索动作<br>调用对应工具]
    Act --> Observe[观察检索结果]
    Observe --> Judge{信息是否充足？}
    Judge -->|是| Next[进入下一阶段]
    Judge -->|否| Think[思考调整策略<br>分析不足原因]
    Think --> Retry[调整参数后迭代]
    Retry --> Limit{迭代次数<br>是否达到上限？}
    Limit -->|否| Act
    Limit -->|是| Force[强制执行<br>进入下一阶段]
    Force --> Next
```
说明：每个检索阶段内部采用ReAct循环（思考-行动-观察）控制执行流程。系统执行检索动作后观察结果，如果信息不足则分析原因、调整查询参数后重试，直到信息充足或达到重试上限。达到上限后强制进入下一阶段，确保流程不会被无限循环阻塞。各阶段独立配置重试上限与判断标准。

## 项目工程结构
```
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
│   └── utils/                      # 系统函数
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
        └── shared/                                # 模型基座
        └── text_recognition_macBert/              # 文本分类模型
```