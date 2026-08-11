#!/bin/bash

# ====================== 全局配置 ======================
export PYTHONPATH="/app/novelAgent/src:/app/novelAgent:$PYTHONPATH"
echo "===== PYTHONPATH ====="
echo $PYTHONPATH
echo "======================"
# 模型宿主机/容器统一挂载目录
MODEL_ROOT="/app/novelAgent/data/models"

MS_TOKEN="${MODELSCOPE_ACCESS_TOKEN}"

# 三个私有魔塔模型映射（使用普通数组，用冒号分隔）
MODEL_LIST=(
    "gulei8agent/text_recognition_macBert:text_recognition_macBert"
    "gulei8agent/relation_recognition_macBert:relation_recognition_macBert"
    "gulei8agent/person_recognition_macBert:person_recognition_macBert"
)

echo "===== 模型加载模式判断 ====="

# 判断离线模式：三个模型目录全部存在且不为空
all_model_exist=true
for item in "${MODEL_LIST[@]}"; do
    # 分割 repo_id 和 local_dir
    repo_id="${item%%:*}"
    local_dir_name="${item##*:}"
    local_dir="${MODEL_ROOT}/${local_dir_name}"
    
    if [ ! -d "$local_dir" ] || [ -z "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "⚠️ 模型缺失：$repo_id -> $local_dir"
        all_model_exist=false
        break
    fi
done

if [ "$all_model_exist" = true ]; then
    echo "【离线私有化模式】检测到本地完整三套模型，跳过魔塔下载，不访问外网"
    export MODEL_LOAD_MODE="local"
    echo "等待Neo4j图数据库服务启动成功"
    sleep 10
else
    echo "【在线演示模式】缺失本地模型，开始登录魔塔批量下载3套私有模型"
    export MODEL_LOAD_MODE="online"
    
    # Python脚本批量拉取私有模型
    python -c "
import os
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

# 读取容器环境变量Token
ms_token = os.getenv('MODELSCOPE_ACCESS_TOKEN', '')
if not ms_token.strip():
    raise Exception('未配置MODELSCOPE_ACCESS_TOKEN，私有模型无法下载')

# 登录私有仓库
api = HubApi()
api.login(ms_token)

# 定义模型列表
model_map = {
    'gulei8agent/text_recognition_macBert': 'text_recognition_macBert',
    'gulei8agent/relation_recognition_macBert': 'relation_recognition_macBert',
    'gulei8agent/person_recognition_macBert': 'person_recognition_macBert'
}
root_dir = '/app/novelAgent/data/models'

# 循环下载每一个私有模型
for repo_id, folder_name in model_map.items():
    save_path = os.path.join(root_dir, folder_name)
    print(f'开始下载模型：{repo_id} -> {save_path}')
    snapshot_download(
        repo_id,
        local_dir=save_path
    )
    print(f'下载完成：{repo_id}')
print('3个私有模型全部下载完成')
"
fi
# ====================== 等待Neo4j数据库就绪 ======================
echo "===== 等待Neo4j图数据库服务初始化 ====="

# 打印 Neo4j 配置信息（调试用）
echo "🔍 Neo4j 配置信息："
echo "  APP_NEO4J__URL: ${APP_NEO4J__URL:-未设置}"
echo "  APP_NEO4J__USER: ${APP_NEO4J__USER:-未设置}"
echo "  APP_NEO4J__PASSWORD: ${APP_NEO4J__PASSWORD:0:4}******"

# 使用 Python 检查 Neo4j（从环境变量读取）
MAX_RETRIES=30
RETRY_COUNT=0

until python -c "
from neo4j import GraphDatabase
import os

# 从环境变量读取 Neo4j 配置
uri = os.getenv('APP_NEO4J__URL', 'bolt://neo4j:7687')
user = os.getenv('APP_NEO4J__USER', 'neo4j')
password = os.getenv('APP_NEO4J__PASSWORD', 'agent12345')

print(f'连接 Neo4j: {uri}')
print(f'用户名: {user}')

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run('RETURN 1')
    print('Neo4j 连接成功')
    exit(0)
except Exception as e:
    print(f'连接失败: {e}')
    exit(1)
" 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo " Neo4j 启动超时，退出"
        exit 1
    fi
    echo "Neo4j 未就绪，等待 2 秒... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "===== 开始初始化数据库（清空并重建向量库和图数据库） ====="

# 初始化数据库的 Python 脚本
python -c "
from tools.storage.chroma_document_store import ChromaDocumentStore
from tools.storage.chroma_vector_store import ChromaVectorStore
from tools.storage.graph_store import GraphStore

def init_database():
    \"\"\"初始化数据库：清空并重建\"\"\"
    try:
        print('开始清空图数据库...')
        graph_store = GraphStore()
        graph_store.clear_all()
        print('图数据库已清空')
        
        print('开始构建图数据...')
        graph_store.handle_data()
        print('图数据构建完成')
        
        print('开始构建片段级向量数据库...')
        vector_store = ChromaVectorStore()
        vector_store.build_vector_database()
        print('向量数据库构建完成')
        
        print('开始构建文档级向量数据库...')
        doc_store = ChromaDocumentStore()
        doc_store.handle_chapter_file()
        doc_store.build_document_database()
        print('文档数据库构建完成')
        
        print('数据库初始化全部完成！')
        return True
    except Exception as e:
        print(f'❌ 数据库初始化失败: {e}')
        import traceback
        traceback.print_exc()
        return False

# 执行初始化
success = init_database()
if success:
    print('数据库初始化成功')
else:
    print('数据库初始化失败，但继续启动服务')
"
echo "===== 数据库初始化完成 ====="

echo "===== Neo4j数据库就绪，启动FastAPI Agent问答服务 ====="

# ====================== 启动后端服务 ======================
uvicorn src.ai_agent.api.api_main:app --host 0.0.0.0 --port 8000