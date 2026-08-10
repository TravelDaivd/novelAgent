import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


# ========== 环境检测 ==========

def detect_env() -> str:
    """
    自动检测运行环境
    - docker: 在 Docker 容器中运行
    - local: 本地开发
    """
    load_dotenv()
    # 1. 检查环境变量
    env = os.getenv('APP_ENV', '').lower()
    if env in ['docker', 'local']:
        return env

    # 2. 检查是否在 Docker 容器中
    if os.path.exists('/.dockerenv'):
        return 'docker'

    # 3. 检查 cgroup（另一种 Docker 检测方式）
    try:
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                return 'docker'
    except:
        pass

    # 4. 默认本地
    return 'local'


def get_project_root() -> Path:
    """
    获取项目根目录（自动检测）
    通过查找 config 目录的父目录来确定
    """
    # 当前文件所在目录: /path/to/project/config/
    current_file = Path(__file__).resolve()
    # 项目根目录: /path/to/project/
    return current_file.parent.parent


def load_env_file(env: Optional[str] = None) -> None:
    """
    根据环境加载对应的 .env 文件
    - local: 加载 local.env
    - docker: 加载 docker.env
    """
    project_root = get_project_root()

    # 如果没有指定环境，自动检测
    if env is None:
        env = detect_env()

    # 根据环境选择 .env 文件
    env_filename = f"{env}.env"
    env_file = project_root / env_filename

    if env_file.exists():
        load_dotenv(env_file, override=True)
        logging.info(f"加载环境配置: {env_file}")
    else:
        logging.info(f" 配置文件不存在: {env_file}")
        # 尝试加载默认的 .env
        default_env = project_root / '.env'
        if default_env.exists():
            load_dotenv(default_env, override=True)
            logging.info(f" 使用默认配置: {default_env}")
        else:
            logging.info(f" 未找到任何 .env 文件")

# ========== 嵌套配置模型 ==========

class ProjectConfig(BaseModel):
    """项目路径配置（使用相对路径）"""
    # 相对于项目根目录的路径
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    vector_databases_dir: str = "data/vector_databases"
    splits_dir: str = "data/processed/splits"

    # 以下属性在运行时动态计算为绝对路径
    @classmethod
    def create(cls, root: Path, **kwargs) -> "ProjectConfig":
        """创建配置并解析路径"""
        config = cls(**kwargs)
        config._root = root
        return config

    def _resolve(self, path: str) -> str:
        """将相对路径解析为绝对路径"""
        if hasattr(self, '_root'):
            return str(self._root / path)
        return str(Path.cwd() / path)

    @property
    def abs_data_dir(self) -> str:
        return self._resolve(self.data_dir)

    @property
    def abs_raw_dir(self) -> str:
        return self._resolve(self.raw_dir)

    @property
    def abs_processed_dir(self) -> str:
        return self._resolve(self.processed_dir)

    @property
    def abs_models_dir(self) -> str:
        return self._resolve(self.models_dir)

    @property
    def abs_vector_databases_dir(self) -> str:
        return self._resolve(self.vector_databases_dir)

    @property
    def abs_splits_dir(self) -> str:
        return self._resolve(self.splits_dir)


class ModelsConfig(BaseModel):
    """模型配置"""
    shared: str = "shared"
    person_recognition: str = "person_recognition_macBert"
    relation_recognition: str = "relation_recognition_macBert"
    text_recognition: str = "text_recognition_macBert"


class BaseModelsConfig(BaseModel):
    """基础模型配置"""
    chinese_mac_bert: str = "chinese_macBert_base"
    bge_small_zh: str = "bge_small_zh_v1.5"


class TrainingFilesConfig(BaseModel):
    """训练文件配置"""
    person_train: str = "person_train.json"
    relation_train: str = "relation_train.json"
    text_train: str = "text_train.json"
    
    person_validation: str = "person_validation.json"
    relation_validation: str = "relation_validation.json"
    text_validation: str = "text_validation.json"
    
    auto_person: str = "auto_person_marginalia.json"
    auto_relation: str = "auto_relation_marginalia.json"
    auto_text: str = "auto_text_marginalia.json"
    
    appraise_person: str = "appraise_person_data.json"
    appraise_relation: str = "appraise_relation_data.json"
    appraise_text: str = "appraise_text_data.json"


class TrainingConfig(BaseModel):
    """训练配置"""
    manual_dir: str = "manual"
    validation_dir: str = "validation"
    auto_dir: str = "auto"
    appraise_dir: str = "appraise"
    files: TrainingFilesConfig = TrainingFilesConfig()


class VectorConfig(BaseModel):
    """向量数据库配置"""
    chroma_document_data: str = "chroma_document_data.jsonl"
    chroma_segment_data: str = "chroma_vectro_segment_data.jsonl"
    graph_segment_data: str = "graph_segment_data.jsonl"
    
    """chroma 集合"""
    chroma_document_collection: str = "chroma_document_collection"
    chroma_segment_collection: str = "chroma_segment_collection"


class SamplingConfig(BaseModel):
    """采样配置"""
    samples_per_chapter: int = 110
    score_threshold: float = 0.2


class LLMConfig(BaseModel):
    """LLM 配置"""
    deepseek_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: Optional[str] = None


class Neo4jConfig(BaseModel):
    """Neo4j 配置"""
    url: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: Optional[str] = None


class ModelScopeConfig(BaseModel):
    """魔塔配置"""
    access_token: Optional[str] = None
    cache_dir: str = "data/models"


# ========== 主配置 ==========

class AppConfig(BaseSettings):
    """应用主配置"""
    # 环境标识（自动检测）
    env: str = Field(default_factory=detect_env)

    # 项目路径配置
    project: ProjectConfig = ProjectConfig()

    # 其他配置
    models: ModelsConfig = ModelsConfig()
    base_models: BaseModelsConfig = BaseModelsConfig()
    training: TrainingConfig = TrainingConfig()
    vector: VectorConfig = VectorConfig()
    sampling: SamplingConfig = SamplingConfig()
    llm: LLMConfig = LLMConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    modelscope: ModelScopeConfig = ModelScopeConfig()

    # 内部：项目根目录
    _root: Optional[Path] = None

    class Config:
        env_prefix = "APP_"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"

    @classmethod
    def from_yaml(cls, yaml_path: str, root: Path) -> "AppConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 创建配置
        config = cls(**data)
        config._root = root

        # 解析项目路径
        if 'project' in data:
            config.project = ProjectConfig.create(root, **data['project'])

        return config

    @classmethod
    def load(cls, env: Optional[str] = None) -> "AppConfig":
        """
        加载配置
        """
        # 1. 确定环境
        if env is None:
            env = detect_env()
            
        # 2. 加载对应的 .env 文件（关键！）
        load_env_file(env)
        
        # 2. 获取项目根目录
        root = get_project_root()

        # 3. 加载 YAML 配置文件
        config_dir = Path(__file__).parent
        config_file = config_dir / f"config.{env}.yaml"

        if not config_file.exists():
            # 如果配置文件不存在，使用默认配置
            config = cls()
            config._root = root
            config.env = env
            # 使用默认的 ProjectConfig，但传入 root
            config.project = ProjectConfig.create(root)
            return config

        # 4. 从 YAML 加载
        config = cls.from_yaml(str(config_file), root)

        # 5. 环境变量覆盖（Pydantic Settings 自动处理）
        return config

    # ========== 便捷属性（兼容旧 config.py） ==========

    @property
    def PROJECT_ROOT(self) -> str:
        return str(self._root) if self._root else str(Path.cwd())

    @property
    def DATA_DIR(self) -> str:
        return self.project.abs_data_dir

    @property
    def RAW_DATA_DIR(self) -> str:
        return self.project.abs_raw_dir

    @property
    def PROCESSED_DATA_DIR(self) -> str:
        return self.project.abs_processed_dir

    @property
    def MODELS_DATA_DIR(self) -> str:
        return self.project.abs_models_dir

    @property
    def VECTOR_DATABASES_DATA_DIR(self) -> str:
        return self.project.abs_vector_databases_dir

    @property
    def SPLITS_DATA_DIR(self) -> str:
        return self.project.abs_splits_dir

    @property
    def SHARED_DATA_DIR(self) -> str:
        return os.path.join(self.MODELS_DATA_DIR, self.models.shared)

    @property
    def PERSON_MODEL_NAME(self) -> str:
        return self.models.person_recognition

    @property
    def RELATION_MODEL_NAME(self) -> str:
        return self.models.relation_recognition

    @property
    def TEXT_MODEL_NAME(self) -> str:
        return self.models.text_recognition

    @property
    def CHROMA_DOCUMENT_DATA(self) -> str:
        return self.vector.chroma_document_data

    @property
    def CHROMA_SEGMENT_DATA(self) -> str:
        return self.vector.chroma_segment_data

    @property
    def GRAPH_SEGMENT_DATA(self) -> str:
        return self.vector.graph_segment_data

    @property
    def CHROMA_DOCUMENT_COLLECTION(self) -> str:
        return self.vector.chroma_document_collection

    @property
    def CHROMA_SEGMENT_COLLECTION(self) -> str:
        return self.vector.chroma_segment_collection

    @property
    def SAMPLING_CONFIG(self) -> Dict[str, Any]:
        return {
            'samples_per_chapter': self.sampling.samples_per_chapter,
            'score_threshold': self.sampling.score_threshold
        }

    @property
    def DEEPSEEK_API_KEY(self) -> Optional[str]:
        return self.llm.deepseek_api_key

    @property
    def DEEPSEEK_URL(self) -> str:
        return self.llm.deepseek_url

    @property
    def NEO4J_URL(self) -> str:
        return self.neo4j.url

    @property
    def NEO4J_USER(self) -> str:
        return self.neo4j.user

    @property
    def NEO4J_PASSWORD(self) -> Optional[str]:
        return self.neo4j.password

    @property
    def MODELSCOPE_ACCESS_TOKEN(self) -> Optional[str]:
        return self.modelscope.access_token

    @property
    def MODELSCOPE_CACHE_DIR(self) -> str:
        return self.project.abs_models_dir


# ========== 全局配置实例 ==========

_config: Optional[AppConfig] = None

def get_config(env: Optional[str] = None) -> AppConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = AppConfig.load(env)
    return _config


# ========== 兼容旧代码的导入方式 ==========

def __getattr__(name):
    """动态属性访问，兼容旧 config.py"""
    config = get_config()
    if hasattr(config, name):
        return getattr(config, name)
    
    raise AttributeError(f"配置项不存在: {name}")