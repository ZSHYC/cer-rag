"""全局配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# 路径
PROJECT_ROOT = Path("/home/zshyc/cer")
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma_db"
KG_DIR = DATA_DIR / "knowledge_graph"

# 源数据
LECTURE_DIR = PROJECT_ROOT / "课件"
EXAM_DIR = PROJECT_ROOT / "题目"
EXTRACTED_TXT_DIR = PROJECT_ROOT / "extracted"
REVIEW_MD_DIR = PROJECT_ROOT / "船舶结构物期末复习资料"

# 模型
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# DeepSeek API (兼容 OpenAI 格式)
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# 分块参数
LECTURE_CHUNK_SIZE = 800
LECTURE_CHUNK_OVERLAP = 150
FORMULA_CHUNK_SIZE = 300
FORMULA_CHUNK_OVERLAP = 50
CHILD_CHUNK_SIZE = 300
PARENT_CHUNK_SIZE = 1500

# 检索参数
DENSE_K = 20
SPARSE_K = 20
RRF_K = 60
RERANK_TOP_K = 8
MAX_CONTEXT_TOKENS = 8000

# 对话
MAX_CHAT_HISTORY = 10
MAX_RETRY = 2

# 可观测
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "cer-rag-tutor")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# 课件编号 -> 主题名映射
LECTURE_TOPICS = {
    1: "Introduction to Marine Structures",
    2: "Ship Structural Design Principles",
    3: "Loads and Structural Response",
    4: "Structural Analysis Fundamentals",
    5: "Loads and Hull Girder Response",
    6: "Shear Lag and Structural Response",
    7: "Basics of Finite Element Analysis",
    8: "FEA - Stiffness and Flexibility Methods",
    9: "FEA - Bar and Beam Elements",
    10: "Beam Elements for Ship Frame Analysis",
    11: "Plate Theory Fundamentals",
    12: "Plate Bending and Small Deflection Theory",
    13: "Buckling of Plates",
    14: "Stiffened Panels - Longitudinal and Transverse",
    15: "Column Buckling and Tripping",
    16: "Ultimate Strength of Stiffened Panels",
    17: "Post-Buckling Behaviour",
    18: "Effective Width and Faulkner's Equation",
    19: "Ultimate Bending Strength",
    20: "Hull Girder Strength Part 1",
    21: "Hull Girder Strength Part 2",
}

# 知识点优先级（来自复习路线图）
PRIORITY_MAP = {
    "A": ["三杆桁架刚度法", "板屈曲与有效宽度", "形状系数/塑性截面模数/塑性塌落",
          "甲板板格/车甲板设计", "Rational Design 与传统 Rulebook 对比"],
    "B": ["Hull girder ultimate strength", "Interframe progressive collapse",
          "Effective width 物理意义", "Vibration 影响", "FEA stiffness/beam/bar element 区别"],
    "C": ["Sandwich/GRP sandwich beam", "Mindlin theory vs thin plate theory",
          "Patch load/point load 板挠度", "非对称梁弯曲/能量法"],
}
