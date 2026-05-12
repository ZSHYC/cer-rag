"""元数据映射构建——课件号→主题、试题→考点等"""
import json
from pathlib import Path
from rag.config import DATA_DIR, LECTURE_TOPICS


def build_topic_tag_map() -> dict:
    """构建课件号到主题标签的映射"""
    return {str(k): v for k, v in LECTURE_TOPICS.items()}


def build_exam_knowledge_map() -> dict:
    """从复习笔记提取试卷考点映射（Rule-based + 硬编码）"""
    # 基于复习笔记 01_历年试卷题型索引.md 的整理
    return {
        "2022/23": {
            "Q1": ["Rational Ship Structural Design", "Traditional Rulebook", "Plastic Design", "Elasto-plastic Bending"],
            "Q2": ["Elastic Section Modulus", "Plastic Section Modulus", "Shape Factor", "Plastic Collapse"],
            "Q3": ["Plate Buckling", "Longitudinal Stiffening", "Column Collapse", "Faulkner Effective Width"],
            "Q4": ["Stiffness Matrix", "Pin-jointed Structure", "Displacements", "Reactions"],
        },
        "2023/24": {
            "Q1": ["Plate Structures", "Lightweight", "Stiffness Method", "Plastic Design", "Critical Buckling"],
            "Q2": ["Elastic Section Modulus", "Plastic Section Modulus", "Stiffened Panel", "Plate Thickness"],
            "Q3": ["Deck Plate Buckling", "Faulkner Equation", "Effective Width", "Column Buckling"],
            "Q4": ["Stiffness Matrix", "Pin-jointed Structure"],
        },
        "2024/25": {
            "Q1": ["Failure Modes", "Small Deflection Theory", "Structural Analysis", "Buckling Coefficient"],
            "Q2": ["Car Deck Design", "Flat Bar Thickness", "Plate Thickness", "Design Criteria"],
            "Q3": ["Shape Factor", "Hollow Section", "Column Buckling", "Safety Factor"],
            "Q4": ["Stiffness Matrix", "Displacements", "Reactions", "Bar Extensions"],
        },
    }


def build_all_metadata():
    """构建所有元数据并保存"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "lecture_topics": build_topic_tag_map(),
        "exam_knowledge": build_exam_knowledge_map(),
    }

    output = DATA_DIR / "metadata.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据已保存到: {output}")
    return metadata
