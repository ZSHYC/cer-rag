"""知识图谱构建：概念 ↔ 公式 ↔ 课件 ↔ 考题 ↔ 答案"""
import json
import itertools
from pathlib import Path
from typing import List, Dict, Set, Tuple
from langchain_core.documents import Document
import networkx as nx

from rag.config import DATA_DIR, KG_DIR, CHUNKS_DIR, LECTURE_TOPICS


def build_knowledge_graph(chunks_path: str = None) -> nx.Graph:
    """从 chunks 和元数据构建知识图谱"""
    if chunks_path is None:
        chunks_path = str(CHUNKS_DIR / "all_chunks.jsonl")

    KG_DIR.mkdir(parents=True, exist_ok=True)
    G = nx.Graph()

    # --- 加载实体 ---
    concept_patterns = [
        "rational ship structural design", "rational design",
        "traditional rulebook", "plastic design", "shape factor",
        "plate buckling", "buckling coefficient", "effective width",
        "faulkner", "hull girder", "stiffened panel", "stiffness matrix",
        "pin-jointed", "column buckling", "elasto-plastic bending",
        "ultimate strength", "progressive collapse", "vibration",
        "finite element analysis", "beam element", "bar element",
        "grillage", "plastic collapse", "section modulus",
        "sandwich", "mindlin", "shear lag",
    ]

    formula_patterns = [
        ("shape factor", "φ = Zp/Z = Mp/My"),
        ("elastic buckling", "σcr = kπ²E/[12(1-ν²)]·(t/b)²"),
        ("faulkner effective width", "be = b·(σcr/σy)"),
        ("euler buckling", "Pcr = π²EI/L²"),
        ("section modulus", "Z = I/ymax"),
        ("plastic moment", "Mp = σy·Zp"),
        ("stiffness matrix", "K = (EA/L)·[[c²,cs],[-c²,-cs],[cs,s²],[-cs,-s²],...]"),
        ("grillage collapse", "Pc = n·Mp/L"),
    ]

    # 添加概念节点
    for concept in concept_patterns:
        G.add_node(f"concept:{concept}", type="Concept", label=concept)

    # 添加公式节点
    for name, formula in formula_patterns:
        G.add_node(f"formula:{name}", type="Formula", label=name, expression=formula)

    # 添加课件节点
    for num, topic in LECTURE_TOPICS.items():
        G.add_node(f"lecture:{num}", type="Lecture", label=f"L{num}: {topic}", lecture_num=num)

    # --- 从 chunks 提取考题和答案节点 ---
    exam_questions = []
    with open(chunks_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data["metadata"]
            st = meta.get("source_type", "")
            if st == "exam":
                exam_questions.append(data)
            elif st == "solution":
                exam_questions.append(data)

    # 添加考题节点
    for eq in exam_questions:
        meta = eq["metadata"]
        year = meta.get("exam_year", "unknown")
        qnum = meta.get("question_number", 0)
        node_id = f"exam:{year}:Q{qnum}"
        G.add_node(node_id, type="ExamQuestion", label=f"{year} Q{qnum}",
                    year=year, question_number=qnum)

    # --- 建立关系 ---
    # 概念 → 课件（基于课件主题包含的概念关键词）
    for concept in concept_patterns:
        concept_words = set(concept.split())
        for num, topic in LECTURE_TOPICS.items():
            topic_lower = topic.lower()
            # 如果课件主题包含概念关键词
            if any(w in topic_lower for w in concept_words if len(w) > 3):
                G.add_edge(f"concept:{concept}", f"lecture:{num}",
                           relation="EXPLAINED_IN")

    # 概念 → 考题（基于题目内容包含概念关键词）
    for eq in exam_questions:
        meta = eq["metadata"]
        year = meta.get("exam_year", "")
        qnum = meta.get("question_number", 0)
        node_id = f"exam:{year}:Q{qnum}"
        content = eq["content"].lower()

        for concept in concept_patterns:
            concept_words = set(concept.split())
            if any(w in content for w in concept_words if len(w) > 3):
                if G.has_node(f"concept:{concept}") and G.has_node(node_id):
                    G.add_edge(f"concept:{concept}", node_id, relation="TESTED_BY")

    # 公式 → 概念
    for name, _ in formula_patterns:
        name_lower = name.lower()
        for concept in concept_patterns:
            if name_lower in concept or concept in name_lower:
                G.add_edge(f"formula:{name}", f"concept:{concept}", relation="RELATED_TO")

    print(f"Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Concepts: {len([n for n,d in G.nodes(data=True) if d.get('type')=='Concept'])}")
    print(f"  Formulas: {len([n for n,d in G.nodes(data=True) if d.get('type')=='Formula'])}")
    print(f"  Lectures: {len([n for n,d in G.nodes(data=True) if d.get('type')=='Lecture'])}")
    print(f"  Exams: {len([n for n,d in G.nodes(data=True) if d.get('type')=='ExamQuestion'])}")

    # 保存
    nx.write_gml(G, str(KG_DIR / "knowledge_graph.gml"))
    print(f"Saved to {KG_DIR / 'knowledge_graph.gml'}")

    return G


_kg = None


def get_kg() -> nx.Graph:
    global _kg
    if _kg is None:
        gml_path = KG_DIR / "knowledge_graph.gml"
        if gml_path.exists():
            _kg = nx.read_gml(str(gml_path))
        else:
            _kg = build_knowledge_graph()
    return _kg


def graph_traverse(entity_name: str, relation: str = None, max_depth: int = 2) -> List[dict]:
    """从知识图谱中的实体出发，游走获取关联信息

    返回: [{"node_id": ..., "type": ..., "label": ..., "relation": ..., "distance": ...}]
    """
    G = get_kg()

    # 查找起始节点（模糊匹配）
    start_nodes = []
    entity_lower = entity_name.lower()
    for node_id, data in G.nodes(data=True):
        label = data.get("label", "").lower()
        if entity_lower in label or entity_lower in node_id.lower():
            start_nodes.append(node_id)

    if not start_nodes:
        return []

    results = []
    seen = set()

    for start in start_nodes[:3]:  # 最多3个起始节点
        seen.add(start)
        # BFS 游走
        for depth in range(1, max_depth + 1):
            for node in nx.descendants_at_distance(G, start, depth):
                if node in seen:
                    continue
                seen.add(node)
                data = G.nodes[node]
                # 找到连接边
                edges = list(G.edges([start, node]))
                edge_data = G.get_edge_data(start, node) if G.has_edge(start, node) else {}
                relation_found = edge_data.get("relation", "") if edge_data else ""

                results.append({
                    "node_id": node,
                    "type": data.get("type", ""),
                    "label": data.get("label", ""),
                    "expression": data.get("expression", ""),
                    "year": data.get("year", ""),
                    "relation": relation_found,
                    "distance": depth,
                })

    # 按距离排序
    results.sort(key=lambda x: x["distance"])
    return results
