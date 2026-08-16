import torch
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
from train_discriminator_with_edges import DiscriminatorWithEdges
from training_thinking_gan import Generator
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
import sys
import time
import argparse

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TASK_DEFAULT = "Разработать устройство для управляемого перемещения человека по воздуху."

ALLOWED_NODE_TYPES = {'device', 'part', 'component', 'mechanism', 'system', 'material'}
EXCLUDED_NAME_PATTERNS = ['whirling', 'free flight', 'history', 'experiment', 'test', 'model', 'scale', 'prototype', 'steam-powered', 'моноплан', 'аэроплан']
EDGE_RELEVANCE_THRESHOLD = 0.3
BFS_MAX_DEPTH = 3
MAX_NODES = 8
NUM_CANDIDATES = 60

class KnowledgeGraph:
    def __init__(self):
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.node_names: Dict[str, str] = {}
        self.node_types: Dict[str, str] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)

    def load_from_db(self, embedder):
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)

        # Загружаем узлы, обновляем эмбеддинги, если их нет или размерность не совпадает
        cur.execute("SELECT id, name, node_type, description, embedding FROM agi_evolution.knowledge_nodes")
        for row in cur.fetchall():
            node_id = row['id']
            self.node_names[node_id] = row['name']
            self.node_types[node_id] = row['node_type'].lower()
            if row['embedding'] is not None:
                emb = np.array(row['embedding'])
                # Проверяем размерность (должна быть 384 для all-MiniLM-L6-v2)
                if emb.shape[0] != 384:
                    text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
                    emb = embedder.encode(text)
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    cur.execute("UPDATE agi_evolution.knowledge_nodes SET embedding = %s WHERE id = %s", (emb.tolist(), node_id))
                    conn.commit()
            else:
                text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
                emb = embedder.encode(text)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                cur.execute("UPDATE agi_evolution.knowledge_nodes SET embedding = %s WHERE id = %s", (emb.tolist(), node_id))
                conn.commit()
            self.node_embeddings[node_id] = emb

        # Загружаем рёбра
        cur.execute("""
            SELECT source_id, target_id, edge_type, description, properties
            FROM agi_evolution.knowledge_edges
        """)
        for row in cur.fetchall():
            src = row['source_id']
            tgt = row['target_id']
            self.edges[(src, tgt)] = {
                'type': row['edge_type'],
                'description': row['description'] or '',
                'properties': row['properties'] or {}
            }
            self.edges[(tgt, src)] = self.edges[(src, tgt)]
            self.edge_index[src].add(tgt)
            self.edge_index[tgt].add(src)

        cur.close()
        conn.close()
        print(f"✅ Загружено {len(self.node_embeddings)} узлов, {len(self.edges)//2} рёбер.")

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        return self.node_embeddings.get(node_id)

    def get_node_name(self, node_id: str) -> Optional[str]:
        return self.node_names.get(node_id)

    def get_node_type(self, node_id: str) -> Optional[str]:
        return self.node_types.get(node_id)

    def get_edge_between(self, n1: str, n2: str) -> Optional[Dict]:
        return self.edges.get((n1, n2))

    def get_neighbors(self, node_id: str) -> Set[str]:
        return self.edge_index.get(node_id, set())

    def get_edge_embedding(self, n1: str, n2: str, embedder) -> Optional[np.ndarray]:
        edge = self.get_edge_between(n1, n2)
        if not edge:
            return None
        text = f"{edge['type']}: {edge['description']} {json.dumps(edge['properties'], ensure_ascii=False)}"
        emb = embedder.encode(text)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def get_degree(self, node_id: str) -> int:
        return len(self.edge_index.get(node_id, set()))

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def is_excluded_name(name):
    name_lower = name.lower()
    for pattern in EXCLUDED_NAME_PATTERNS:
        if pattern in name_lower:
            return True
    return False

def compute_edge_features(node_ids, task_emb, kg, embedder):
    if len(node_ids) < 2:
        return np.array([0.0, 0.0, 0.0])
    edge_sims = []
    edge_types = []
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            n1, n2 = node_ids[i], node_ids[j]
            edge_emb = kg.get_edge_embedding(n1, n2, embedder)
            if edge_emb is not None:
                edge_sims.append(cosine_sim(task_emb, edge_emb))
                edge = kg.get_edge_between(n1, n2)
                edge_types.append(edge['type'] if edge else 'unknown')
    edge_sim_mean = np.mean(edge_sims) if edge_sims else 0.0
    edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids)-1) / 2)
    type_diversity = len(set(edge_types)) / max(1, len(edge_types))
    return np.array([edge_sim_mean, edge_density, type_diversity])

def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
                        discriminator, num_candidates: int = NUM_CANDIDATES) -> Tuple[List[str], float, Dict]:
    device = next(generator.parameters()).device
    task_emb = embedder.encode(task_text)
    task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
    task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)

    # Фильтруем узлы
    all_node_ids = []
    for nid in kg.node_embeddings.keys():
        ntype = kg.get_node_type(nid)
        if ntype in ALLOWED_NODE_TYPES:
            name = kg.get_node_name(nid)
            if not is_excluded_name(name):
                all_node_ids.append(nid)

    if not all_node_ids:
        all_node_ids = list(kg.node_embeddings.keys())

    # Проверяем, что все эмбеддинги имеют одинаковую размерность
    try:
        all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
    except ValueError as e:
        print(f"⚠️ Ошибка при создании массива эмбеддингов: {e}", file=sys.stderr)
        print("Попытка исправить: перегенерируем эмбеддинги для проблемных узлов.", file=sys.stderr)
        # В редких случаях может быть, что какой-то эмбеддинг имеет нестандартную форму, пересоздадим их все
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)
        for nid in all_node_ids:
            name = kg.get_node_name(nid)
            desc = ''  # можно не загружать, т.к. мы уже сохранили
            text = name
            emb = embedder.encode(text)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            kg.node_embeddings[nid] = emb
            cur.execute("UPDATE agi_evolution.knowledge_nodes SET embedding = %s WHERE id = %s", (emb.tolist(), nid))
            conn.commit()
        cur.close()
        conn.close()
        all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])

    # Релевантность рёбер и степень
    node_edge_relevance = {}
    node_degree = {}
    for nid in all_node_ids:
        neighbor_sims = []
        for neighbor in kg.get_neighbors(nid):
            edge_emb = kg.get_edge_embedding(nid, neighbor, embedder)
            if edge_emb is not None:
                neighbor_sims.append(cosine_sim(task_emb, edge_emb))
        node_edge_relevance[nid] = np.mean(neighbor_sims) if neighbor_sims else 0.0
        node_degree[nid] = kg.get_degree(nid)

    relevance_values = np.array([node_edge_relevance[nid] for nid in all_node_ids])
    degree_values = np.array([node_degree[nid] for nid in all_node_ids])
    max_degree = np.max(degree_values) if len(degree_values) > 0 else 1
    norm_degree = degree_values / (max_degree + 1e-8)

    best_score = -1.0
    best_combo = None
    best_anchor = None
    best_path = None

    for _ in range(num_candidates):
        noise = torch.randn(1, 64).to(device)
        with torch.no_grad():
            combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
            combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)

        node_sim = np.dot(all_node_embs, combo_emb)
        combined_score = 0.3 * node_sim + 0.6 * relevance_values + 0.1 * norm_degree
        # Берём топ-3 кандидата для выбора якоря
        top_indices = np.argsort(combined_score)[-3:][::-1]
        for idx in top_indices:
            anchor_id = all_node_ids[idx]

            # BFS
            visited = {anchor_id}
            queue = deque([(anchor_id, 0)])
            combo_ids = [anchor_id]
            path = {anchor_id: []}

            while queue and len(combo_ids) < MAX_NODES:
                current_id, dist = queue.popleft()
                if dist >= BFS_MAX_DEPTH:
                    continue
                for neighbor in kg.get_neighbors(current_id):
                    if neighbor in visited or neighbor not in kg.node_embeddings:
                        continue
                    if is_excluded_name(kg.get_node_name(neighbor)):
                        continue
                    edge = kg.get_edge_between(current_id, neighbor)
                    if edge:
                        edge_emb = kg.get_edge_embedding(current_id, neighbor, embedder)
                        if edge_emb is not None:
                            rel = cosine_sim(task_emb, edge_emb)
                            if rel >= EDGE_RELEVANCE_THRESHOLD:
                                visited.add(neighbor)
                                queue.append((neighbor, dist + 1))
                                combo_ids.append(neighbor)
                                path[neighbor] = current_id

            # Если комбинация маленькая, добавить соседей по релевантности
            if len(combo_ids) < 3:
                neighbor_scores = []
                for nid in kg.get_neighbors(anchor_id):
                    if is_excluded_name(kg.get_node_name(nid)):
                        continue
                    edge = kg.get_edge_between(anchor_id, nid)
                    if edge:
                        edge_emb = kg.get_edge_embedding(anchor_id, nid, embedder)
                        if edge_emb is not None:
                            rel = cosine_sim(task_emb, edge_emb)
                            neighbor_scores.append((nid, rel))
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                for nid, _ in neighbor_scores[:3]:
                    if nid not in combo_ids:
                        combo_ids.append(nid)
                        path[nid] = anchor_id

            # Оценка комбинации дискриминатором
            combo_emb_avg = np.mean([kg.get_node_embedding(nid) for nid in combo_ids if nid in kg.node_embeddings], axis=0)
            combo_emb_avg = combo_emb_avg / (np.linalg.norm(combo_emb_avg) + 1e-8)
            combo_emb_t = torch.FloatTensor(combo_emb_avg).unsqueeze(0).to(device)
            edge_feat = compute_edge_features(combo_ids, task_emb, kg, embedder)
            edge_feat_t = torch.FloatTensor(edge_feat).unsqueeze(0).to(device)
            with torch.no_grad():
                score = discriminator(task_emb_t, combo_emb_t, edge_feat_t).item()

            if score > best_score:
                best_score = score
                best_combo = combo_ids
                best_anchor = anchor_id
                best_path = path

    return best_combo, best_score, {'anchor': best_anchor, 'path': best_path}

def main():
    parser = argparse.ArgumentParser(description='Генерация гипотезы для задачи.')
    parser.add_argument('task', nargs='?', default=TASK_DEFAULT, help='Текст задачи')
    args = parser.parse_args()

    task_text = args.task
    print(f"📝 Задача: {task_text}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()
    print("✅ Генератор загружен.")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Модель эмбеддингов загружена.")

    kg = KnowledgeGraph()
    kg.load_from_db(embedder)

    discriminator = DiscriminatorWithEdges().to(device)
    discriminator.load_state_dict(torch.load("discriminator_with_edges.pth", map_location=device))
    discriminator.eval()
    print("✅ Дискриминатор с рёбрами загружен.")

    start_time = time.time()
    hypothesis_ids, score, info = generate_hypothesis(task_text, generator, kg, embedder, discriminator)
    elapsed = time.time() - start_time

    if not hypothesis_ids:
        print("❌ Не удалось сгенерировать гипотезу.")
        return

    print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
    for nid in hypothesis_ids:
        name = kg.get_node_name(nid)
        ntype = kg.get_node_type(nid)
        print(f"  - {name} ({ntype})")

    print(f"\n📊 Оценка дискриминатора: {score:.3f}")
    print(f"⏱️ Время выполнения: {elapsed:.2f} сек.")

    if info.get('anchor'):
        print(f"\n🔍 Якорный узел: {kg.get_node_name(info['anchor'])}")
        if info.get('path'):
            print("🔗 Путь распространения BFS:")
            for node, parent in info['path'].items():
                if parent:
                    print(f"  {kg.get_node_name(parent)} → {kg.get_node_name(node)}")

if __name__ == "__main__":
    main()




# import torch
# import numpy as np
# import psycopg2
# from psycopg2.extras import DictCursor
# from sentence_transformers import SentenceTransformer
# from train_discriminator_with_edges import DiscriminatorWithEdges
# from training_thinking_gan import Generator
# import json
# from typing import List, Dict, Tuple, Optional, Set
# from collections import defaultdict, deque
#
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "postgres",
#     "user": "postgres",
#     "password": "postgres"
# }
#
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# EMBEDDING_DIM = 384
# TASK = "Разработать устройство для управляемого перемещения человека по воздуху."
#
# ALLOWED_NODE_TYPES = {'device', 'part', 'component', 'mechanism', 'system', 'material'}
# EXCLUDED_NAME_PATTERNS = ['whirling', 'free flight', 'history', 'experiment', 'test', 'model', 'scale', 'prototype', 'steam-powered', 'моноплан', 'аэроплан']
# EDGE_RELEVANCE_THRESHOLD = 0.3
# BFS_MAX_DEPTH = 3
# MAX_NODES = 8
# NUM_CANDIDATES = 60
#
#
# class KnowledgeGraph:
#     def __init__(self):
#         self.node_embeddings: Dict[str, np.ndarray] = {}
#         self.node_names: Dict[str, str] = {}
#         self.node_types: Dict[str, str] = {}
#         self.edges: Dict[Tuple[str, str], Dict] = {}
#         self.edge_index: Dict[str, Set[str]] = defaultdict(set)
#         self.missing_embeddings_count = 0
#
#     def load_from_db(self, embedder):
#         conn = psycopg2.connect(**DB_CONFIG)
#         cur = conn.cursor(cursor_factory=DictCursor)
#
#         cur.execute("SELECT id, name, node_type, description, embedding FROM agi_evolution.knowledge_nodes")
#         for row in cur.fetchall():
#             node_id = row['id']
#             self.node_names[node_id] = row['name']
#             self.node_types[node_id] = row['node_type'].lower()
#
#             # Получаем эмбеддинг из БД или генерируем
#             emb = None
#             if row['embedding'] is not None:
#                 emb = np.array(row['embedding'])
#                 # Проверяем размерность
#                 if emb.shape[0] != EMBEDDING_DIM:
#                     emb = None  # неверная размерность – перегенерируем
#
#             if emb is None:
#                 # Генерируем эмбеддинг
#                 text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
#                 emb = embedder.encode(text)
#                 emb = emb / (np.linalg.norm(emb) + 1e-8)
#                 # Сохраняем в БД
#                 cur.execute("UPDATE agi_evolution.knowledge_nodes SET embedding = %s WHERE id = %s", (emb.tolist(), node_id))
#                 conn.commit()
#                 self.missing_embeddings_count += 1
#
#             self.node_embeddings[node_id] = emb
#
#         cur.execute("""
#             SELECT source_id, target_id, edge_type, description, properties
#             FROM agi_evolution.knowledge_edges
#         """)
#         for row in cur.fetchall():
#             src = row['source_id']
#             tgt = row['target_id']
#             self.edges[(src, tgt)] = {
#                 'type': row['edge_type'],
#                 'description': row['description'] or '',
#                 'properties': row['properties'] or {}
#             }
#             self.edges[(tgt, src)] = self.edges[(src, tgt)]
#             self.edge_index[src].add(tgt)
#             self.edge_index[tgt].add(src)
#
#         cur.close()
#         conn.close()
#         print(f"✅ Загружено {len(self.node_embeddings)} узлов, {len(self.edges) // 2} рёбер.")
#         if self.missing_embeddings_count > 0:
#             print(f"   (сгенерировано новых эмбеддингов: {self.missing_embeddings_count})")
#
#     def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
#         return self.node_embeddings.get(node_id)
#
#     def get_node_name(self, node_id: str) -> Optional[str]:
#         return self.node_names.get(node_id)
#
#     def get_node_type(self, node_id: str) -> Optional[str]:
#         return self.node_types.get(node_id)
#
#     def get_edge_between(self, n1: str, n2: str) -> Optional[Dict]:
#         return self.edges.get((n1, n2))
#
#     def get_neighbors(self, node_id: str) -> Set[str]:
#         return self.edge_index.get(node_id, set())
#
#     def get_edge_embedding(self, n1: str, n2: str, embedder) -> Optional[np.ndarray]:
#         edge = self.get_edge_between(n1, n2)
#         if not edge:
#             return None
#         text = f"{edge['type']}: {edge['description']} {json.dumps(edge['properties'], ensure_ascii=False)}"
#         emb = embedder.encode(text)
#         return emb / (np.linalg.norm(emb) + 1e-8)
#
#     def get_degree(self, node_id: str) -> int:
#         return len(self.edge_index.get(node_id, set()))
#
#
# def cosine_sim(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
#
#
# def is_excluded_name(name):
#     name_lower = name.lower()
#     for pattern in EXCLUDED_NAME_PATTERNS:
#         if pattern in name_lower:
#             return True
#     return False
#
#
# def compute_edge_features(node_ids, task_emb, kg, embedder):
#     if len(node_ids) < 2:
#         return np.array([0.0, 0.0, 0.0])
#     edge_sims = []
#     edge_types = []
#     for i in range(len(node_ids)):
#         for j in range(i + 1, len(node_ids)):
#             n1, n2 = node_ids[i], node_ids[j]
#             edge_emb = kg.get_edge_embedding(n1, n2, embedder)
#             if edge_emb is not None:
#                 edge_sims.append(cosine_sim(task_emb, edge_emb))
#                 edge = kg.get_edge_between(n1, n2)
#                 edge_types.append(edge['type'] if edge else 'unknown')
#     edge_sim_mean = np.mean(edge_sims) if edge_sims else 0.0
#     edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids) - 1) / 2)
#     type_diversity = len(set(edge_types)) / max(1, len(edge_types))
#     return np.array([edge_sim_mean, edge_density, type_diversity])
#
#
# def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
#                         discriminator, num_candidates: int = NUM_CANDIDATES) -> Tuple[List[str], float, Dict]:
#     device = next(generator.parameters()).device
#     task_emb = embedder.encode(task_text)
#     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
#     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
#
#     # Фильтруем узлы: только разрешённые типы, исключаем исторические
#     all_node_ids = []
#     for nid, emb in kg.node_embeddings.items():
#         if emb is None:
#             continue
#         if emb.shape[0] != EMBEDDING_DIM:
#             continue
#         ntype = kg.get_node_type(nid)
#         if ntype in ALLOWED_NODE_TYPES:
#             name = kg.get_node_name(nid)
#             if not is_excluded_name(name):
#                 all_node_ids.append(nid)
#
#     if not all_node_ids:
#         print("⚠️ Нет узлов после фильтрации. Используем все доступные.")
#         all_node_ids = [nid for nid, emb in kg.node_embeddings.items() if emb is not None and emb.shape[0] == EMBEDDING_DIM]
#
#     all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
#
#     # Вычисляем релевантность рёбер и степень для каждого узла
#     node_edge_relevance = {}
#     node_degree = {}
#     for nid in all_node_ids:
#         neighbor_sims = []
#         for neighbor in kg.get_neighbors(nid):
#             edge_emb = kg.get_edge_embedding(nid, neighbor, embedder)
#             if edge_emb is not None:
#                 neighbor_sims.append(cosine_sim(task_emb, edge_emb))
#         node_edge_relevance[nid] = np.mean(neighbor_sims) if neighbor_sims else 0.0
#         node_degree[nid] = kg.get_degree(nid)
#
#     relevance_values = np.array([node_edge_relevance[nid] for nid in all_node_ids])
#     degree_values = np.array([node_degree[nid] for nid in all_node_ids])
#     max_degree = np.max(degree_values) if len(degree_values) > 0 else 1
#     norm_degree = degree_values / (max_degree + 1e-8)
#
#     best_score = -1.0
#     best_combo = None
#     best_info = {}
#
#     for _ in range(num_candidates):
#         noise = torch.randn(1, 64).to(device)
#         with torch.no_grad():
#             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
#             combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
#
#         node_sim = np.dot(all_node_embs, combo_emb)
#         combined_score = 0.3 * node_sim + 0.6 * relevance_values + 0.1 * norm_degree
#         anchor_idx = np.argmax(combined_score)
#         anchor_id = all_node_ids[anchor_idx]
#
#         # BFS с фильтром по релевантности рёбер
#         visited = {anchor_id}
#         queue = deque([(anchor_id, 0)])
#         combo_ids = [anchor_id]
#
#         while queue and len(combo_ids) < MAX_NODES:
#             current_id, dist = queue.popleft()
#             if dist >= BFS_MAX_DEPTH:
#                 continue
#             for neighbor in kg.get_neighbors(current_id):
#                 if neighbor in visited or neighbor not in kg.node_embeddings:
#                     continue
#                 if is_excluded_name(kg.get_node_name(neighbor)):
#                     continue
#                 edge = kg.get_edge_between(current_id, neighbor)
#                 if edge:
#                     edge_emb = kg.get_edge_embedding(current_id, neighbor, embedder)
#                     if edge_emb is not None:
#                         rel = cosine_sim(task_emb, edge_emb)
#                         if rel >= EDGE_RELEVANCE_THRESHOLD:
#                             visited.add(neighbor)
#                             queue.append((neighbor, dist + 1))
#                             combo_ids.append(neighbor)
#
#         if len(combo_ids) < 3:
#             neighbor_scores = []
#             for nid in kg.get_neighbors(anchor_id):
#                 if is_excluded_name(kg.get_node_name(nid)):
#                     continue
#                 edge = kg.get_edge_between(anchor_id, nid)
#                 if edge:
#                     edge_emb = kg.get_edge_embedding(anchor_id, nid, embedder)
#                     if edge_emb is not None:
#                         rel = cosine_sim(task_emb, edge_emb)
#                         neighbor_scores.append((nid, rel))
#             neighbor_scores.sort(key=lambda x: x[1], reverse=True)
#             for nid, _ in neighbor_scores[:3]:
#                 if nid not in combo_ids:
#                     combo_ids.append(nid)
#
#         # Оценка комбинации
#         valid_embeddings = [kg.get_node_embedding(nid) for nid in combo_ids if nid in kg.node_embeddings and kg.node_embeddings[nid] is not None]
#         if not valid_embeddings:
#             continue
#         combo_emb_avg = np.mean(valid_embeddings, axis=0)
#         combo_emb_avg = combo_emb_avg / (np.linalg.norm(combo_emb_avg) + 1e-8)
#         combo_emb_t = torch.FloatTensor(combo_emb_avg).unsqueeze(0).to(device)
#         edge_feat = compute_edge_features(combo_ids, task_emb, kg, embedder)
#         edge_feat_t = torch.FloatTensor(edge_feat).unsqueeze(0).to(device)
#         with torch.no_grad():
#             score = discriminator(task_emb_t, combo_emb_t, edge_feat_t).item()
#
#         if score > best_score:
#             best_score = score
#             best_combo = combo_ids
#             best_info = {
#                 'anchor': kg.get_node_name(anchor_id),
#                 'anchor_id': anchor_id,
#                 'score': score,
#                 'edge_feat': edge_feat.tolist(),
#                 'num_nodes': len(combo_ids)
#             }
#
#     return best_combo, best_score, best_info
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Чтение задачи из аргументов командной строки или использование TASK по умолчанию
#     import sys
#     if len(sys.argv) > 1:
#         task_text = ' '.join(sys.argv[1:])
#     else:
#         task_text = TASK
#     print(f"📝 Задача: {task_text}")
#
#     generator = Generator().to(device)
#     generator.load_state_dict(torch.load("generator.pth", map_location=device))
#     generator.eval()
#     print("✅ Генератор загружен.")
#
#     embedder = SentenceTransformer(EMBEDDING_MODEL)
#     print("✅ Модель эмбеддингов загружена.")
#
#     kg = KnowledgeGraph()
#     kg.load_from_db(embedder)
#
#     discriminator = DiscriminatorWithEdges().to(device)
#     discriminator.load_state_dict(torch.load("discriminator_with_edges.pth", map_location=device))
#     discriminator.eval()
#     print("✅ Дискриминатор с рёбрами загружен.")
#
#     hypothesis_ids, score, info = generate_hypothesis(task_text, generator, kg, embedder, discriminator)
#
#     if not hypothesis_ids:
#         print("❌ Не удалось сгенерировать гипотезу.")
#         return
#
#     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
#     for nid in hypothesis_ids:
#         name = kg.get_node_name(nid)
#         ntype = kg.get_node_type(nid)
#         print(f"  - {name} ({ntype})")
#
#     if info:
#         print(f"\n📊 Информация:")
#         print(f"  - Якорный узел: {info.get('anchor')}")
#         print(f"  - Количество узлов: {info.get('num_nodes')}")
#         print(f"  - Оценка дискриминатора: {info.get('score'):.3f}")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
# # import torch
# # import numpy as np
# # import psycopg2
# # from psycopg2.extras import DictCursor
# # from sentence_transformers import SentenceTransformer
# # from train_discriminator_with_edges import DiscriminatorWithEdges
# # from training_thinking_gan import Generator
# # import json
# # from typing import List, Dict, Tuple, Optional, Set
# # from collections import defaultdict, deque
# # import sys
# # import time
# # import argparse
# #
# # DB_CONFIG = {
# #     "host": "localhost",
# #     "database": "agi_evolution",
# #     "user": "postgres",
# #     "password": "your_password"
# # }
# #
# # EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# # TASK_DEFAULT = "Разработать устройство для управляемого перемещения человека по воздуху."
# #
# # ALLOWED_NODE_TYPES = {'device', 'part', 'component', 'mechanism', 'system', 'material'}
# # EXCLUDED_NAME_PATTERNS = ['whirling', 'free flight', 'history', 'experiment', 'test', 'model', 'scale', 'prototype', 'steam-powered', 'моноплан', 'аэроплан']
# # EDGE_RELEVANCE_THRESHOLD = 0.3
# # BFS_MAX_DEPTH = 3
# # MAX_NODES = 8
# # NUM_CANDIDATES = 60
# #
# # class KnowledgeGraph:
# #     def __init__(self):
# #         self.node_embeddings: Dict[str, np.ndarray] = {}
# #         self.node_names: Dict[str, str] = {}
# #         self.node_types: Dict[str, str] = {}
# #         self.edges: Dict[Tuple[str, str], Dict] = {}
# #         self.edge_index: Dict[str, Set[str]] = defaultdict(set)
# #
# #     def load_from_db(self, embedder):
# #         conn = psycopg2.connect(**DB_CONFIG)
# #         cur = conn.cursor(cursor_factory=DictCursor)
# #
# #         # Загружаем узлы, обновляем эмбеддинги, если их нет
# #         cur.execute("SELECT id, name, node_type, description, embedding FROM agi_evolution.knowledge_nodes")
# #         for row in cur.fetchall():
# #             node_id = row['id']
# #             self.node_names[node_id] = row['name']
# #             self.node_types[node_id] = row['node_type'].lower()
# #             if row['embedding'] is not None:
# #                 emb = np.array(row['embedding'])
# #             else:
# #                 text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
# #                 emb = embedder.encode(text)
# #                 emb = emb / (np.linalg.norm(emb) + 1e-8)
# #                 # Сохраняем в БД
# #                 cur.execute("UPDATE agi_evolution.knowledge_nodes SET embedding = %s WHERE id = %s", (emb.tolist(), node_id))
# #                 conn.commit()
# #             self.node_embeddings[node_id] = emb
# #
# #         # Загружаем рёбра
# #         cur.execute("""
# #             SELECT source_id, target_id, edge_type, description, properties
# #             FROM agi_evolution.knowledge_edges
# #         """)
# #         for row in cur.fetchall():
# #             src = row['source_id']
# #             tgt = row['target_id']
# #             self.edges[(src, tgt)] = {
# #                 'type': row['edge_type'],
# #                 'description': row['description'] or '',
# #                 'properties': row['properties'] or {}
# #             }
# #             self.edges[(tgt, src)] = self.edges[(src, tgt)]
# #             self.edge_index[src].add(tgt)
# #             self.edge_index[tgt].add(src)
# #
# #         cur.close()
# #         conn.close()
# #         print(f"✅ Загружено {len(self.node_embeddings)} узлов, {len(self.edges)//2} рёбер.")
# #
# #     def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
# #         return self.node_embeddings.get(node_id)
# #
# #     def get_node_name(self, node_id: str) -> Optional[str]:
# #         return self.node_names.get(node_id)
# #
# #     def get_node_type(self, node_id: str) -> Optional[str]:
# #         return self.node_types.get(node_id)
# #
# #     def get_edge_between(self, n1: str, n2: str) -> Optional[Dict]:
# #         return self.edges.get((n1, n2))
# #
# #     def get_neighbors(self, node_id: str) -> Set[str]:
# #         return self.edge_index.get(node_id, set())
# #
# #     def get_edge_embedding(self, n1: str, n2: str, embedder) -> Optional[np.ndarray]:
# #         edge = self.get_edge_between(n1, n2)
# #         if not edge:
# #             return None
# #         text = f"{edge['type']}: {edge['description']} {json.dumps(edge['properties'], ensure_ascii=False)}"
# #         emb = embedder.encode(text)
# #         return emb / (np.linalg.norm(emb) + 1e-8)
# #
# #     def get_degree(self, node_id: str) -> int:
# #         return len(self.edge_index.get(node_id, set()))
# #
# # def cosine_sim(a, b):
# #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
# #
# # def is_excluded_name(name):
# #     name_lower = name.lower()
# #     for pattern in EXCLUDED_NAME_PATTERNS:
# #         if pattern in name_lower:
# #             return True
# #     return False
# #
# # def compute_edge_features(node_ids, task_emb, kg, embedder):
# #     if len(node_ids) < 2:
# #         return np.array([0.0, 0.0, 0.0])
# #     edge_sims = []
# #     edge_types = []
# #     for i in range(len(node_ids)):
# #         for j in range(i+1, len(node_ids)):
# #             n1, n2 = node_ids[i], node_ids[j]
# #             edge_emb = kg.get_edge_embedding(n1, n2, embedder)
# #             if edge_emb is not None:
# #                 edge_sims.append(cosine_sim(task_emb, edge_emb))
# #                 edge = kg.get_edge_between(n1, n2)
# #                 edge_types.append(edge['type'] if edge else 'unknown')
# #     edge_sim_mean = np.mean(edge_sims) if edge_sims else 0.0
# #     edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids)-1) / 2)
# #     type_diversity = len(set(edge_types)) / max(1, len(edge_types))
# #     return np.array([edge_sim_mean, edge_density, type_diversity])
# #
# # def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
# #                         discriminator, num_candidates: int = NUM_CANDIDATES) -> Tuple[List[str], float, Dict]:
# #     device = next(generator.parameters()).device
# #     task_emb = embedder.encode(task_text)
# #     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
# #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# #
# #     # Фильтруем узлы
# #     all_node_ids = []
# #     for nid in kg.node_embeddings.keys():
# #         ntype = kg.get_node_type(nid)
# #         if ntype in ALLOWED_NODE_TYPES:
# #             name = kg.get_node_name(nid)
# #             if not is_excluded_name(name):
# #                 all_node_ids.append(nid)
# #
# #     if not all_node_ids:
# #         all_node_ids = list(kg.node_embeddings.keys())
# #
# #     all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
# #
# #     # Релевантность рёбер и степень
# #     node_edge_relevance = {}
# #     node_degree = {}
# #     for nid in all_node_ids:
# #         neighbor_sims = []
# #         for neighbor in kg.get_neighbors(nid):
# #             edge_emb = kg.get_edge_embedding(nid, neighbor, embedder)
# #             if edge_emb is not None:
# #                 neighbor_sims.append(cosine_sim(task_emb, edge_emb))
# #         node_edge_relevance[nid] = np.mean(neighbor_sims) if neighbor_sims else 0.0
# #         node_degree[nid] = kg.get_degree(nid)
# #
# #     relevance_values = np.array([node_edge_relevance[nid] for nid in all_node_ids])
# #     degree_values = np.array([node_degree[nid] for nid in all_node_ids])
# #     max_degree = np.max(degree_values) if len(degree_values) > 0 else 1
# #     norm_degree = degree_values / (max_degree + 1e-8)
# #
# #     best_score = -1.0
# #     best_combo = None
# #     best_anchor = None
# #     best_path = None
# #
# #     for _ in range(num_candidates):
# #         noise = torch.randn(1, 64).to(device)
# #         with torch.no_grad():
# #             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
# #             combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
# #
# #         node_sim = np.dot(all_node_embs, combo_emb)
# #         combined_score = 0.3 * node_sim + 0.6 * relevance_values + 0.1 * norm_degree
# #         # Берём топ-3 кандидата для выбора якоря (улучшенный выбор)
# #         top_indices = np.argsort(combined_score)[-3:][::-1]
# #         for idx in top_indices:
# #             anchor_id = all_node_ids[idx]
# #
# #             # BFS
# #             visited = {anchor_id}
# #             queue = deque([(anchor_id, 0)])
# #             combo_ids = [anchor_id]
# #             path = {anchor_id: []}
# #
# #             while queue and len(combo_ids) < MAX_NODES:
# #                 current_id, dist = queue.popleft()
# #                 if dist >= BFS_MAX_DEPTH:
# #                     continue
# #                 for neighbor in kg.get_neighbors(current_id):
# #                     if neighbor in visited or neighbor not in kg.node_embeddings:
# #                         continue
# #                     if is_excluded_name(kg.get_node_name(neighbor)):
# #                         continue
# #                     edge = kg.get_edge_between(current_id, neighbor)
# #                     if edge:
# #                         edge_emb = kg.get_edge_embedding(current_id, neighbor, embedder)
# #                         if edge_emb is not None:
# #                             rel = cosine_sim(task_emb, edge_emb)
# #                             if rel >= EDGE_RELEVANCE_THRESHOLD:
# #                                 visited.add(neighbor)
# #                                 queue.append((neighbor, dist + 1))
# #                                 combo_ids.append(neighbor)
# #                                 path[neighbor] = current_id
# #
# #             # Если комбинация маленькая, добавить соседей по релевантности
# #             if len(combo_ids) < 3:
# #                 neighbor_scores = []
# #                 for nid in kg.get_neighbors(anchor_id):
# #                     if is_excluded_name(kg.get_node_name(nid)):
# #                         continue
# #                     edge = kg.get_edge_between(anchor_id, nid)
# #                     if edge:
# #                         edge_emb = kg.get_edge_embedding(anchor_id, nid, embedder)
# #                         if edge_emb is not None:
# #                             rel = cosine_sim(task_emb, edge_emb)
# #                             neighbor_scores.append((nid, rel))
# #                 neighbor_scores.sort(key=lambda x: x[1], reverse=True)
# #                 for nid, _ in neighbor_scores[:3]:
# #                     if nid not in combo_ids:
# #                         combo_ids.append(nid)
# #                         path[nid] = anchor_id
# #
# #             # Оценка комбинации дискриминатором
# #             combo_emb_avg = np.mean([kg.get_node_embedding(nid) for nid in combo_ids if nid in kg.node_embeddings], axis=0)
# #             combo_emb_avg = combo_emb_avg / (np.linalg.norm(combo_emb_avg) + 1e-8)
# #             combo_emb_t = torch.FloatTensor(combo_emb_avg).unsqueeze(0).to(device)
# #             edge_feat = compute_edge_features(combo_ids, task_emb, kg, embedder)
# #             edge_feat_t = torch.FloatTensor(edge_feat).unsqueeze(0).to(device)
# #             with torch.no_grad():
# #                 score = discriminator(task_emb_t, combo_emb_t, edge_feat_t).item()
# #
# #             if score > best_score:
# #                 best_score = score
# #                 best_combo = combo_ids
# #                 best_anchor = anchor_id
# #                 best_path = path
# #
# #     return best_combo, best_score, {'anchor': best_anchor, 'path': best_path}
# #
# # def main():
# #     parser = argparse.ArgumentParser(description='Генерация гипотезы для задачи.')
# #     parser.add_argument('task', nargs='?', default=TASK_DEFAULT, help='Текст задачи')
# #     args = parser.parse_args()
# #
# #     task_text = args.task
# #     print(f"📝 Задача: {task_text}")
# #
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #     generator = Generator().to(device)
# #     generator.load_state_dict(torch.load("generator.pth", map_location=device))
# #     generator.eval()
# #     print("✅ Генератор загружен.")
# #
# #     embedder = SentenceTransformer(EMBEDDING_MODEL)
# #     print("✅ Модель эмбеддингов загружена.")
# #
# #     kg = KnowledgeGraph()
# #     kg.load_from_db(embedder)
# #
# #     discriminator = DiscriminatorWithEdges().to(device)
# #     discriminator.load_state_dict(torch.load("discriminator_with_edges.pth", map_location=device))
# #     discriminator.eval()
# #     print("✅ Дискриминатор с рёбрами загружен.")
# #
# #     start_time = time.time()
# #     hypothesis_ids, score, info = generate_hypothesis(task_text, generator, kg, embedder, discriminator)
# #     elapsed = time.time() - start_time
# #
# #     if not hypothesis_ids:
# #         print("❌ Не удалось сгенерировать гипотезу.")
# #         return
# #
# #     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
# #     for nid in hypothesis_ids:
# #         name = kg.get_node_name(nid)
# #         ntype = kg.get_node_type(nid)
# #         print(f"  - {name} ({ntype})")
# #
# #     print(f"\n📊 Оценка дискриминатора: {score:.3f}")
# #     print(f"⏱️ Время выполнения: {elapsed:.2f} сек.")
# #
# #     if info.get('anchor'):
# #         print(f"\n🔍 Якорный узел: {kg.get_node_name(info['anchor'])}")
# #         if info.get('path'):
# #             print("🔗 Путь распространения BFS:")
# #             for node, parent in info['path'].items():
# #                 if parent:
# #                     print(f"  {kg.get_node_name(parent)} → {kg.get_node_name(node)}")
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
# #
# # # import torch
# # # import numpy as np
# # # import psycopg2
# # # from psycopg2.extras import DictCursor
# # # from sentence_transformers import SentenceTransformer
# # # from train_discriminator_with_edges import DiscriminatorWithEdges
# # # from training_thinking_gan import Generator
# # # import json
# # # from typing import List, Dict, Tuple, Optional, Set
# # # from collections import defaultdict, deque
# # #
# # # DB_CONFIG = {
# # #     "host": "localhost",
# # #     "database": "postgres",
# # #     "user": "postgres",
# # #     "password": "postgres"
# # # }
# # #
# # # EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# # # TASK = "Разработать устройство для управляемого перемещения человека по воздуху."
# # #
# # # ALLOWED_NODE_TYPES = {'device', 'part', 'component', 'mechanism', 'system', 'material'}
# # # # Список слов для исключения исторических/экспериментальных узлов
# # # EXCLUDED_NAME_PATTERNS = ['whirling', 'free flight', 'history', 'experiment', 'test', 'model', 'scale', 'prototype', 'steam-powered', 'моноплан', 'аэроплан']
# # # EDGE_RELEVANCE_THRESHOLD = 0.3
# # # BFS_MAX_DEPTH = 3
# # # MAX_NODES = 8
# # # NUM_CANDIDATES = 60
# # #
# # # class KnowledgeGraph:
# # #     def __init__(self):
# # #         self.node_embeddings: Dict[str, np.ndarray] = {}
# # #         self.node_names: Dict[str, str] = {}
# # #         self.node_types: Dict[str, str] = {}
# # #         self.edges: Dict[Tuple[str, str], Dict] = {}
# # #         self.edge_index: Dict[str, Set[str]] = defaultdict(set)
# # #
# # #     def load_from_db(self, embedder):
# # #         conn = psycopg2.connect(**DB_CONFIG)
# # #         cur = conn.cursor(cursor_factory=DictCursor)
# # #
# # #         cur.execute("SELECT id, name, node_type, description FROM agi_evolution.knowledge_nodes")
# # #         for row in cur.fetchall():
# # #             node_id = row['id']
# # #             self.node_names[node_id] = row['name']
# # #             self.node_types[node_id] = row['node_type'].lower()
# # #             text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
# # #             emb = embedder.encode(text)
# # #             emb = emb / (np.linalg.norm(emb) + 1e-8)
# # #             self.node_embeddings[node_id] = emb
# # #
# # #         cur.execute("""
# # #             SELECT source_id, target_id, edge_type, description, properties
# # #             FROM agi_evolution.knowledge_edges
# # #         """)
# # #         for row in cur.fetchall():
# # #             src = row['source_id']
# # #             tgt = row['target_id']
# # #             self.edges[(src, tgt)] = {
# # #                 'type': row['edge_type'],
# # #                 'description': row['description'] or '',
# # #                 'properties': row['properties'] or {}
# # #             }
# # #             self.edges[(tgt, src)] = self.edges[(src, tgt)]
# # #             self.edge_index[src].add(tgt)
# # #             self.edge_index[tgt].add(src)
# # #
# # #         cur.close()
# # #         conn.close()
# # #         print(f"✅ Загружено {len(self.node_embeddings)} узлов, {len(self.edges)//2} рёбер.")
# # #
# # #     def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
# # #         return self.node_embeddings.get(node_id)
# # #
# # #     def get_node_name(self, node_id: str) -> Optional[str]:
# # #         return self.node_names.get(node_id)
# # #
# # #     def get_node_type(self, node_id: str) -> Optional[str]:
# # #         return self.node_types.get(node_id)
# # #
# # #     def get_edge_between(self, n1: str, n2: str) -> Optional[Dict]:
# # #         return self.edges.get((n1, n2))
# # #
# # #     def get_neighbors(self, node_id: str) -> Set[str]:
# # #         return self.edge_index.get(node_id, set())
# # #
# # #     def get_edge_embedding(self, n1: str, n2: str, embedder) -> Optional[np.ndarray]:
# # #         edge = self.get_edge_between(n1, n2)
# # #         if not edge:
# # #             return None
# # #         text = f"{edge['type']}: {edge['description']} {json.dumps(edge['properties'], ensure_ascii=False)}"
# # #         emb = embedder.encode(text)
# # #         return emb / (np.linalg.norm(emb) + 1e-8)
# # #
# # #     def get_degree(self, node_id: str) -> int:
# # #         return len(self.edge_index.get(node_id, set()))
# # #
# # # def cosine_sim(a, b):
# # #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
# # #
# # # def is_excluded_name(name):
# # #     """Проверяет, не является ли название узла историческим/экспериментальным."""
# # #     name_lower = name.lower()
# # #     for pattern in EXCLUDED_NAME_PATTERNS:
# # #         if pattern in name_lower:
# # #             return True
# # #     return False
# # #
# # # def compute_edge_features(node_ids, task_emb, kg, embedder):
# # #     if len(node_ids) < 2:
# # #         return np.array([0.0, 0.0, 0.0])
# # #     edge_sims = []
# # #     edge_types = []
# # #     for i in range(len(node_ids)):
# # #         for j in range(i+1, len(node_ids)):
# # #             n1, n2 = node_ids[i], node_ids[j]
# # #             edge_emb = kg.get_edge_embedding(n1, n2, embedder)
# # #             if edge_emb is not None:
# # #                 edge_sims.append(cosine_sim(task_emb, edge_emb))
# # #                 edge = kg.get_edge_between(n1, n2)
# # #                 edge_types.append(edge['type'] if edge else 'unknown')
# # #     edge_sim_mean = np.mean(edge_sims) if edge_sims else 0.0
# # #     edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids)-1) / 2)
# # #     type_diversity = len(set(edge_types)) / max(1, len(edge_types))
# # #     return np.array([edge_sim_mean, edge_density, type_diversity])
# # #
# # # def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
# # #                         discriminator, num_candidates: int = NUM_CANDIDATES) -> Tuple[List[str], float]:
# # #     device = next(generator.parameters()).device
# # #     task_emb = embedder.encode(task_text)
# # #     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
# # #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# # #
# # #     # Фильтруем узлы: только разрешённые типы, исключаем исторические
# # #     all_node_ids = []
# # #     for nid in kg.node_embeddings.keys():
# # #         ntype = kg.get_node_type(nid)
# # #         if ntype in ALLOWED_NODE_TYPES:
# # #             name = kg.get_node_name(nid)
# # #             if not is_excluded_name(name):
# # #                 all_node_ids.append(nid)
# # #
# # #     if not all_node_ids:
# # #         print("⚠️ Нет узлов после фильтрации. Используем все.")
# # #         all_node_ids = list(kg.node_embeddings.keys())
# # #
# # #     all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
# # #
# # #     # Вычисляем релевантность рёбер и степень для каждого узла
# # #     node_edge_relevance = {}
# # #     node_degree = {}
# # #     for nid in all_node_ids:
# # #         neighbor_sims = []
# # #         for neighbor in kg.get_neighbors(nid):
# # #             edge_emb = kg.get_edge_embedding(nid, neighbor, embedder)
# # #             if edge_emb is not None:
# # #                 neighbor_sims.append(cosine_sim(task_emb, edge_emb))
# # #         node_edge_relevance[nid] = np.mean(neighbor_sims) if neighbor_sims else 0.0
# # #         node_degree[nid] = kg.get_degree(nid)
# # #
# # #     relevance_values = np.array([node_edge_relevance[nid] for nid in all_node_ids])
# # #     degree_values = np.array([node_degree[nid] for nid in all_node_ids])
# # #     max_degree = np.max(degree_values) if len(degree_values) > 0 else 1
# # #     norm_degree = degree_values / (max_degree + 1e-8)
# # #
# # #     best_score = -1.0
# # #     best_combo = None
# # #
# # #     for _ in range(num_candidates):
# # #         noise = torch.randn(1, 64).to(device)
# # #         with torch.no_grad():
# # #             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
# # #             combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
# # #
# # #         node_sim = np.dot(all_node_embs, combo_emb)
# # #         # Веса: 30% сходство эмбеддинга, 60% релевантность рёбер, 10% степень
# # #         combined_score = 0.3 * node_sim + 0.6 * relevance_values + 0.1 * norm_degree
# # #         anchor_idx = np.argmax(combined_score)
# # #         anchor_id = all_node_ids[anchor_idx]
# # #
# # #         # BFS с фильтром по релевантности рёбер
# # #         visited = {anchor_id}
# # #         queue = deque([(anchor_id, 0)])
# # #         combo_ids = [anchor_id]
# # #
# # #         while queue and len(combo_ids) < MAX_NODES:
# # #             current_id, dist = queue.popleft()
# # #             if dist >= BFS_MAX_DEPTH:
# # #                 continue
# # #             for neighbor in kg.get_neighbors(current_id):
# # #                 if neighbor in visited or neighbor not in kg.node_embeddings:
# # #                     continue
# # #                 # Проверяем, что узел не исключён по имени
# # #                 if is_excluded_name(kg.get_node_name(neighbor)):
# # #                     continue
# # #                 edge = kg.get_edge_between(current_id, neighbor)
# # #                 if edge:
# # #                     edge_emb = kg.get_edge_embedding(current_id, neighbor, embedder)
# # #                     if edge_emb is not None:
# # #                         rel = cosine_sim(task_emb, edge_emb)
# # #                         if rel >= EDGE_RELEVANCE_THRESHOLD:
# # #                             visited.add(neighbor)
# # #                             queue.append((neighbor, dist + 1))
# # #                             combo_ids.append(neighbor)
# # #
# # #         # Если комбинация слишком маленькая, добавляем топ-соседей якоря
# # #         if len(combo_ids) < 3:
# # #             neighbor_scores = []
# # #             for nid in kg.get_neighbors(anchor_id):
# # #                 if is_excluded_name(kg.get_node_name(nid)):
# # #                     continue
# # #                 edge = kg.get_edge_between(anchor_id, nid)
# # #                 if edge:
# # #                     edge_emb = kg.get_edge_embedding(anchor_id, nid, embedder)
# # #                     if edge_emb is not None:
# # #                         rel = cosine_sim(task_emb, edge_emb)
# # #                         neighbor_scores.append((nid, rel))
# # #             neighbor_scores.sort(key=lambda x: x[1], reverse=True)
# # #             for nid, _ in neighbor_scores[:3]:
# # #                 if nid not in combo_ids:
# # #                     combo_ids.append(nid)
# # #
# # #         # Оценка комбинации
# # #         combo_emb_avg = np.mean([kg.get_node_embedding(nid) for nid in combo_ids if nid in kg.node_embeddings], axis=0)
# # #         combo_emb_avg = combo_emb_avg / (np.linalg.norm(combo_emb_avg) + 1e-8)
# # #         combo_emb_t = torch.FloatTensor(combo_emb_avg).unsqueeze(0).to(device)
# # #         edge_feat = compute_edge_features(combo_ids, task_emb, kg, embedder)
# # #         edge_feat_t = torch.FloatTensor(edge_feat).unsqueeze(0).to(device)
# # #         with torch.no_grad():
# # #             score = discriminator(task_emb_t, combo_emb_t, edge_feat_t).item()
# # #
# # #         if score > best_score:
# # #             best_score = score
# # #             best_combo = combo_ids
# # #
# # #     return best_combo, best_score
# # #
# # # def main():
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # #
# # #     generator = Generator().to(device)
# # #     generator.load_state_dict(torch.load("generator.pth", map_location=device))
# # #     generator.eval()
# # #     print("✅ Генератор загружен.")
# # #
# # #     embedder = SentenceTransformer(EMBEDDING_MODEL)
# # #     print("✅ Модель эмбеддингов загружена.")
# # #
# # #     kg = KnowledgeGraph()
# # #     kg.load_from_db(embedder)
# # #
# # #     discriminator = DiscriminatorWithEdges().to(device)
# # #     discriminator.load_state_dict(torch.load("discriminator_with_edges.pth", map_location=device))
# # #     discriminator.eval()
# # #     print("✅ Дискриминатор с рёбрами загружен.")
# # #
# # #     hypothesis_ids, _ = generate_hypothesis(TASK, generator, kg, embedder, discriminator)
# # #
# # #     if not hypothesis_ids:
# # #         print("❌ Не удалось сгенерировать гипотезу.")
# # #         return
# # #
# # #     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
# # #     for nid in hypothesis_ids:
# # #         name = kg.get_node_name(nid)
# # #         ntype = kg.get_node_type(nid)
# # #         print(f"  - {name} ({ntype})")
# # #
# # #     task_emb = embedder.encode(TASK)
# # #     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
# # #     combo_emb = np.mean([kg.get_node_embedding(nid) for nid in hypothesis_ids if nid in kg.node_embeddings], axis=0)
# # #     combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
# # #     edge_feat = compute_edge_features(hypothesis_ids, task_emb, kg, embedder)
# # #     edge_feat_t = torch.FloatTensor(edge_feat).unsqueeze(0).to(device)
# # #     combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
# # #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# # #
# # #     with torch.no_grad():
# # #         disc_score = discriminator(task_emb_t, combo_emb_t, edge_feat_t).item()
# # #     print(f"📊 Оценка дискриминатора (с рёбрами): {disc_score:.3f}")
# # #
# # # if __name__ == "__main__":
# # #     main()
# # #
