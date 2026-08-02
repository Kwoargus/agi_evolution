import torch
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
from training_thinking_gan import Generator, Discriminator
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# ============================================================
# Конфигурация
# ============================================================
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TASK = "Разработать устройство для управляемого перемещения человека по воздуху."

# ============================================================
# Класс графа знаний
# ============================================================
class KnowledgeGraph:
    def __init__(self):
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.node_names: Dict[str, str] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)

    def load_from_db(self, embedder):
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)

        cur.execute("SELECT id, name, description FROM agi_evolution.knowledge_nodes")
        for row in cur.fetchall():
            node_id = row['id']
            self.node_names[node_id] = row['name']
            text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
            emb = embedder.encode(text)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self.node_embeddings[node_id] = emb

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

# ============================================================
# Основные функции
# ============================================================
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def evaluate_combination(node_ids: List[str], task_embedding: np.ndarray, kg: KnowledgeGraph, embedder) -> float:
    if len(node_ids) < 2:
        return 0.0

    node_sims = []
    for nid in node_ids:
        emb = kg.get_node_embedding(nid)
        if emb is not None:
            node_sims.append(cosine_sim(task_embedding, emb))
    node_sim = np.mean(node_sims) if node_sims else 0.0

    edge_sims = []
    edge_types = []
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            n1, n2 = node_ids[i], node_ids[j]
            edge_emb = kg.get_edge_embedding(n1, n2, embedder)
            if edge_emb is not None:
                edge_sims.append(cosine_sim(task_embedding, edge_emb))
                edge = kg.get_edge_between(n1, n2)
                edge_types.append(edge['type'] if edge else 'unknown')
    edge_sim = np.mean(edge_sims) if edge_sims else 0.0
    edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids)-1) / 2)
    type_diversity = len(set(edge_types)) / max(1, len(edge_types))

    score = (0.3 * node_sim +
             0.4 * edge_sim +
             0.2 * edge_density +
             0.1 * type_diversity)
    return score

def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
                        num_candidates: int = 30, top_k: int = 10) -> Tuple[List[str], float]:
    device = next(generator.parameters()).device
    task_emb = embedder.encode(task_text)
    task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
    task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)

    # --- Вычисляем релевантность каждого узла через его рёбра ---
    node_edge_relevance = {}
    for nid in kg.node_embeddings:
        edge_sims = []
        for neighbor in kg.get_neighbors(nid):
            edge_emb = kg.get_edge_embedding(nid, neighbor, embedder)
            if edge_emb is not None:
                edge_sims.append(cosine_sim(task_emb, edge_emb))
        node_edge_relevance[nid] = np.mean(edge_sims) if edge_sims else 0.0

    all_node_ids = list(kg.node_embeddings.keys())
    all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
    relevance_values = np.array([node_edge_relevance[nid] for nid in all_node_ids])

    best_score = -1.0
    best_combo = None

    for _ in range(num_candidates):
        noise = torch.randn(1, 64).to(device)
        with torch.no_grad():
            combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
            combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)

        # Сходство узлов с combo_emb
        node_sim = np.dot(all_node_embs, combo_emb)
        # Взвешенная сумма: 60% сходство узлов, 40% релевантность рёбер
        combined_score = 0.6 * node_sim + 0.4 * relevance_values
        top_indices = np.argsort(combined_score)[-top_k:][::-1]
        candidate_ids = [all_node_ids[i] for i in top_indices]

        score = evaluate_combination(candidate_ids, task_emb, kg, embedder)
        if score > best_score:
            best_score = score
            best_combo = candidate_ids

    return best_combo, best_score

# ============================================================
# Основной скрипт
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()
    print("✅ Генератор загружен.")

    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
    discriminator.eval()

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Модель эмбеддингов загружена.")

    kg = KnowledgeGraph()
    kg.load_from_db(embedder)

    hypothesis_ids, score = generate_hypothesis(TASK, generator, kg, embedder,
                                                num_candidates=50, top_k=12)

    if not hypothesis_ids:
        print("❌ Не удалось сгенерировать гипотезу.")
        return

    print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
    for nid in hypothesis_ids:
        name = kg.get_node_name(nid)
        print(f"  - {name} ({nid[:8]}...)")

    print(f"\n📊 Оценка (с учётом рёбер): {score:.3f}")

    # Старая оценка дискриминатора
    combo_emb = np.mean([kg.get_node_embedding(nid) for nid in hypothesis_ids if nid in kg.node_embeddings], axis=0)
    combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
    task_emb = embedder.encode(TASK)
    task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
    combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
    task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
    with torch.no_grad():
        disc_score = discriminator(task_emb_t, combo_emb_t).item()
    print(f"📊 Оценка дискриминатора (старая): {disc_score:.3f}")

if __name__ == "__main__":
    main()




# import torch
# import numpy as np
# import psycopg2
# from psycopg2.extras import DictCursor
# from sentence_transformers import SentenceTransformer
# from training_thinking_gan import Generator, Discriminator
# import json
# from typing import List, Dict, Tuple, Optional, Set
# from collections import defaultdict
#
# # ============================================================
# # Конфигурация
# # ============================================================
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "postgres",
#     "user": "postgres",
#     "password": "postgres"
# }
#
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# TASK = "Разработать устройство для управляемого перемещения человека по воздуху."
#
# # ============================================================
# # Класс графа знаний (упрощённый, только для чтения)
# # ============================================================
# class KnowledgeGraph:
#     def __init__(self):
#         self.node_embeddings: Dict[str, np.ndarray] = {}
#         self.node_names: Dict[str, str] = {}
#         self.edges: Dict[Tuple[str, str], Dict] = {}  # (source, target) -> {type, desc, props}
#         self.edge_index: Dict[str, Set[str]] = defaultdict(set)  # node_id -> соседние node_id
#
#     def load_from_db(self, embedder):
#         """Загружает узлы и рёбра из БД, генерирует эмбеддинги для узлов и рёбер."""
#         conn = psycopg2.connect(**DB_CONFIG)
#         cur = conn.cursor(cursor_factory=DictCursor)
#
#         # --- Загрузка узлов ---
#         cur.execute("SELECT id, name, description FROM agi_evolution.knowledge_nodes")
#         for row in cur.fetchall():
#             node_id = row['id']
#             self.node_names[node_id] = row['name']
#             text = f"{row['name']}: {row['description']}" if row['description'] else row['name']
#             emb = embedder.encode(text)
#             emb = emb / (np.linalg.norm(emb) + 1e-8)
#             self.node_embeddings[node_id] = emb
#
#         # --- Загрузка рёбер ---
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
#             self.edges[(tgt, src)] = self.edges[(src, tgt)]  # симметрия для удобства
#             self.edge_index[src].add(tgt)
#             self.edge_index[tgt].add(src)
#
#         cur.close()
#         conn.close()
#         print(f"✅ Загружено {len(self.node_embeddings)} узлов, {len(self.edges)//2} рёбер.")
#
#     def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
#         return self.node_embeddings.get(node_id)
#
#     def get_node_name(self, node_id: str) -> Optional[str]:
#         return self.node_names.get(node_id)
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
# # ============================================================
# # Основные функции оценки и генерации
# # ============================================================
# def cosine_sim(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
#
# def evaluate_combination(node_ids: List[str], task_embedding: np.ndarray, kg: KnowledgeGraph, embedder) -> float:
#     """
#     Оценивает комбинацию узлов на основе:
#     - среднего сходства узлов с задачей
#     - среднего сходства рёбер между узлами с задачей
#     - доли связанных пар
#     - разнообразия типов связей
#     """
#     if len(node_ids) < 2:
#         return 0.0
#
#     # 1. Сходство узлов
#     node_sims = []
#     for nid in node_ids:
#         emb = kg.get_node_embedding(nid)
#         if emb is not None:
#             node_sims.append(cosine_sim(task_embedding, emb))
#     node_sim = np.mean(node_sims) if node_sims else 0.0
#
#     # 2. Сходство рёбер
#     edge_sims = []
#     edge_types = []
#     for i in range(len(node_ids)):
#         for j in range(i+1, len(node_ids)):
#             n1, n2 = node_ids[i], node_ids[j]
#             edge_emb = kg.get_edge_embedding(n1, n2, embedder)
#             if edge_emb is not None:
#                 edge_sims.append(cosine_sim(task_embedding, edge_emb))
#                 edge = kg.get_edge_between(n1, n2)
#                 edge_types.append(edge['type'] if edge else 'unknown')
#     edge_sim = np.mean(edge_sims) if edge_sims else 0.0
#     edge_density = len(edge_sims) / (len(node_ids) * (len(node_ids)-1) / 2)
#     type_diversity = len(set(edge_types)) / max(1, len(edge_types))
#
#     # Итоговая оценка (веса подбираются эмпирически)
#     score = (0.3 * node_sim +
#              0.4 * edge_sim +
#              0.2 * edge_density +
#              0.1 * type_diversity)
#     return score
#
# def generate_hypothesis(task_text: str, generator, kg: KnowledgeGraph, embedder,
#                         num_candidates: int = 30, top_k: int = 10) -> Tuple[List[str], float]:
#     """
#     Генерирует гипотезу, выбирая кандидата с максимальной оценкой, учитывающей рёбра.
#     """
#     device = next(generator.parameters()).device
#     task_emb = embedder.encode(task_text)
#     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
#     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
#
#     best_score = -1.0
#     best_combo = None
#     all_node_ids = list(kg.node_embeddings.keys())
#     all_node_embs = np.array([kg.node_embeddings[nid] for nid in all_node_ids])
#
#     for _ in range(num_candidates):
#         # Генерируем эмбеддинг комбинации
#         noise = torch.randn(1, 64).to(device)
#         with torch.no_grad():
#             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
#             combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
#
#         # Находим ближайшие узлы
#         sims = np.dot(all_node_embs, combo_emb)
#         top_indices = np.argsort(sims)[-top_k:][::-1]
#         candidate_ids = [all_node_ids[i] for i in top_indices]
#
#         # Оцениваем кандидата
#         score = evaluate_combination(candidate_ids, task_emb, kg, embedder)
#         if score > best_score:
#             best_score = score
#             best_combo = candidate_ids
#
#     return best_combo, best_score
#
# # ============================================================
# # Основной скрипт
# # ============================================================
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Загружаем модели GAN
#     generator = Generator().to(device)
#     generator.load_state_dict(torch.load("generator.pth", map_location=device))
#     generator.eval()
#     print("✅ Генератор загружен.")
#
#     discriminator = Discriminator().to(device)
#     discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
#     discriminator.eval()
#
#     # Инициализируем эмбеддер
#     embedder = SentenceTransformer(EMBEDDING_MODEL)
#     print("✅ Модель эмбеддингов загружена.")
#
#     # Загружаем граф знаний
#     kg = KnowledgeGraph()
#     kg.load_from_db(embedder)
#
#     # Генерация гипотезы
#     hypothesis_ids, score = generate_hypothesis(TASK, generator, kg, embedder,
#                                                 num_candidates=30, top_k=10)
#
#     if not hypothesis_ids:
#         print("❌ Не удалось сгенерировать гипотезу.")
#         return
#
#     # Вывод результатов
#     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
#     for nid in hypothesis_ids:
#         name = kg.get_node_name(nid)
#         print(f"  - {name} ({nid[:8]}...)")
#
#     print(f"\n📊 Оценка (с учётом рёбер): {score:.3f}")
#
#     # Дополнительно: оценка дискриминатора (для сравнения)
#     combo_emb = np.mean([kg.get_node_embedding(nid) for nid in hypothesis_ids if nid in kg.node_embeddings], axis=0)
#     combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
#     task_emb = embedder.encode(TASK)
#     task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
#     combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
#     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
#     with torch.no_grad():
#         disc_score = discriminator(task_emb_t, combo_emb_t).item()
#     print(f"📊 Оценка дискриминатора (старая): {disc_score:.3f}")
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
# # from training_thinking_gan import Generator, Discriminator
# #
# # DB_CONFIG = {
# #     "host": "localhost",
# #     "database": "postgres",
# #     "user": "postgres",
# #     "password": "postgres"
# # }
# #
# # # Загружаем модель эмбеддингов один раз
# # embedder = SentenceTransformer('all-MiniLM-L6-v2')
# #
# # def get_embedding(text):
# #     emb = embedder.encode(text)
# #     emb = np.array(emb)
# #     norm = np.linalg.norm(emb) + 1e-8
# #     return emb / norm
# #
# # def load_node_embeddings_from_db():
# #     """
# #     Загружает все узлы из БД и генерирует эмбеддинги на основе имени и описания.
# #     (Если в БД уже есть эмбеддинги, можно было бы использовать их, но для простоты генерируем заново)
# #     """
# #     conn = psycopg2.connect(**DB_CONFIG)
# #     cur = conn.cursor(cursor_factory=DictCursor)
# #     cur.execute("SELECT id, name, description FROM agi_evolution.knowledge_nodes")
# #     node_embeds = {}
# #     for row in cur.fetchall():
# #         node_id = row['id']
# #         name = row['name']
# #         description = row['description'] or ''
# #         text = f"{name}: {description}" if description else name
# #         emb = get_embedding(text)
# #         node_embeds[node_id] = emb
# #     cur.close()
# #     conn.close()
# #     return node_embeds
# #
# # def load_edges_from_db():
# #     """Загружает рёбра как множество пар (source_id, target_id)."""
# #     conn = psycopg2.connect(**DB_CONFIG)
# #     cur = conn.cursor()
# #     cur.execute("SELECT source_id, target_id FROM agi_evolution.knowledge_edges")
# #     edges = set()
# #     for row in cur.fetchall():
# #         edges.add((row[0], row[1]))
# #     cur.close()
# #     conn.close()
# #     return edges
# #
# # def compute_edge_ratio(node_ids, edges):
# #     """Доля пар узлов, между которыми есть ребро (в любом направлении)."""
# #     if len(node_ids) < 2:
# #         return 0.0
# #     node_set = set(node_ids)
# #     total_pairs = len(node_ids) * (len(node_ids) - 1) // 2
# #     linked_pairs = 0
# #     for i in range(len(node_ids)):
# #         for j in range(i+1, len(node_ids)):
# #             if (node_ids[i], node_ids[j]) in edges or (node_ids[j], node_ids[i]) in edges:
# #                 linked_pairs += 1
# #     return linked_pairs / total_pairs if total_pairs > 0 else 0.0
# #
# # def generate_hypothesis(task_text, generator, node_embeddings, edges, top_k=10, num_candidates=30):
# #     device = next(generator.parameters()).device
# #     task_emb = get_embedding(task_text)
# #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# #
# #     candidates = []
# #     for _ in range(num_candidates):
# #         noise = torch.randn(1, 64).to(device)
# #         with torch.no_grad():
# #             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
# #         # Ищем ближайшие узлы
# #         scores = []
# #         for node_id, emb in node_embeddings.items():
# #             sim = np.dot(combo_emb, emb) / (np.linalg.norm(combo_emb) * np.linalg.norm(emb) + 1e-8)
# #             scores.append((node_id, sim))
# #         scores.sort(key=lambda x: x[1], reverse=True)
# #         top_nodes = [nid for nid, _ in scores[:top_k]]
# #         edge_ratio = compute_edge_ratio(top_nodes, edges)
# #         candidates.append((top_nodes, edge_ratio))
# #
# #     best_candidate = max(candidates, key=lambda x: x[1])
# #     return best_candidate[0], best_candidate[1]
# #
# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #     # Загружаем генератор
# #     generator = Generator().to(device)
# #     generator.load_state_dict(torch.load("generator.pth", map_location=device))
# #     generator.eval()
# #     print("✅ Генератор загружен.")
# #
# #     # Загружаем эмбеддинги узлов (генерируем на лету)
# #     node_embeddings = load_node_embeddings_from_db()
# #     print(f"✅ Загружено {len(node_embeddings)} узлов с эмбеддингами.")
# #
# #     # Загружаем рёбра
# #     edges = load_edges_from_db()
# #     print(f"✅ Загружено {len(edges)} рёбер.")
# #
# #     # Задача
# #     task = "Разработать устройство для управляемого перемещения человека по воздуху."
# #
# #     # Генерация гипотезы
# #     hypothesis_ids, edge_ratio = generate_hypothesis(
# #         task, generator, node_embeddings, edges,
# #         top_k=10, num_candidates=30
# #     )
# #
# #     if not hypothesis_ids:
# #         print("❌ Не удалось сгенерировать гипотезу.")
# #         return
# #
# #     # Получаем названия узлов
# #     conn = psycopg2.connect(**DB_CONFIG)
# #     cur = conn.cursor()
# #     placeholders = ','.join(['%s'] * len(hypothesis_ids))
# #     cur.execute(f"SELECT name FROM agi_evolution.knowledge_nodes WHERE id IN ({placeholders})", hypothesis_ids)
# #     names = [row[0] for row in cur.fetchall()]
# #     cur.close()
# #     conn.close()
# #
# #     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
# #     for name in names:
# #         print(f"  - {name}")
# #     print(f"\n🔗 Доля связанных пар: {edge_ratio:.2f}")
# #
# #     # Оценка дискриминатора
# #     discriminator = Discriminator().to(device)
# #     discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
# #     discriminator.eval()
# #
# #     combo_emb = np.mean([node_embeddings[nid] for nid in hypothesis_ids if nid in node_embeddings], axis=0)
# #     combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
# #     task_emb_t = torch.FloatTensor(get_embedding(task)).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         score = discriminator(task_emb_t, combo_emb_t).item()
# #     print(f"📊 Оценка дискриминатора: {score:.3f}")
# #
# # if __name__ == "__main__":
# #     main()
# #
# # # import torch
# # # import numpy as np
# # # import json
# # # import psycopg2
# # # from psycopg2.extras import DictCursor
# # # from sentence_transformers import SentenceTransformer
# # # from training_thinking_gan import Generator, Discriminator
# # # import sys
# # #
# # # DB_CONFIG = {
# # #     "host": "localhost",
# # #     "database": "postgres",
# # #     "user": "postgres",
# # #     "password": "postgres"
# # # }
# # #
# # # embedder = SentenceTransformer('all-MiniLM-L6-v2')
# # #
# # #
# # # def get_embedding(text):
# # #     emb = embedder.encode(text)
# # #     emb = np.array(emb)
# # #     norm = np.linalg.norm(emb) + 1e-8
# # #     return emb / norm
# # #
# # #
# # # def load_node_embeddings_from_db():
# # #     """Загружает эмбеддинги узлов из БД. Если отсутствуют – генерирует на лету."""
# # #     conn = psycopg2.connect(**DB_CONFIG)
# # #     cur = conn.cursor(cursor_factory=DictCursor)
# # #     cur.execute("SELECT id, name, description, embedding FROM agi_evolution.knowledge_nodes")
# # #     node_embeds = {}
# # #     for row in cur.fetchall():
# # #         node_id = row['id']
# # #         if row['embedding'] is not None:
# # #             emb = np.array(row['embedding'])
# # #             if emb.shape[0] == 384:
# # #                 node_embeds[node_id] = emb
# # #                 continue
# # #         # Если эмбеддинга нет или размерность не 384 – генерируем
# # #         text = row['name'] + ": " + (row['description'] or '')
# # #         emb = get_embedding(text)
# # #         node_embeds[node_id] = emb
# # #     cur.close()
# # #     conn.close()
# # #     return node_embeds
# # #
# # #
# # # def generate_hypothesis(task_text, generator, node_embeddings, top_k=10, num_candidates=20):
# # #     """Генерирует гипотезу для задачи."""
# # #     device = next(generator.parameters()).device
# # #     task_emb = get_embedding(task_text)
# # #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# # #
# # #     if not node_embeddings:
# # #         print("⚠️ Нет эмбеддингов узлов. Генерация невозможна.")
# # #         return []
# # #
# # #     candidates = []
# # #     for _ in range(num_candidates):
# # #         noise = torch.randn(1, 64).to(device)
# # #         with torch.no_grad():
# # #             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
# # #         scores = []
# # #         for node_id, emb in node_embeddings.items():
# # #             sim = np.dot(combo_emb, emb) / (np.linalg.norm(combo_emb) * np.linalg.norm(emb) + 1e-8)
# # #             scores.append((node_id, sim))
# # #         scores.sort(key=lambda x: x[1], reverse=True)
# # #         candidates.append(scores[:top_k])
# # #
# # #     # Выбираем комбинацию с максимальной суммой сходств
# # #     best_combo = max(candidates, key=lambda combo: sum(s for _, s in combo))
# # #     return [node_id for node_id, _ in best_combo]
# # #
# # #
# # # def main():
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # #
# # #     # Загружаем генератор
# # #     try:
# # #         generator = Generator().to(device)
# # #         generator.load_state_dict(torch.load("generator.pth", map_location=device))
# # #         generator.eval()
# # #         print("✅ Генератор загружен.")
# # #     except FileNotFoundError:
# # #         print("❌ Файл generator.pth не найден. Сначала обучите модель.")
# # #         sys.exit(1)
# # #
# # #     # Загружаем эмбеддинги узлов
# # #     node_embeddings = load_node_embeddings_from_db()
# # #     print(f"✅ Загружено {len(node_embeddings)} узлов с эмбеддингами.")
# # #
# # #     if not node_embeddings:
# # #         print("❌ Нет ни одного узла с эмбеддингом. Проверьте БД.")
# # #         sys.exit(1)
# # #
# # #     # Задача
# # #     task = "Разработать устройство для управляемого перемещения человека по воздуху."
# # #
# # #     # Генерируем гипотезу
# # #     hypothesis_ids = generate_hypothesis(task, generator, node_embeddings, top_k=10, num_candidates=20)
# # #
# # #     if not hypothesis_ids:
# # #         print("❌ Не удалось сгенерировать гипотезу.")
# # #         sys.exit(1)
# # #
# # #     # Получаем названия узлов
# # #     conn = psycopg2.connect(**DB_CONFIG)
# # #     cur = conn.cursor()
# # #     placeholders = ','.join(['%s'] * len(hypothesis_ids))
# # #     cur.execute(f"SELECT name FROM agi_evolution.knowledge_nodes WHERE id IN ({placeholders})", hypothesis_ids)
# # #     names = [row[0] for row in cur.fetchall()]
# # #     cur.close()
# # #     conn.close()
# # #
# # #     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
# # #     for name in names:
# # #         print(f"  - {name}")
# # #
# # #     # Оцениваем дискриминатором
# # #     try:
# # #         discriminator = Discriminator().to(device)
# # #         discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
# # #         discriminator.eval()
# # #
# # #         combo_emb = np.mean([node_embeddings[nid] for nid in hypothesis_ids if nid in node_embeddings], axis=0)
# # #         combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
# # #         task_emb_t = torch.FloatTensor(get_embedding(task)).unsqueeze(0).to(device)
# # #         with torch.no_grad():
# # #             score = discriminator(task_emb_t, combo_emb_t).item()
# # #         print(f"\n📊 Оценка дискриминатора: {score:.3f}")
# # #     except FileNotFoundError:
# # #         print("⚠️ Дискриминатор не найден, оценка невозможна.")
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# # #
# # #
# # #
# # # # import torch
# # # # import numpy as np
# # # # import json
# # # # import psycopg2
# # # # from psycopg2.extras import DictCursor
# # # # from sentence_transformers import SentenceTransformer
# # # # from training_thinking_gan import Generator, Discriminator
# # # #
# # # # DB_CONFIG = {
# # # #     "host": "localhost",
# # # #     "database": "postgres",
# # # #     "user": "postgres",
# # # #     "password": "postgres"
# # # # }
# # # #
# # # # # Загружаем модель эмбеддингов
# # # # embedder = SentenceTransformer('all-MiniLM-L6-v2')
# # # #
# # # # def get_embedding(text):
# # # #     emb = embedder.encode(text)
# # # #     emb = np.array(emb)
# # # #     norm = np.linalg.norm(emb) + 1e-8
# # # #     return emb / norm
# # # #
# # # # def load_node_embeddings_from_db():
# # # #     """Загружает эмбеддинги всех узлов из БД (используем уже сохранённые)."""
# # # #     conn = psycopg2.connect(**DB_CONFIG)
# # # #     cur = conn.cursor(cursor_factory=DictCursor)
# # # #     cur.execute("SELECT id, name, embedding FROM agi_evolution.knowledge_nodes WHERE embedding IS NOT NULL")
# # # #     node_embeds = {}
# # # #     for row in cur.fetchall():
# # # #         emb = np.array(row['embedding'])
# # # #         if emb.shape[0] != 384:
# # # #             continue  # пропускаем узлы с нестандартной размерностью
# # # #         node_embeds[row['id']] = emb
# # # #     cur.close()
# # # #     conn.close()
# # # #     return node_embeds
# # # #
# # # # def generate_hypothesis(task_text, generator, node_embeddings, top_k=10, num_candidates=20):
# # # #     """Генерирует гипотезу для задачи."""
# # # #     device = next(generator.parameters()).device
# # # #     task_emb = get_embedding(task_text)
# # # #     task_emb_t = torch.FloatTensor(task_emb).unsqueeze(0).to(device)
# # # #
# # # #     candidates = []
# # # #     for _ in range(num_candidates):
# # # #         noise = torch.randn(1, 64).to(device)
# # # #         with torch.no_grad():
# # # #             combo_emb = generator(task_emb_t, noise).cpu().numpy()[0]
# # # #         # Ищем ближайшие узлы к сгенерированному эмбеддингу
# # # #         scores = []
# # # #         for node_id, emb in node_embeddings.items():
# # # #             sim = np.dot(combo_emb, emb) / (np.linalg.norm(combo_emb) * np.linalg.norm(emb) + 1e-8)
# # # #             scores.append((node_id, sim))
# # # #         scores.sort(key=lambda x: x[1], reverse=True)
# # # #         candidates.append(scores[:top_k])
# # # #
# # # #     # Выбираем комбинацию с максимальной суммой косинусных сходств
# # # #     best_combo = max(candidates, key=lambda combo: sum(s for _, s in combo))
# # # #     return [node_id for node_id, _ in best_combo]
# # # #
# # # # def main():
# # # #     # Загружаем обученный генератор
# # # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # #     generator = Generator().to(device)
# # # #     generator.load_state_dict(torch.load("generator.pth", map_location=device))
# # # #     generator.eval()
# # # #     print("✅ Генератор загружен.")
# # # #
# # # #     # Загружаем эмбеддинги узлов из БД
# # # #     node_embeddings = load_node_embeddings_from_db()
# # # #     print(f"✅ Загружено {len(node_embeddings)} узлов с эмбеддингами.")
# # # #
# # # #     # Задача
# # # #     task = "Разработать устройство для управляемого перемещения человека по воздуху."
# # # #
# # # #     # Генерируем гипотезу
# # # #     hypothesis_ids = generate_hypothesis(task, generator, node_embeddings, top_k=10, num_candidates=20)
# # # #
# # # #     # Получаем названия узлов
# # # #     conn = psycopg2.connect(**DB_CONFIG)
# # # #     cur = conn.cursor()
# # # #     placeholders = ','.join(['%s'] * len(hypothesis_ids))
# # # #     cur.execute(f"SELECT name FROM agi_evolution.knowledge_nodes WHERE id IN ({placeholders})", hypothesis_ids)
# # # #     names = [row[0] for row in cur.fetchall()]
# # # #     cur.close()
# # # #     conn.close()
# # # #
# # # #     print("\n🧠 ГИПОТЕЗА (комбинация узлов):")
# # # #     for name in names:
# # # #         print(f"  - {name}")
# # # #
# # # #     # Опционально: оценить гипотезу дискриминатором
# # # #     discriminator = Discriminator().to(device)
# # # #     discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
# # # #     discriminator.eval()
# # # #
# # # #     # Эмбеддинг комбинации (усреднение)
# # # #     combo_emb = np.mean([node_embeddings[nid] for nid in hypothesis_ids if nid in node_embeddings], axis=0)
# # # #     combo_emb_t = torch.FloatTensor(combo_emb).unsqueeze(0).to(device)
# # # #     task_emb_t = torch.FloatTensor(get_embedding(task)).unsqueeze(0).to(device)
# # # #     with torch.no_grad():
# # # #         score = discriminator(task_emb_t, combo_emb_t).item()
# # # #     print(f"\n📊 Оценка дискриминатора: {score:.3f}")
# # # #
# # # # if __name__ == "__main__":
# # # #     main()