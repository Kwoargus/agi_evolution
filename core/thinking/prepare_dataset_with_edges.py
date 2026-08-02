import json
import numpy as np
from sentence_transformers import SentenceTransformer
from generate_hypothesis import KnowledgeGraph, cosine_sim

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def prepare_dataset_with_edges(dataset_file, kg, embedder):
    """Добавляет признаки рёбер в датасет."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        task = item['task']
        task_emb = embedder.encode(task)
        task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
        node_ids = item['combination']

        # Убедимся, что node_ids – плоский список строк
        if isinstance(node_ids, list) and len(node_ids) > 0:
            # Если первый элемент – список, распакуем
            if isinstance(node_ids[0], list):
                node_ids = [node_id for sublist in node_ids for node_id in sublist]
        elif isinstance(node_ids, dict):
            node_ids = list(node_ids.values())

        # Преобразуем все элементы в строки (если они не строки)
        node_ids = [str(nid) for nid in node_ids]

        if len(node_ids) < 2:
            continue

        # Вычисляем признаки рёбер
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

        # Копируем исходный пример и добавляем признаки
        new_item = item.copy()
        new_item['edge_features'] = {
            'edge_sim_mean': edge_sim_mean,
            'edge_density': edge_density,
            'type_diversity': type_diversity
        }
        new_data.append(new_item)

    with open('training_data_inventions_with_edges.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Сохранено {len(new_data)} примеров с признаками рёбер.")

if __name__ == "__main__":
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kg = KnowledgeGraph()
    kg.load_from_db(embedder)
    prepare_dataset_with_edges('training_data_inventions.json', kg, embedder)