# core/thinking/train_massive_for_thinking_gan.py

import json
import numpy as np

from core.thinking.create_embeddings_kg import get_node_embedding


def load_dataset_with_embeddings(dataset_file, node_embeddings):
    """Загружает датасет и преобразует в эмбеддинги."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X_task = []
    X_combo = []
    y = []

    for item in data:
        task_emb = get_node_embedding(item['task'])  # эмбеддинг задачи
        combo_emb = np.mean([node_embeddings.get(node_id, np.zeros(384))
                             for node_id in item['combination']], axis=0)
        # Нормализуем
        norm = np.linalg.norm(combo_emb) + 1e-8
        combo_emb = combo_emb / norm

        X_task.append(task_emb)
        X_combo.append(combo_emb)
        y.append(item['score'])

    return np.array(X_task), np.array(X_combo), np.array(y)