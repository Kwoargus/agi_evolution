import psycopg2
import numpy as np
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

# Инициализируем модель один раз
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_node_embedding(text):
    """Возвращает эмбеддинг текста (384-мерный)."""
    emb = embedding_model.encode(text)
    emb = np.array(emb)
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm

def load_all_node_embeddings():
    """Загружает эмбеддинги узлов из БД. Если отсутствуют – генерирует."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT id, name, description, embedding FROM agi_evolution.knowledge_nodes")
    node_embeddings = {}
    for row in cur.fetchall():
        node_id = row['id']
        if row['embedding'] is not None:
            emb = np.array(row['embedding'])
            # Проверяем размерность (если не 384 – пересоздаём)
            if emb.shape[0] != 384:
                emb = get_node_embedding(row['name'] + ": " + (row['description'] or ''))
        else:
            emb = get_node_embedding(row['name'] + ": " + (row['description'] or ''))
        node_embeddings[node_id] = emb
    cur.close()
    conn.close()
    return node_embeddings



# # core/thinking/create_embeddings_kg.py
#
# import psycopg2
# import numpy as np
# from psycopg2.extras import DictCursor
# import ollama
#
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "postgres",
#     "user": "postgres",
#     "password": "postgres"
# }
#
# def get_node_embedding(name, description=None):
#     """Получает эмбеддинг узла через Ollama."""
#     text = f"{name}: {description}" if description else name
#     response = ollama.embeddings(model='qwen2.5:7b', prompt=text)
#     emb = np.array(response['embedding'])
#     norm = np.linalg.norm(emb) + 1e-8
#     return emb / norm
#
# def load_all_node_embeddings():
#     """Загружает эмбеддинги всех узлов из БД (генерирует, если их нет)."""
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor(cursor_factory=DictCursor)
#     cur.execute("SELECT id, name, description, embedding FROM agi_evolution.knowledge_nodes")
#     node_embeddings = {}
#     for row in cur.fetchall():
#         node_id = row['id']
#         if row['embedding'] is not None:
#             emb = np.array(row['embedding'])
#         else:
#             emb = get_node_embedding(row['name'], row['description'])
#         node_embeddings[node_id] = emb
#     cur.close()
#     conn.close()
#     return node_embeddings