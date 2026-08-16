import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def generate_link_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Получаем все связи с эмбеддингами события и эмоции
    cur.execute("""
        SELECT 
            l.id AS link_id,
            e.description AS event_desc,
            e.properties AS event_props,
            e.embedding AS event_embedding,
            r.type AS emotion_type,
            r.description AS emotion_desc,
            r.properties AS emotion_props,
            r.embedding AS emotion_embedding
        FROM agi_evolution.event_emotion_link l
        JOIN agi_evolution.trigger_event e ON l.event_id = e.id
        JOIN agi_evolution.emotion_respons r ON l.emotion_id = r.id
    """)
    rows = cur.fetchall()

    for row in rows:
        # Формируем текст для эмбеддинга связи
        event_emb = np.array(row['event_embedding']) if row['event_embedding'] is not None else np.zeros(384)
        emotion_emb = np.array(row['emotion_embedding']) if row['emotion_embedding'] is not None else np.zeros(384)
        # Усредняем эмбеддинги события и эмоции
        combined_emb = (event_emb + emotion_emb) / 2
        # Нормализуем
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)

        cur.execute("""
            UPDATE agi_evolution.event_emotion_link 
            SET embedding = %s 
            WHERE id = %s
        """, (json.dumps(combined_emb.tolist()), row['link_id']))
        print(f"✅ Эмбеддинг для связи {row['link_id']}")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Эмбеддинги для всех связей сгенерированы.")

if __name__ == '__main__':
    generate_link_embeddings()