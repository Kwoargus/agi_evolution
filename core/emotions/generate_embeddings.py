import json
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def generate_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Эмбеддинги для событий
    cur.execute("SELECT id, description, properties FROM agi_evolution.trigger_event WHERE embedding IS NULL")
    rows = cur.fetchall()
    for row in rows:
        text = f"{row['description']} {json.dumps(row['properties'], ensure_ascii=False)}"
        emb = model.encode(text)
        cur.execute("UPDATE agi_evolution.trigger_event SET embedding = %s WHERE id = %s", (json.dumps(emb.tolist()), row['id']))
        print(f"✅ Эмбеддинг для события {row['id']}")

    # Эмбеддинги для эмоций
    cur.execute("SELECT id, type, description, properties FROM agi_evolution.emotion_respons WHERE embedding IS NULL")
    rows = cur.fetchall()
    for row in rows:
        text = f"{row['type']} {row['description']} {json.dumps(row['properties'], ensure_ascii=False)}"
        emb = model.encode(text)
        cur.execute("UPDATE agi_evolution.emotion_respons SET embedding = %s WHERE id = %s", (json.dumps(emb.tolist()), row['id']))
        print(f"✅ Эмбеддинг для эмоции {row['id']}")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Все эмбеддинги сгенерированы и сохранены.")

if __name__ == '__main__':
    generate_embeddings()