# scripts/import_to_db.py
"""
Импорт сущностей и связей из Parquet-файлов GraphRAG в базу данных PostgreSQL.
"""

import pandas as pd
import psycopg2
import uuid
import os

# Конфигурация подключения к вашей БД
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def import_graphrag_to_db():
    print("\n" + "=" * 70)
    print("📥 ИМПОРТ ГРАФА ЗНАНИЙ ИЗ GRAPHRAG В БД")
    print("=" * 70)

    # Чтение Parquet-файлов
    output_dir = "../graphrag-main/output/"
    entities_df = pd.read_parquet(os.path.join(output_dir, "entities.parquet"))
    relationships_df = pd.read_parquet(os.path.join(output_dir, "relationships.parquet"))

    print(f"\n📊 Прочитано сущностей: {len(entities_df)}")
    print(f"📊 Прочитано связей: {len(relationships_df)}")

    # Подключение к БД
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # 1. Импорт узлов
        print("\n🔹 Импорт узлов...")
        for _, row in entities_df.iterrows():
            node_id = str(uuid.uuid4())
            name = row['title']
            node_type = row['type']
            description = row['description']
            properties = [row['type'], str(row['frequency'])]

            cur.execute("""
                INSERT INTO agi_evolution.knowledge_nodes 
                (id, name, node_type, properties, description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (node_id, name, node_type, properties, description))

        print("   ✅ Узлы импортированы")

        # 2. Импорт связей
        print("\n🔹 Импорт связей...")
        for _, row in relationships_df.iterrows():
            source_name = row['source']
            target_name = row['target']
            weight = row['weight']
            description = row.get('description', '')

            # Ищем ID исходного узла
            cur.execute("""
                SELECT id FROM agi_evolution.knowledge_nodes WHERE name = %s
            """, (source_name,))
            source_row = cur.fetchone()
            if not source_row:
                continue

            # Ищем ID целевого узла
            cur.execute("""
                SELECT id FROM agi_evolution.knowledge_nodes WHERE name = %s
            """, (target_name,))
            target_row = cur.fetchone()
            if not target_row:
                continue

            source_id = source_row[0]
            target_id = target_row[0]

            edge_id = f"edge_{source_id[:8]}_{target_id[:8]}"

            cur.execute("""
                INSERT INTO agi_evolution.knowledge_edges 
                (id, source_id, target_id, edge_type, weight, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (edge_id, source_id, target_id, "related_to", weight, description))

        print("   ✅ Связи импортированы")

        conn.commit()

    except Exception as e:
        print(f"\n❌ Ошибка импорта: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()

    print("\n✅ ИМПОРТ ЗАВЕРШЁН")

if __name__ == "__main__":
    import_graphrag_to_db()