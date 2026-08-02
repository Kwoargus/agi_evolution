import json
import psycopg2
from psycopg2.extras import DictCursor

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def diagnose_dataset_and_db():
    # 1. Загружаем датасет
    try:
        with open("training_data_inventions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Файл training_data_inventions.json не найден.")
        return

    print(f"📊 Всего примеров в датасете: {len(data)}")

    # 2. Собираем все ID узлов из комбинаций
    all_node_ids = set()
    for item in data:
        for node_id in item.get("combination", []):
            if node_id and isinstance(node_id, str):
                all_node_ids.add(node_id)

    print(f"🔢 Уникальных ID узлов в датасете: {len(all_node_ids)}")

    if not all_node_ids:
        print("⚠️ В датасете нет ID узлов (возможно, используются строки).")
        return

    # 3. Проверяем наличие в БД
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Правильный запрос с приведением к UUID
    # Преобразуем список ID в кортеж и используем ANY с явным приведением типа
    # Создаём массив UUID из строк
    placeholders = ','.join(['%s::uuid'] * len(all_node_ids))
    query = f"SELECT id FROM agi_evolution.knowledge_nodes WHERE id IN ({placeholders})"
    cur.execute(query, tuple(all_node_ids))

    existing_ids = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()

    missing_ids = all_node_ids - existing_ids
    print(f"✅ Узлов, присутствующих в БД: {len(existing_ids)}")
    print(f"❌ Узлов, отсутствующих в БД: {len(missing_ids)}")

    if missing_ids:
        print("Примеры отсутствующих ID (первые 5):")
        for mid in list(missing_ids)[:5]:
            print(f"  - {mid}")

    # 4. Проверяем, есть ли в датасете примеры с оценками, но без ID
    string_entries = []
    for item in data:
        if any(not isinstance(nid, str) or len(nid) != 36 for nid in item.get("combination", [])):
            string_entries.append(item)
    if string_entries:
        print(f"⚠️ Найдено {len(string_entries)} примеров с не-UUID значениями (возможно, строки).")
    else:
        print("✅ Все комбинации содержат корректные UUID.")

if __name__ == "__main__":
    diagnose_dataset_and_db()