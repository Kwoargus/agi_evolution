import json
import uuid
import re
import time
import ollama
import psycopg2
from psycopg2.extras import DictCursor

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def load_dataset(dataset_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_all_node_names(dataset):
    """Извлекает все названия узлов из датасета."""
    names = set()
    for item in dataset:
        for node_id in item.get('combination', []):
            if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', node_id):
                continue
            names.add(node_id)
    return names

def get_existing_node_ids(conn, names):
    cur = conn.cursor(cursor_factory=DictCursor)
    placeholders = ','.join(['%s'] * len(names))
    cur.execute(f"SELECT id, name FROM agi_evolution.knowledge_nodes WHERE name IN ({placeholders})", tuple(names))
    result = {row['name']: row['id'] for row in cur.fetchall()}
    cur.close()
    return result

def generate_node_data(name, use_llm=True):
    if use_llm:
        prompt = f"""
Ты — эксперт по техническим системам и инженерии. 
Для концептуального узла «{name}» сгенерируй:
- краткое описание (1–2 предложения) на русском языке,
- список свойств (ключевые характеристики, 3–5 пунктов) на русском языке (китайский язык использовать нельзя).

Ответ должен быть в формате JSON:
{{"description": "...", "properties": ["свойство1", "свойство2", ...]}}
"""
        try:
            response = ollama.chat(
                model='qwen2.5:7b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            content = response['message']['content'].strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ Ошибка LLM для {name}: {e}")
    return {
        "description": f"Концептуальный узел: {name}",
        "properties": ["компонент технической системы"]
    }

def generate_insert_sql(name, node_id, data):
    description = data.get('description', f'Узел: {name}')
    properties = data.get('properties', [])
    # Экранируем кавычки в каждом свойстве
    escaped_props = [p.replace("'", "''") for p in properties]
    # Формируем массив строк
    properties_array = "ARRAY[" + ", ".join([f"'{p}'" for p in escaped_props]) + "]::text[]"

    return f"""
INSERT INTO agi_evolution.knowledge_nodes (id, name, node_type, properties, description, embedding, parameters, metadata, created_at, updated_at, theme_id)
VALUES (
    '{node_id}',
    '{name.replace("'", "''")}',
    'CONCEPT',
    {properties_array},
    '{description.replace("'", "''")}',
    NULL,
    '{{}}'::jsonb,
    '{{}}'::jsonb,
    NOW(),
    NOW(),
    NULL
);
"""

def main():
    dataset = load_dataset("training_data_inventions.json")
    print(f"📊 Загружено {len(dataset)} примеров.")

    all_names = extract_all_node_names(dataset)
    print(f"🔢 Уникальных названий в датасете: {len(all_names)}")

    conn = psycopg2.connect(**DB_CONFIG)
    existing = get_existing_node_ids(conn, all_names)
    missing_names = all_names - set(existing.keys())
    print(f"❌ Отсутствует в БД: {len(missing_names)} узлов.")

    if missing_names:
        print("Генерируем SQL-скрипт для добавления отсутствующих узлов...")
        insert_statements = []
        name_to_id = existing.copy()

        for name in missing_names:
            node_id = str(uuid.uuid4())
            data = generate_node_data(name, use_llm=True)
            insert_sql = generate_insert_sql(name, node_id, data)
            insert_statements.append(insert_sql)
            name_to_id[name] = node_id
            print(f"  {name} → {node_id}")
            time.sleep(0.3)

        with open("insert_missing_nodes.sql", "w", encoding="utf-8") as f:
            f.write("BEGIN;\n")
            f.write("\n".join(insert_statements))
            f.write("\nCOMMIT;\n")
        print("✅ SQL-скрипт сохранён в insert_missing_nodes.sql")
        print("⚠️ Выполните его в БД, затем запустите скрипт снова для обновления датасета.")

        with open("name_to_id.json", "w", encoding="utf-8") as f:
            json.dump(name_to_id, f, ensure_ascii=False, indent=2)
        print("✅ Словарь name→id сохранён в name_to_id.json")

        conn.close()
        return

    print("✅ Все узлы присутствуют в БД. Обновляем датасет...")
    updated_dataset = []
    for item in dataset:
        new_item = item.copy()
        new_combination = []
        for node_id in item.get('combination', []):
            if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', node_id):
                new_combination.append(node_id)
            else:
                new_id = existing.get(node_id)
                if new_id:
                    new_combination.append(new_id)
                else:
                    print(f"⚠️ Узел '{node_id}' не найден. Пропускаем.")
        new_item['combination'] = new_combination
        updated_dataset.append(new_item)

    with open("training_data_inventions_fixed.json", "w", encoding="utf-8") as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
    print("✅ Обновлённый датасет сохранён в training_data_inventions_fixed.json")
    conn.close()

if __name__ == "__main__":
    main()