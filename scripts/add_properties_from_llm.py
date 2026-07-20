import psycopg2
import json
import requests
import time
import re

conn = psycopg2.connect(
    host='localhost',
    port=5432,
    dbname='postgres',
    user='postgres',
    password='postgres'
)


def to_postgres_array(lst):
    if not lst:
        return '{}'
    # Экранируем двойные кавычки внутри элементов
    escaped = [str(s).replace('"', '\\"') for s in lst]
    return '{' + ','.join(f'"{s}"' for s in escaped) + '}'


cur = conn.cursor()

# Получаем узлы без свойств (или с пустыми)
cur.execute("SELECT id, name, description FROM agi_evolution.knowledge_nodes WHERE properties IS NULL OR properties = '{}' LIMIT 50")
nodes = cur.fetchall()

for node_id, name, description in nodes:
    # Сбрасываем транзакцию, если была ошибка
    if conn.closed == 0:
        conn.rollback()

    prompt = f"""
    Ты — инженер-аналитик. Извлеки из описания узла 1–3 ключевых функциональных свойства (действия или способности). 
    Верни ТОЛЬКО список в формате JSON, например: ["создавать подъёмную силу", "перевозить"]
    Если описание пустое или не содержит функций, верни [].

    Узел: {name}
    Описание: {description[:300]}
    """

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen2.5:7b',
                'prompt': prompt,
                'stream': False,
                'temperature': 0.1,
                'max_tokens': 100
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '[]')
            # Извлекаем JSON-массив
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                properties_list = json.loads(match.group())
                if isinstance(properties_list, list) and properties_list:
                    # Преобразуем в формат PostgreSQL text[]
                    pg_array = to_postgres_array(properties_list)
                    cur.execute(
                        "UPDATE agi_evolution.knowledge_nodes SET properties = %s::text[] WHERE id = %s",
                        (pg_array, node_id)
                    )
                    conn.commit()
                    print(f"✅ {name} → {properties_list}")
                else:
                    # Если список пуст, устанавливаем пустой массив
                    cur.execute(
                        "UPDATE agi_evolution.knowledge_nodes SET properties = '{}' WHERE id = %s",
                        (node_id,)
                    )
                    conn.commit()
                    print(f"⏭️ {name} → (пусто)")
    except Exception as e:
        print(f"⚠️ Ошибка для {name}: {e}")
        conn.rollback()

    time.sleep(0.5)

cur.close()
conn.close()
print("✅ Готово")