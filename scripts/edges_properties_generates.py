import json
import sys
import time
import re
import psycopg2
from psycopg2.extras import DictCursor
import ollama

# --- Конфигурация ---
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

MODEL_NAME = "qwen2.5:7b"

def call_ollama(prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    return response['message']['content'].strip()

def generate_edge_properties(description, source_name, target_name, edge_type):
    prompt = f"""Ты — эксперт по семантическому обогащению графов знаний.

У тебя есть ребро (связь) между двумя сущностями:
- Сущность 1: {source_name if source_name else 'неизвестно'}
- Сущность 2: {target_name if target_name else 'неизвестно'}
- Тип связи: {edge_type}
- Описание связи: {description[:500] if description else 'нет описания'}

Сгенерируй свойства для этого ребра в формате JSON. Используй следующие ключи:
- "контекст" – краткое описание связи в контексте (1–2 предложения)
- "доказательства" – обоснование, почему эта связь существует (цитаты или логика)
- "уверенность" – число от 0 до 1, насколько ты уверен в этой связи (например, 0.9)
- "краткое_описание" – одна фраза, суть связи
- "направление" – одно из: "прямая", "обратная", "симметричная"
- "сила_связи" – одно из: "слабая", "средняя", "сильная"

Ответ должен быть только JSON, без пояснений.
"""
    response = call_ollama(prompt)
    # Извлекаем JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            return None
    return None

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)

    # Убеждаемся, что все properties = '{}' (для пустых)
    cur.execute("UPDATE agi_evolution.knowledge_edges SET properties = '{}'::jsonb WHERE properties IS NULL;")
    conn.commit()

    # Выбираем рёбра с пустыми properties
    cur.execute("""
        SELECT 
            e.id, e.description, e.edge_type,
            n1.name AS source_name, n2.name AS target_name
        FROM agi_evolution.knowledge_edges e
        LEFT JOIN agi_evolution.knowledge_nodes n1 ON e.source_id = n1.id
        LEFT JOIN agi_evolution.knowledge_nodes n2 ON e.target_id = n2.id
        WHERE e.properties = '{}'::jsonb
    """)
    edges = cur.fetchall()
    total = len(edges)
    print(f"Найдено рёбер для генерации свойств: {total}", file=sys.stderr)

    if total == 0:
        print("Нет рёбер для обработки.", file=sys.stderr)
        return

    updated = 0
    errors = 0

    for idx, row in enumerate(edges, start=1):
        edge_id = row['id']
        desc = row['description'] or ''
        source_name = row['source_name'] or ''
        target_name = row['target_name'] or ''
        edge_type = row['edge_type'] or 'related_to'

        print(f"[{idx}/{total}] Обработка ребра {edge_id}...", file=sys.stderr)

        props = generate_edge_properties(desc, source_name, target_name, edge_type)
        if props:
            props_json = json.dumps(props, ensure_ascii=False)
            cur.execute(
                "UPDATE agi_evolution.knowledge_edges SET properties = %s::jsonb, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (props_json, edge_id)
            )
            updated += 1
            print(f"  -> обновлено", file=sys.stderr)
        else:
            errors += 1
            print(f"  -> ошибка генерации", file=sys.stderr)

        time.sleep(0.5)

    conn.commit()
    print(f"Всего обработано: {total}, обновлено: {updated}, ошибок: {errors}", file=sys.stderr)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()