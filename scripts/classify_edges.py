import json
import sys
import time
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

# Возможные типы связей (можно дополнить)
EDGE_TYPES = [
    "причина-следствие",
    "часть-целое",
    "предок-потомок",
    "группа-представитель",
    "группа-подгруппа",
    "старший-младший",
    "ведущий-ведомый",
    "родственник-родственник",
    "структура-элемент",
    "конструкция-деталь",
    "агрегат-компонент",
    "агент-контрагент",
    "предыдущий-следующий"
]

MODEL_NAME = "qwen2.5:7b"


def call_ollama(prompt, max_retries=3):
    """Вызов Ollama с повторными попытками при ошибках."""
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,
                    'num_predict': 50  # ограничиваем длину ответа
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ошибка при вызове Ollama (попытка {attempt + 1}): {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # экспоненциальная задержка
            else:
                return "related_to"  # если все попытки неудачны
    return "related_to"


def get_edge_type_from_llm(description, source_name, target_name):
    prompt = f"""Ты — эксперт по классификации связей в графе знаний.

У тебя есть описание связи между двумя сущностями:
- Сущность 1: {source_name if source_name else 'неизвестно'}
- Сущность 2: {target_name if target_name else 'неизвестно'}
- Описание связи: {description[:500] if description else 'нет описания'}

Определи, какой из перечисленных типов связи лучше всего подходит для этой пары.
Возможные типы (выбери один):
{', '.join(EDGE_TYPES)}

Ответ должен содержать только название типа связи из списка.
Если ни один не подходит, ответь "related_to"."""

    response = call_ollama(prompt)

    # Проверка и очистка ответа
    response = response.strip()
    # Если модель вернула несколько слов, пытаемся найти подходящий тип
    for t in EDGE_TYPES:
        if t in response:
            return t
    if "related_to" in response.lower():
        return "related_to"
    # Если ничего не подошло, возвращаем related_to
    return "related_to"


def main():
    # Подключение к БД
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)

    # Выбираем рёбра с edge_type = 'related_to' и непустым description
    cur.execute("""
        SELECT 
            e.id, e.description, e.source_id, e.target_id,
            n1.name AS source_name, n2.name AS target_name
        FROM agi_evolution.knowledge_edges e
        LEFT JOIN agi_evolution.knowledge_nodes n1 ON e.source_id = n1.id
        LEFT JOIN agi_evolution.knowledge_nodes n2 ON e.target_id = n2.id
        WHERE e.edge_type = 'related_to'
          AND e.description IS NOT NULL AND e.description != ''
    """)
    edges = cur.fetchall()
    total = len(edges)
    print(f"Найдено рёбер для классификации: {total}", file=sys.stderr)

    if total == 0:
        print("Нет рёбер для обработки.", file=sys.stderr)
        return

    updated = 0
    skipped = 0

    for idx, row in enumerate(edges, start=1):
        edge_id = row['id']
        desc = row['description'] or ''
        source_name = row['source_name'] or ''
        target_name = row['target_name'] or ''

        print(f"[{idx}/{total}] Обработка ребра {edge_id}...", file=sys.stderr)

        new_type = get_edge_type_from_llm(desc, source_name, target_name)

        if new_type != 'related_to':
            cur.execute(
                "UPDATE agi_evolution.knowledge_edges SET edge_type = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (new_type, edge_id)
            )
            updated += 1
            print(f"  -> обновлено на '{new_type}'", file=sys.stderr)
        else:
            skipped += 1
            print(f"  -> пропущено (остаётся 'related_to')", file=sys.stderr)

        # Задержка, чтобы не перегружать модель
        time.sleep(0.5)

    conn.commit()
    print(f"\nВсего обработано: {total}, обновлено: {updated}, пропущено: {skipped}", file=sys.stderr)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()