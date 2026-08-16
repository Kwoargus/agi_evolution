# core/emotions/augment_emotion_graph.py
import json
import uuid
import random
import time
import re
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import ollama

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

EMOTION_TYPES = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'anticipation', 'empathy', 'interest', 'boredom']

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

def get_existing_emotion_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, type, description FROM agi_evolution.emotion_respons")
    rows = cur.fetchall()
    cur.close()
    return {f"{row[1]}:{row[2]}": row[0] for row in rows}

def get_emotion_properties_by_type(conn, emotion_type):
    """Возвращает properties первой найденной эмоции данного типа (ручной)."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT properties FROM agi_evolution.emotion_respons 
        WHERE type = %s AND (metadata->>'source' = 'manual' OR metadata->>'source' IS NULL)
        LIMIT 1
    """, (emotion_type,))
    row = cur.fetchone()
    cur.close()
    if row:
        return row['properties'] or {}
    return {'intensity': 0.5, 'positive_or_negative': 0.0}

def get_manual_emotion_properties(conn, emotion_type):
    """Возвращает properties ручной эмоции данного типа."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT properties FROM agi_evolution.emotion_respons 
        WHERE type = %s AND (metadata->>'source' = 'manual' OR metadata->>'source' IS NULL)
        LIMIT 1
    """, (emotion_type,))
    row = cur.fetchone()
    cur.close()
    if row:
        return row['properties'] or {}
    # Если нет ручной, создаём дефолтные
    return {'intensity': 0.5, 'positive_or_negative': 0.0, 'causes_action': ''}

def create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids, event_props=None):
    key = f"{emotion_type}:{emotion_desc}"
    if key in emotion_ids:
        return emotion_ids[key]
    new_id = str(uuid.uuid4())
    base_props = get_emotion_properties_by_type(conn, emotion_type)
    # Модифицируем свойства на основе параметров события
    if event_props:
        # Например, увеличиваем интенсивность в зависимости от привлекательности события
        attractiveness = event_props.get('attractiveness', 0.5)
        base_props['intensity'] = min(1.0, base_props.get('intensity', 0.5) * (0.5 + attractiveness * 0.5))
        # Можно добавить другие модификации
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
        VALUES (%s, %s, %s, %s, %s)
    """, (new_id, emotion_type, emotion_desc, Json(base_props), Json({'generated_by': 'llm'})))
    conn.commit()
    cur.close()
    emotion_ids[key] = new_id
    return new_id

# def create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids):
#     key = f"{emotion_type}:{emotion_desc}"
#     if key in emotion_ids:
#         return emotion_ids[key]
#     new_id = str(uuid.uuid4())
#     # Копируем свойства из существующей эмоции этого типа
#     base_props = get_emotion_properties_by_type(conn, emotion_type)
#     cur = conn.cursor()
#     cur.execute("""
#         INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
#         VALUES (%s, %s, %s, %s, %s)
#     """, (new_id, emotion_type, emotion_desc, Json(base_props), Json({'generated_by': 'llm'})))
#     conn.commit()
#     cur.close()
#     emotion_ids[key] = new_id
#     return new_id


def augment_events():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT id, type, description, properties 
        FROM agi_evolution.trigger_event 
        WHERE metadata->>'source' = 'manual'
    """)
    base_events = cur.fetchall()

    emotion_ids = get_existing_emotion_ids(conn)

    for ev in base_events:
        ev_id = ev['id']
        ev_type = ev['type']
        ev_desc = ev['description']
        ev_props = ev['properties'] or {}

        print(f"🔁 Аугментация события: {ev_type} - {ev_desc}")

        variations = random.randint(100, 200)
        for i in range(variations):
            new_props = ev_props.copy()
            for key in ['attractiveness', 'intensity', 'number_of_participants']:
                if key in new_props and isinstance(new_props[key], (int, float)):
                    factor = random.uniform(0.6, 1.4)
                    new_props[key] = round(new_props[key] * factor, 2)
            if random.random() < 0.3:
                new_props['random_factor'] = round(random.uniform(0.1, 1.0), 2)

            props_str = ", ".join([f"{k}={v}" for k, v in new_props.items() if not k.startswith('_')])
            full_desc = f"{ev_desc} (with {props_str})" if props_str else ev_desc

            prompt = f"""
Ты — эксперт по психологии эмоций.
Дано событие: {full_desc}
Свойства события: {json.dumps(new_props, ensure_ascii=False)}
Какая эмоция из списка {EMOTION_TYPES} лучше всего соответствует этому событию?
Также укажи вероятность (0-1) возникновения этой эмоции и коэффициент интенсивности (0.5-2.0).
Ответь ТОЛЬКО в формате JSON: {{"emotion": "...", "probability": 0.0-1.0, "intensity_factor": 0.5-2.0}}
Без пояснений.
"""
            try:
                response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
                content = response['message']['content'].strip()
                result = extract_json(content)
                if result is None:
                    print(f"  ❌ Невалидный JSON: {content[:100]}")
                    continue
                emotion_type = result.get('emotion', 'trust')
                probability = min(1.0, max(0.0, result.get('probability', 0.5)))
                intensity = min(2.0, max(0.5, result.get('intensity_factor', 1.0)))

                emotion_desc = f"Аугментированная {emotion_type} для {ev_desc[:30]}"
                emotion_id = create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids, new_props)

                new_event_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (new_event_id, ev_type, full_desc, Json(new_props), Json({'augmented': True, 'source_event': ev_id})))

                cur.execute("""
                    INSERT INTO agi_evolution.event_emotion_link (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (new_event_id, emotion_id, 'event_emotion', probability, intensity, Json({'generated_by': 'llm', 'variation': i+1})))

                conn.commit()
                print(f"  ✅ Вариация {i+1}: {emotion_type} (p={probability:.2f}, intensity={intensity:.2f})")
                time.sleep(0.5)

            except Exception as e:
                print(f"  ❌ Ошибка вариации {i+1}: {e}")
                conn.rollback()

    cur.close()
    conn.close()
    print("✅ Аугментация завершена.")

if __name__ == '__main__':
    augment_events()



# import json
# import uuid
# import random
# import time
# import re
# import psycopg2
# from psycopg2.extras import Json, RealDictCursor
# import ollama
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
# EMOTION_TYPES = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'anticipation', 'empathy', 'interest', 'boredom']
#
# def extract_json(text):
#     """Извлекает JSON-объект из текста."""
#     match = re.search(r'\{.*\}', text, re.DOTALL)
#     if match:
#         try:
#             return json.loads(match.group())
#         except:
#             return None
#     return None
#
# def get_existing_emotion_ids(conn):
#     cur = conn.cursor()
#     cur.execute("SELECT id, type, description FROM agi_evolution.emotion_respons")
#     rows = cur.fetchall()
#     cur.close()
#     # Индексируем по type:description
#     return {f"{row[1]}:{row[2]}": row[0] for row in rows}
#
# def create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids):
#     key = f"{emotion_type}:{emotion_desc}"
#     if key in emotion_ids:
#         return emotion_ids[key]
#     new_id = str(uuid.uuid4())
#     cur = conn.cursor()
#     cur.execute("""
#         INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
#         VALUES (%s, %s, %s, %s, %s)
#     """, (new_id, emotion_type, emotion_desc, Json({}), Json({'generated_by': 'llm'})))
#     conn.commit()
#     cur.close()
#     emotion_ids[key] = new_id
#     return new_id
#
# def augment_events():
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor(cursor_factory=RealDictCursor)
#
#     cur.execute("""
#         SELECT id, type, description, properties
#         FROM agi_evolution.trigger_event
#         WHERE metadata->>'source' = 'manual'
#     """)
#     base_events = cur.fetchall()
#
#     emotion_ids = get_existing_emotion_ids(conn)
#
#     for ev in base_events:
#         ev_id = ev['id']
#         ev_type = ev['type']
#         ev_desc = ev['description']
#         ev_props = ev['properties'] or {}
#
#         print(f"🔁 Аугментация события: {ev_type} - {ev_desc}")
#
#         variations = random.randint(5, 8)
#         for i in range(variations):
#             new_props = ev_props.copy()
#             for key in ['attractiveness', 'intensity', 'number_of_participants']:
#                 if key in new_props and isinstance(new_props[key], (int, float)):
#                     factor = random.uniform(0.6, 1.4)
#                     new_props[key] = round(new_props[key] * factor, 2)
#             if random.random() < 0.3:
#                 new_props['random_factor'] = round(random.uniform(0.1, 1.0), 2)
#
#             props_str = ", ".join([f"{k}={v}" for k, v in new_props.items() if not k.startswith('_')])
#             full_desc = f"{ev_desc} (with {props_str})" if props_str else ev_desc
#
#             prompt = f"""
# Ты — эксперт по психологии эмоций.
# Дано событие: {full_desc}
# Свойства события: {json.dumps(new_props, ensure_ascii=False)}
# Какая эмоция из списка {EMOTION_TYPES} лучше всего соответствует этому событию?
# Также укажи вероятность (0-1) возникновения этой эмоции и коэффициент интенсивности (0.5-2.0).
# Ответь ТОЛЬКО в формате JSON: {{"emotion": "...", "probability": 0.0-1.0, "intensity_factor": 0.5-2.0}}
# Без пояснений.
# """
#             try:
#                 response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
#                 content = response['message']['content'].strip()
#                 result = extract_json(content)
#                 if result is None:
#                     print(f"  ❌ Невалидный JSON: {content[:100]}")
#                     continue
#                 emotion_type = result.get('emotion', 'trust')
#                 probability = min(1.0, max(0.0, result.get('probability', 0.5)))
#                 intensity = min(2.0, max(0.5, result.get('intensity_factor', 1.0)))
#
#                 emotion_desc = f"Аугментированная {emotion_type} для {ev_desc[:30]}"
#                 emotion_id = create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids)
#
#                 new_event_id = str(uuid.uuid4())
#                 cur.execute("""
#                     INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
#                     VALUES (%s, %s, %s, %s, %s)
#                 """, (new_event_id, ev_type, full_desc, Json(new_props), Json({'augmented': True, 'source_event': ev_id})))
#
#                 cur.execute("""
#                     INSERT INTO agi_evolution.event_emotion_link (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
#                     VALUES (%s, %s, %s, %s, %s, %s)
#                 """, (new_event_id, emotion_id, 'event_emotion', probability, intensity, Json({'generated_by': 'llm', 'variation': i+1})))
#
#                 conn.commit()
#                 print(f"  ✅ Вариация {i+1}: {emotion_type} (p={probability:.2f}, intensity={intensity:.2f})")
#                 time.sleep(0.5)
#
#             except Exception as e:
#                 print(f"  ❌ Ошибка вариации {i+1}: {e}")
#                 conn.rollback()
#
#     cur.close()
#     conn.close()
#     print("✅ Аугментация завершена.")
#
# if __name__ == '__main__':
#     augment_events()




# import json
# import uuid
# import random
# import time
# import psycopg2
# from psycopg2.extras import Json, RealDictCursor
# import ollama
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
# EMOTION_TYPES = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'anticipation', 'empathy', 'interest', 'boredom']
#
# def get_existing_emotion_ids(conn):
#     cur = conn.cursor()
#     cur.execute("SELECT id, type, description FROM agi_evolution.emotion_respons")
#     rows = cur.fetchall()
#     cur.close()
#     # Индексируем по type:description (чтобы можно было искать)
#     return {f"{row[1]}:{row[2]}": row[0] for row in rows}
#
# def create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids):
#     key = f"{emotion_type}:{emotion_desc}"
#     if key in emotion_ids:
#         return emotion_ids[key]
#     new_id = str(uuid.uuid4())
#     cur = conn.cursor()
#     cur.execute("""
#         INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
#         VALUES (%s, %s, %s, %s, %s)
#     """, (new_id, emotion_type, emotion_desc, Json({}), Json({'generated_by': 'llm'})))
#     conn.commit()
#     cur.close()
#     emotion_ids[key] = new_id
#     return new_id
#
# def augment_events():
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor(cursor_factory=RealDictCursor)
#
#     # Получаем все базовые события (только те, что созданы вручную, чтобы не аугментировать аугментированные)
#     cur.execute("""
#         SELECT id, type, description, properties
#         FROM agi_evolution.trigger_event
#         WHERE metadata->>'source' = 'manual'
#     """)
#     base_events = cur.fetchall()
#
#     emotion_ids = get_existing_emotion_ids(conn)
#
#     for ev in base_events:
#         ev_id = ev['id']
#         ev_type = ev['type']
#         ev_desc = ev['description']
#         ev_props = ev['properties'] or {}
#
#         print(f"🔁 Аугментация события: {ev_type} - {ev_desc}")
#
#         # Генерируем 5–8 вариаций
#         variations = random.randint(5, 8)
#         for i in range(variations):
#             # Модифицируем свойства
#             new_props = ev_props.copy()
#             # Случайно изменяем числовые параметры (если есть)
#             for key in ['attractiveness', 'intensity', 'number_of_participants']:
#                 if key in new_props and isinstance(new_props[key], (int, float)):
#                     factor = random.uniform(0.6, 1.4)
#                     new_props[key] = round(new_props[key] * factor, 2)
#             # Добавляем случайный параметр
#             if random.random() < 0.3:
#                 new_props['random_factor'] = round(random.uniform(0.1, 1.0), 2)
#
#             # Формируем описание с учётом свойств
#             props_str = ", ".join([f"{k}={v}" for k, v in new_props.items() if not k.startswith('_')])
#             full_desc = f"{ev_desc} (with {props_str})" if props_str else ev_desc
#
#             # Запрос к LLM
#             prompt = f"""
# Ты — эксперт по психологии эмоций.
# Дано событие: {full_desc}
# Свойства события: {json.dumps(new_props, ensure_ascii=False)}
# Какая эмоция из списка {EMOTION_TYPES} лучше всего соответствует этому событию?
# Также укажи вероятность (0-1) возникновения этой эмоции и коэффициент интенсивности (0.5-2.0).
# Ответь JSON: {{"emotion": "...", "probability": 0.0-1.0, "intensity_factor": 0.5-2.0}}
# """
#             try:
#                 response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
#                 result = json.loads(response['message']['content'])
#                 emotion_type = result.get('emotion', 'trust')
#                 probability = min(1.0, max(0.0, result.get('probability', 0.5)))
#                 intensity = min(2.0, max(0.5, result.get('intensity_factor', 1.0)))
#
#                 # Создаём или получаем эмоцию
#                 emotion_desc = f"Аугментированная {emotion_type} для {ev_desc[:30]}"
#                 emotion_id = create_or_get_emotion(conn, emotion_type, emotion_desc, emotion_ids)
#
#                 # Создаём новое событие (вариацию)
#                 new_event_id = str(uuid.uuid4())
#                 cur.execute("""
#                     INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
#                     VALUES (%s, %s, %s, %s, %s)
#                 """, (new_event_id, ev_type, full_desc, Json(new_props), Json({'augmented': True, 'source_event': ev_id})))
#
#                 # Добавляем связь
#                 cur.execute("""
#                     INSERT INTO agi_evolution.event_emotion_links (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
#                     VALUES (%s, %s, %s, %s, %s, %s)
#                 """, (new_event_id, emotion_id, 'event_emotion', probability, intensity, Json({'generated_by': 'llm', 'variation': i+1})))
#
#                 conn.commit()
#                 print(f"  ✅ Вариация {i+1}: {emotion_type} (p={probability:.2f}, intensity={intensity:.2f})")
#                 time.sleep(0.5)  # пауза, чтобы не перегружать LLM
#
#             except Exception as e:
#                 print(f"  ❌ Ошибка вариации {i+1}: {e}")
#                 conn.rollback()
#
#     cur.close()
#     conn.close()
#     print("✅ Аугментация завершена.")
#
# if __name__ == '__main__':
#     augment_events()