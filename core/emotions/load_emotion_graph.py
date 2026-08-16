#core/emotions/load_emotion_graph.py
import json
import uuid
import psycopg2
from psycopg2.extras import Json
from typing import List, Dict

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def load_events(conn, events_data: List[Dict]):
    cur = conn.cursor()
    event_ids = {}
    for ev in events_data:
        ev_id = str(uuid.uuid4())
        props_json = json.dumps(ev.get('properties', {}), ensure_ascii=False)
        metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
        cur.execute("""
            INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (ev_id, ev['type'], ev['description'], props_json, metadata_json))
        event_ids[ev['description']] = ev_id
    cur.close()
    conn.commit()
    return event_ids

def load_emotions(conn, emotions_data: List[Dict]):
    cur = conn.cursor()
    emotion_ids = {}
    for em in emotions_data:
        em_id = str(uuid.uuid4())
        key = f"{em['type']}:{em['description']}"
        props_json = json.dumps(em.get('properties', {}), ensure_ascii=False)
        metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
        cur.execute("""
            INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (em_id, em['type'], em['description'], props_json, metadata_json))
        emotion_ids[key] = em_id
    cur.close()
    conn.commit()
    return emotion_ids

def load_links(conn, links_data: List[Dict], event_ids: Dict, emotion_ids: Dict):
    cur = conn.cursor()
    # Создаём словарь {type: id} из emotion_ids
    type_to_id = {}
    for key, eid in emotion_ids.items():
        parts = key.split(':', 1)
        if len(parts) == 2:
            em_type = parts[0]
            if em_type not in type_to_id:
                type_to_id[em_type] = eid

    print("🔍 Доступные типы эмоций в БД:")
    for t, eid in type_to_id.items():
        print(f"  {t} -> {eid}")

    print("🔍 Доступные события в БД:")
    for desc, eid in event_ids.items():
        print(f"  {desc} -> {eid}")

    for link in links_data:
        event_desc = link.get('event')
        emotion_type = link.get('emotion')
        if not event_desc or not emotion_type:
            print("⚠️ Пропущена связь с пустым полем")
            continue
        if event_desc not in event_ids:
            print(f"⚠️ Событие '{event_desc}' не найдено")
            continue
        emotion_id = type_to_id.get(emotion_type)
        if not emotion_id:
            print(f"⚠️ Эмоция типа '{emotion_type}' не найдена")
            continue
        probability = link.get('probability', 0.9)
        intensity = link.get('intensity_factor', 1.0)
        metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
        cur.execute("""
            INSERT INTO agi_evolution.event_emotion_link (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (event_ids[event_desc], emotion_id, 'event_emotion', probability, intensity, metadata_json))
        print(f"✅ Связь: {event_desc} → {emotion_type}")
    cur.close()
    conn.commit()
    print("✅ Связи загружены.")

def main():
    with open('training_data_invention_trigger_events.json', 'r', encoding='utf-8') as f:
        events = json.load(f)
    with open('training_data_invention_emotions.json', 'r', encoding='utf-8') as f:
        emotions = json.load(f)
    with open('training_data_invention_bigraph_links_my.json', 'r', encoding='utf-8') as f:
        links = json.load(f)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        event_ids = load_events(conn, events)
        emotion_ids = load_emotions(conn, emotions)
        load_links(conn, links, event_ids, emotion_ids)
        print(f"✅ Загружено {len(event_ids)} событий, {len(emotion_ids)} эмоций, {len(links)} связей.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    main()




# import json
# import uuid
# import psycopg2
# from psycopg2.extras import Json, RealDictCursor
# from typing import List, Dict, Any
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
# def load_events(conn, events_data: List[Dict]):
#     """Загружает события (триггеры) в таблицу trigger_event."""
#     cur = conn.cursor()
#     event_ids = {}
#     for ev in events_data:
#         ev_id = str(uuid.uuid4())
#         # Сериализуем JSON с ensure_ascii=False для сохранения кириллицы
#         props_json = json.dumps(ev.get('properties', {}), ensure_ascii=False)
#         metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
#         cur.execute("""
#             INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (
#             ev_id,
#             ev['type'],
#             ev['description'],
#             props_json,  # передаём как строку JSON
#             metadata_json
#         ))
#         event_ids[ev['description']] = ev_id
#     cur.close()
#     conn.commit()
#     return event_ids
#
# def load_emotions(conn, emotions_data: List[Dict]):
#     """Загружает эмоциональные реакции в таблицу emotion_respons."""
#     cur = conn.cursor()
#     emotion_ids = {}
#     for em in emotions_data:
#         em_id = str(uuid.uuid4())
#         key = f"{em['type']}:{em['description']}"
#         props_json = json.dumps(em.get('properties', {}), ensure_ascii=False)
#         metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
#         cur.execute("""
#             INSERT INTO agi_evolution.emotion_respons (id, type, description, properties, metadata)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (
#             em_id,
#             em['type'],
#             em['description'],
#             props_json,
#             metadata_json
#         ))
#         emotion_ids[key] = em_id
#     cur.close()
#     conn.commit()
#     return emotion_ids
#
#
# def load_links(conn, links_data: List[Dict], event_ids: Dict, emotion_ids: Dict):
#     """Загружает связи событие ↔ эмоция в таблицу event_emotion_links."""
#     cur = conn.cursor()
#     # Создаём словарь {type: id} из emotion_ids (ключи "type:description")
#     type_to_id = {}
#     for key, eid in emotion_ids.items():
#         parts = key.split(':', 1)
#         if len(parts) == 2:
#             em_type = parts[0]
#             if em_type not in type_to_id:
#                 type_to_id[em_type] = eid
#     for link in links_data:
#         event_desc = link.get('event')
#         emotion_type = link.get('emotion')
#         if not event_desc or not emotion_type:
#             continue
#         if event_desc not in event_ids:
#             print(f"⚠️ Событие '{event_desc}' не найдено")
#             continue
#         emotion_id = type_to_id.get(emotion_type)
#         if not emotion_id:
#             print(f"⚠️ Эмоция '{emotion_type}' не найдена")
#             continue
#         probability = link.get('probability', 0.9)
#         intensity = link.get('intensity_factor', 1.0)
#         metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
#         cur.execute("""
#             INSERT INTO agi_evolution.event_emotion_link (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (
#             event_ids[event_desc],
#             emotion_id,
#             'event_emotion',
#             probability,
#             intensity,
#             metadata_json
#         ))
#     cur.close()
#     conn.commit()
#     print("✅ Связи загружены.")
#
# # def load_links(conn, links_data: List[Dict], event_ids: Dict, emotion_ids: Dict):
# #     """Загружает связи событие ↔ эмоция в таблицу event_emotion_links."""
# #     cur = conn.cursor()
# #     # Создаём отображение для быстрого поиска эмоции по описанию и типу
# #     # emotion_ids уже имеет ключи вида "type:description", но в links у нас только описание эмоции.
# #     # Поэтому создадим словарь: (type, description) -> id, но мы не знаем type в links.
# #     # Вместо этого для каждого link попробуем найти эмоцию по описанию, если она уникальна.
# #     # Или будем искать по первому совпадению.
# #     # Создадим словарь {description: id} для всех эмоций (если описания уникальны).
# #     desc_to_id = {}
# #     for key, eid in emotion_ids.items():
# #         # key = "type:description", извлекаем description
# #         parts = key.split(':', 1)
# #         if len(parts) == 2:
# #             desc = parts[1]
# #             # Если несколько эмоций с одинаковым описанием, возьмём первую
# #             if desc not in desc_to_id:
# #                 desc_to_id[desc] = eid
# #
# #     for link in links_data:
# #         event_desc = link.get('event')
# #         emotion_desc = link.get('emotion')
# #         if not event_desc or not emotion_desc:
# #             continue
# #         if event_desc not in event_ids:
# #             print(f"⚠️ Событие '{event_desc}' не найдено")
# #             continue
# #         # Ищем ID эмоции по описанию
# #         emotion_id = desc_to_id.get(emotion_desc)
# #         if not emotion_id:
# #             print(f"⚠️ Эмоция '{emotion_desc}' не найдена")
# #             continue
# #         # Добавляем вероятность и интенсивность из связей, если есть
# #         probability = link.get('probability', 0.9)
# #         intensity = link.get('intensity_factor', 1.0)
# #         metadata_json = json.dumps({'source': 'manual'}, ensure_ascii=False)
# #         cur.execute("""
# #             INSERT INTO agi_evolution.event_emotion_links (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
# #             VALUES (%s, %s, %s, %s, %s, %s)
# #         """, (
# #             event_ids[event_desc],
# #             emotion_id,
# #             'event_emotion',
# #             probability,
# #             intensity,
# #             metadata_json
# #         ))
# #     cur.close()
# #     conn.commit()
# #     print("✅ Связи загружены.")
#
# def main():
#     # Читаем JSON-файлы
#     with open('training_data_invention_trigger_events.json', 'r', encoding='utf-8') as f:
#         events = json.load(f)
#     with open('training_data_invention_emotions.json', 'r', encoding='utf-8') as f:
#         emotions = json.load(f)
#     with open('training_data_invention_bigraph_links_my.json', 'r', encoding='utf-8') as f:
#         links = json.load(f)
#
#     conn = psycopg2.connect(**DB_CONFIG)
#     try:
#         event_ids = load_events(conn, events)
#         emotion_ids = load_emotions(conn, emotions)
#         load_links(conn, links, event_ids, emotion_ids)
#         print(f"✅ Загружено {len(event_ids)} событий, {len(emotion_ids)} эмоций, {len(links)} связей.")
#     except Exception as e:
#         print(f"❌ Ошибка: {e}")
#         conn.rollback()
#     finally:
#         conn.close()
#
# if __name__ == '__main__':
#     main()



# import json
# import uuid
# import psycopg2
# from psycopg2.extras import Json, RealDictCursor
# from typing import List, Dict, Any
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
# def load_events(conn, events_data: List[Dict]):
#     """Загружает события (триггеры) в таблицу trigger_event."""
#     cur = conn.cursor()
#     event_ids = {}
#     for ev in events_data:
#         ev_id = str(uuid.uuid4())
#         cur.execute("""
#             INSERT INTO agi_evolution.trigger_event (id, type, description, properties, metadata)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (
#             ev_id,
#             ev['type'],
#             ev['description'],
#             Json(ev.get('properties', {})),
#             Json({'source': 'manual'})
#         ))
#         event_ids[ev['description']] = ev_id  # используем описание как ключ для связей
#     cur.close()
#     conn.commit()
#     return event_ids
#
# def load_emotions(conn, emotions_data: List[Dict]):
#     """Загружает эмоциональные реакции в таблицу emotion_responses."""
#     cur = conn.cursor()
#     emotion_ids = {}
#     for em in emotions_data:
#         em_id = str(uuid.uuid4())
#         # Для уникального ключа используем type + description (если несколько эмоций одного типа)
#         key = f"{em['type']}:{em['description']}"
#         cur.execute("""
#             INSERT INTO agi_evolution.emotion_responses (id, emotion_type, description, properties, metadata)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (
#             em_id,
#             em['type'],
#             em['description'],
#             Json(em.get('properties', {})),
#             Json({'source': 'manual'})
#         ))
#         emotion_ids[key] = em_id
#     cur.close()
#     conn.commit()
#     return emotion_ids
#
# def load_links(conn, links_data: List[Dict], event_ids: Dict, emotion_ids: Dict):
#     """Загружает связи событие ↔ эмоция в таблицу event_emotion_links."""
#     cur = conn.cursor()
#     for link in links_data:
#         event_desc = link.get('event')
#         emotion_desc = link.get('emotion')
#         if not event_desc or not emotion_desc:
#             continue  # пропускаем пустые строки
#         event_key = event_desc  # так как event_ids индексированы по description
#         emotion_key = f"{emotion_desc}:{link.get('emotion_description', '')}"
#         # Если в связях не указано описание эмоции, ищем по типу
#         if emotion_key not in emotion_ids:
#             # Пробуем найти по первому совпадению типа
#             for key in emotion_ids:
#                 if key.startswith(emotion_desc + ':'):
#                     emotion_key = key
#                     break
#         if event_key not in event_ids or emotion_key not in emotion_ids:
#             print(f"⚠️ Пропущена связь: {event_desc} → {emotion_desc} (не найдены ID)")
#             continue
#         cur.execute("""
#             INSERT INTO agi_evolution.event_emotion_link (event_id, emotion_id, link_type, probability, intensity_factor, metadata)
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (
#             event_ids[event_key],
#             emotion_ids[emotion_key],
#             'event_emotion',
#             0.9,  # вероятность по умолчанию
#             1.0,  # интенсивность по умолчанию
#             Json({'source': 'manual'})
#         ))
#     cur.close()
#     conn.commit()
#     print("✅ Связи загружены.")
#
# def main():
#     # Читаем JSON-файлы
#     with open('training_data_invention_trigger_events.json', 'r', encoding='utf-8') as f:
#         events = json.load(f)
#     with open('training_data_invention_emotions.json', 'r', encoding='utf-8') as f:
#         emotions = json.load(f)
#     with open('training_data_invention_bigraph_links_ds.json', 'r', encoding='utf-8') as f:
#         links = json.load(f)
#
#     conn = psycopg2.connect(**DB_CONFIG)
#     try:
#         event_ids = load_events(conn, events)
#         emotion_ids = load_emotions(conn, emotions)
#         load_links(conn, links, event_ids, emotion_ids)
#         print(f"✅ Загружено {len(event_ids)} событий, {len(emotion_ids)} эмоций, {len(links)} связей.")
#     except Exception as e:
#         print(f"❌ Ошибка: {e}")
#         conn.rollback()
#     finally:
#         conn.close()
#
# if __name__ == '__main__':
#     main()