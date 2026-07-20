#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Импорт данных из GraphRAG (entities.parquet, relationships.parquet) в БД.
Для каждого запуска задаётся theme_id, чтобы можно было легко фильтровать данные по темам.
"""

import pandas as pd
import psycopg2
import uuid
import json
import os
import numpy as np  # ← ДОБАВЛЯЕМ ИМПОРТ
from psycopg2.extras import execute_values

# ============================================================
# НАСТРОЙКИ (измените перед запуском)
# ============================================================

# Тема/раздел знаний (например, 'history_v1', 'aerodynamics_v1')
THEME_ID = "solid_state_physics_v1"

# Путь к папке с Parquet-файлами (обычно output/)
OUTPUT_DIR = "output"

# Подключение к БД
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}


# ============================================================
# СКРИПТ
# ============================================================

def main():
    # Проверяем наличие файлов
    entities_path = os.path.join(OUTPUT_DIR, "entities.parquet")
    relationships_path = os.path.join(OUTPUT_DIR, "relationships.parquet")

    if not os.path.exists(entities_path):
        print(f"❌ Файл {entities_path} не найден.")
        return
    if not os.path.exists(relationships_path):
        print(f"❌ Файл {relationships_path} не найден.")
        return

    # Подключаемся к БД
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print(f"📥 Импорт данных для темы: {THEME_ID}")

    # 1. Читаем Parquet
    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(relationships_path)

    print(f"   Узлов: {len(entities)}, Связей: {len(relationships)}")

    # 2. Импорт узлов
    node_data = []
    for _, row in entities.iterrows():
        node_id = str(uuid.uuid4())
        name = str(row['title'])[:255]
        node_type = str(row['type'])[:50]
        description = str(row.get('description', ''))[:1000]

        # Преобразуем text_unit_ids в список, если это numpy массив
        text_unit_ids = row.get('text_unit_ids', [])
        if isinstance(text_unit_ids, np.ndarray):
            text_unit_ids = text_unit_ids.tolist()

        metadata = {
            'frequency': int(row.get('frequency', 1)),
            'text_unit_ids': text_unit_ids
        }

        node_data.append((
            node_id,
            name,
            node_type,
            description,
            json.dumps({}),  # properties (пустой массив)
            json.dumps({}),  # embedding (пустой массив)
            json.dumps({}),  # parameters (пустой массив)
            json.dumps(metadata),  # metadata (JSON)
            THEME_ID  # theme_id
        ))

    # Вставляем узлы
    if node_data:
        execute_values(cur, """
            INSERT INTO agi_evolution.knowledge_nodes 
            (id, name, node_type, description, properties, embedding, parameters, metadata, theme_id)
            VALUES %s
        """, node_data)
        print(f"   ✅ Импортировано узлов: {len(node_data)}")
    else:
        print("   ⚠️ Нет узлов для импорта.")

    # 3. Создаём словарь для поиска ID узлов по имени
    cur.execute("SELECT id, name FROM agi_evolution.knowledge_nodes")
    rows = cur.fetchall()
    name_to_id = {row[1]: row[0] for row in rows}

    # 4. Импорт связей
    edge_data = []
    skipped = 0
    for _, row in relationships.iterrows():
        source_name = str(row['source'])
        target_name = str(row['target'])
        source_id = name_to_id.get(source_name)
        target_id = name_to_id.get(target_name)

        if source_id and target_id:
            edge_data.append((
                str(uuid.uuid4()),
                source_id,
                target_id,
                'related_to',  # тип связи по умолчанию
                float(row.get('weight', 1.0)),
                str(row.get('description', ''))[:500],
                THEME_ID
            ))
        else:
            skipped += 1

    if edge_data:
        execute_values(cur, """
            INSERT INTO agi_evolution.knowledge_edges 
            (id, source_id, target_id, edge_type, weight, description, theme_id)
            VALUES %s
        """, edge_data)
        print(f"   ✅ Импортировано связей: {len(edge_data)}")
    else:
        print("   ⚠️ Нет связей для импорта.")

    if skipped:
        print(f"   ⚠️ Пропущено связей (не найдены узлы): {skipped}")

    # 5. Записываем в лог импорта (если таблица существует)
    try:
        cur.execute("""
            INSERT INTO agi_evolution.import_log 
            (theme_id, description, nodes_count, edges_count, status, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            THEME_ID,
            f"Импорт из {OUTPUT_DIR}",
            len(node_data),
            len(edge_data),
            'completed',
            f"Узлов: {len(node_data)}, Связей: {len(edge_data)}, Пропущено: {skipped}"
        ))
        conn.commit()
        print(f"   📝 Запись в лог добавлена.")
    except Exception as e:
        # Если таблица import_log не существует — просто игнорируем
        print(f"   ⚠️ Не удалось записать в лог (возможно, таблица не создана): {e}")
        conn.rollback()

    # Закрываем соединение
    cur.close()
    conn.close()
    print("✅ Импорт завершён.")


if __name__ == "__main__":
    main()






# """
# Импорт данных из GraphRAG (entities.parquet, relationships.parquet) в БД.
# Для каждого запуска задаётся theme_id, чтобы можно было легко фильтровать данные по темам.
# """
#
# import pandas as pd
# import psycopg2
# import uuid
# import json
# import os
# from psycopg2.extras import execute_values
#
# # ============================================================
# # НАСТРОЙКИ (измените перед запуском)
# # ============================================================
#
# # Тема/раздел знаний (например, 'history_v1', 'aerodynamics_v1')
# THEME_ID = "aerodynamics_v1"
#
# # Путь к папке с Parquet-файлами (обычно output/)
# OUTPUT_DIR = "output"
#
# # Подключение к БД
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'dbname': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
#
# # ============================================================
# # СКРИПТ
# # ============================================================
#
# def main():
#     # Проверяем наличие файлов
#     entities_path = os.path.join(OUTPUT_DIR, "entities.parquet")
#     relationships_path = os.path.join(OUTPUT_DIR, "relationships.parquet")
#
#     if not os.path.exists(entities_path):
#         print(f"❌ Файл {entities_path} не найден.")
#         return
#     if not os.path.exists(relationships_path):
#         print(f"❌ Файл {relationships_path} не найден.")
#         return
#
#     # Подключаемся к БД
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor()
#
#     print(f"📥 Импорт данных для темы: {THEME_ID}")
#
#     # 1. Читаем Parquet
#     entities = pd.read_parquet(entities_path)
#     relationships = pd.read_parquet(relationships_path)
#
#     print(f"   Узлов: {len(entities)}, Связей: {len(relationships)}")
#
#     # 2. Импорт узлов
#     node_data = []
#     for _, row in entities.iterrows():
#         node_id = str(uuid.uuid4())
#         name = str(row['title'])[:255]
#         node_type = str(row['type'])[:50]
#         description = str(row.get('description', ''))[:1000]
#
#         # Обработка text_unit_ids (преобразование numpy array в список)
#         text_unit_ids = row.get('text_unit_ids', [])
#         if isinstance(text_unit_ids, np.ndarray):
#             text_unit_ids = text_unit_ids.tolist()
#
#         metadata = {
#             'frequency': int(row.get('frequency', 1)),
#             'text_unit_ids': text_unit_ids
#         }
#
#         node_data.append((
#             node_id,
#             name,
#             node_type,
#             description,
#             json.dumps({}),  # properties (пустой массив)
#             json.dumps({}),  # embedding (пустой массив)
#             json.dumps({}),  # parameters (пустой массив)
#             json.dumps(metadata),  # metadata (JSON)
#             THEME_ID  # theme_id
#         ))
#
#
#     # # 2. Импорт узлов
#     # node_data = []
#     # for _, row in entities.iterrows():
#     #     node_id = str(uuid.uuid4())
#     #     name = str(row['title'])[:255]  # обрезаем до 255 символов
#     #     node_type = str(row['type'])[:50]  # обрезаем до 50
#     #     description = str(row.get('description', ''))[:1000]
#     #
#     #     # Метаданные: frequency и text_unit_ids
#     #     metadata = {
#     #         'frequency': int(row.get('frequency', 1)),
#     #         'text_unit_ids': row.get('text_unit_ids', [])
#     #     }
#     #
#     #     node_data.append((
#     #         node_id,
#     #         name,
#     #         node_type,
#     #         description,
#     #         json.dumps({}),  # properties (пустой массив)
#     #         json.dumps({}),  # embedding (пустой массив)
#     #         json.dumps({}),  # parameters (пустой массив)
#     #         json.dumps(metadata),  # metadata (JSON)
#     #         THEME_ID  # theme_id
#     #     ))
#
#     # Вставляем узлы
#     if node_data:
#         execute_values(cur, """
#             INSERT INTO agi_evolution.knowledge_nodes
#             (id, name, node_type, description, properties, embedding, parameters, metadata, theme_id)
#             VALUES %s
#         """, node_data)
#         print(f"   ✅ Импортировано узлов: {len(node_data)}")
#     else:
#         print("   ⚠️ Нет узлов для импорта.")
#
#     # 3. Создаём словарь для поиска ID узлов по имени
#     cur.execute("SELECT id, name FROM agi_evolution.knowledge_nodes")
#     rows = cur.fetchall()
#     name_to_id = {row[1]: row[0] for row in rows}
#
#     # 4. Импорт связей
#     edge_data = []
#     skipped = 0
#     for _, row in relationships.iterrows():
#         source_name = str(row['source'])
#         target_name = str(row['target'])
#         source_id = name_to_id.get(source_name)
#         target_id = name_to_id.get(target_name)
#
#         if source_id and target_id:
#             edge_data.append((
#                 str(uuid.uuid4()),
#                 source_id,
#                 target_id,
#                 'related_to',  # тип связи по умолчанию
#                 float(row.get('weight', 1.0)),
#                 str(row.get('description', ''))[:500],
#                 THEME_ID
#             ))
#         else:
#             skipped += 1
#
#     if edge_data:
#         execute_values(cur, """
#             INSERT INTO agi_evolution.knowledge_edges
#             (id, source_id, target_id, edge_type, weight, description, theme_id)
#             VALUES %s
#         """, edge_data)
#         print(f"   ✅ Импортировано связей: {len(edge_data)}")
#     else:
#         print("   ⚠️ Нет связей для импорта.")
#
#     if skipped:
#         print(f"   ⚠️ Пропущено связей (не найдены узлы): {skipped}")
#
#     # 5. Записываем в лог импорта (если таблица существует)
#     try:
#         cur.execute("""
#             INSERT INTO agi_evolution.import_log
#             (theme_id, description, nodes_count, edges_count, status, notes)
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (
#             THEME_ID,
#             f"Импорт из {OUTPUT_DIR}",
#             len(node_data),
#             len(edge_data),
#             'completed',
#             f"Узлов: {len(node_data)}, Связей: {len(edge_data)}, Пропущено: {skipped}"
#         ))
#         conn.commit()
#         print(f"   📝 Запись в лог добавлена.")
#     except Exception as e:
#         # Если таблица import_log не существует — просто игнорируем
#         print(f"   ⚠️ Не удалось записать в лог (возможно, таблица не создана): {e}")
#         conn.rollback()
#
#     # Закрываем соединение
#     cur.close()
#     conn.close()
#     print("✅ Импорт завершён.")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
# # import pandas as pd
# # import psycopg2
# # import uuid
# # import json
# # from psycopg2.extras import execute_values
# #
# # # Подключение к БД
# # conn = psycopg2.connect(
# #     host='localhost',
# #     port=5432,
# #     dbname='postgres',
# #     user='postgres',
# #     password='postgres'
# # )
# # cur = conn.cursor()
# #
# # # Читаем Parquet
# # entities = pd.read_parquet("output/entities.parquet")
# # relationships = pd.read_parquet("output/relationships.parquet")
# #
# # # Импорт узлов
# # node_data = []
# # for _, row in entities.iterrows():
# #     node_id = str(uuid.uuid4())
# #     name = row['title'][:255]
# #     node_type = row['type'][:50]
# #     description = row.get('description', '')[:1000]
# #
# #     # Преобразуем text_unit_ids (numpy array) в список
# #     text_unit_ids = row.get('text_unit_ids', [])
# #     if hasattr(text_unit_ids, 'tolist'):
# #         text_unit_ids = text_unit_ids.tolist()
# #     elif isinstance(text_unit_ids, (list, tuple)):
# #         text_unit_ids = list(text_unit_ids)
# #     else:
# #         text_unit_ids = []
# #
# #     metadata = {
# #         'frequency': int(row.get('frequency', 1)),
# #         'text_unit_ids': text_unit_ids
# #     }
# #
# #     node_data.append((
# #         node_id,
# #         name,
# #         node_type,
# #         description,
# #         json.dumps({}),  # properties
# #         json.dumps({}),  # embedding
# #         json.dumps({}),  # parameters
# #         json.dumps(metadata)  # metadata — теперь сериализуется корректно
# #     ))
# #
# # execute_values(cur, """
# #     INSERT INTO agi_evolution.knowledge_nodes
# #     (id, name, node_type, description, properties, embedding, parameters, metadata)
# #     VALUES %s
# # """, node_data)
# #
# # # Создаём словарь для поиска ID узлов по имени
# # cur.execute("SELECT id, name FROM agi_evolution.knowledge_nodes")
# # rows = cur.fetchall()
# # name_to_id = {row[1]: row[0] for row in rows}
# #
# # # Импорт связей
# # edge_data = []
# # for _, row in relationships.iterrows():
# #     source_id = name_to_id.get(row['source'])
# #     target_id = name_to_id.get(row['target'])
# #     if source_id and target_id:
# #         edge_data.append((
# #             str(uuid.uuid4()),
# #             source_id,
# #             target_id,
# #             'related_to',
# #             float(row.get('weight', 1.0)),
# #             row.get('description', '')[:500]
# #         ))
# #
# # if edge_data:
# #     execute_values(cur, """
# #         INSERT INTO agi_evolution.knowledge_edges
# #         (id, source_id, target_id, edge_type, weight, description)
# #         VALUES %s
# #     """, edge_data)
# #
# # conn.commit()
# # cur.close()
# # conn.close()
# #
# # print(f"✅ Импортировано {len(node_data)} узлов и {len(edge_data)} связей.")
# #
# #
# #
# #
# # # import pandas as pd
# # # import psycopg2
# # # import uuid
# # # import json
# # # from psycopg2.extras import execute_values
# # #
# # # # Подключение к БД
# # # conn = psycopg2.connect(
# # #     host='localhost',
# # #     port=5432,
# # #     dbname='postgres',
# # #     user='postgres',
# # #     password='postgres'
# # # )
# # # cur = conn.cursor()
# # #
# # # # Читаем Parquet
# # # entities = pd.read_parquet("output/entities.parquet")
# # # relationships = pd.read_parquet("output/relationships.parquet")
# # #
# # # # Импорт узлов
# # # node_data = []
# # # for _, row in entities.iterrows():
# # #     node_id = str(uuid.uuid4())
# # #     # Обрезаем название до 255 символов (на всякий случай)
# # #     name = row['title'][:255]
# # #     node_type = row['type'][:50]
# # #     description = row.get('description', '')[:1000]  # ограничим описание
# # #     # Собираем метаданные: frequency и text_unit_ids
# # #     metadata = {
# # #         'frequency': int(row.get('frequency', 1)),
# # #         'text_unit_ids': row.get('text_unit_ids', [])
# # #     }
# # #     # Вставляем только нужные поля
# # #     node_data.append((
# # #         node_id,
# # #         name,
# # #         node_type,
# # #         description,
# # #         json.dumps({}),  # properties — пустой JSON
# # #         json.dumps({}),  # embedding — пустой JSON
# # #         json.dumps({}),  # parameters — пустой JSON
# # #         json.dumps(metadata)  # metadata — JSON с frequency и text_unit_ids
# # #     ))
# # #
# # # execute_values(cur, """
# # #     INSERT INTO agi_evolution.knowledge_nodes
# # #     (id, name, node_type, description, properties, embedding, parameters, metadata)
# # #     VALUES %s
# # # """, node_data)
# # #
# # # # Импорт связей
# # # # Сначала создадим словарь для поиска ID узлов по имени
# # # cur.execute("SELECT id, name FROM agi_evolution.knowledge_nodes")
# # # rows = cur.fetchall()
# # # name_to_id = {row[1]: row[0] for row in rows}
# # #
# # # edge_data = []
# # # for _, row in relationships.iterrows():
# # #     source_name = row['source']
# # #     target_name = row['target']
# # #     source_id = name_to_id.get(source_name)
# # #     target_id = name_to_id.get(target_name)
# # #     if source_id and target_id:
# # #         edge_data.append((
# # #             str(uuid.uuid4()),
# # #             source_id,
# # #             target_id,
# # #             'related_to',  # тип связи по умолчанию
# # #             float(row.get('weight', 1.0)),
# # #             row.get('description', '')[:500]
# # #         ))
# # #
# # # execute_values(cur, """
# # #     INSERT INTO agi_evolution.knowledge_edges
# # #     (id, source_id, target_id, edge_type, weight, description)
# # #     VALUES %s
# # # """, edge_data)
# # #
# # # conn.commit()
# # # cur.close()
# # # conn.close()
# # #
# # # print(f"✅ Импортировано {len(node_data)} узлов и {len(edge_data)} связей.")
# # #
# # #
# # #
# # #
# # # # import pandas as pd
# # # # import psycopg2
# # # # import uuid
# # # # import json
# # # # from psycopg2.extras import execute_values
# # # #
# # # # # Подключение к БД (настройте под себя)
# # # # conn = psycopg2.connect(
# # # #     host='localhost',
# # # #     port=5432,
# # # #     dbname='postgres',
# # # #     user='postgres',
# # # #     password='postgres'
# # # # )
# # # # cur = conn.cursor()
# # # #
# # # # # Читаем Parquet
# # # # entities = pd.read_parquet("output/entities.parquet")
# # # # relationships = pd.read_parquet("output/relationships.parquet")
# # # #
# # # # # ============================================================
# # # # # 1. ИМПОРТ УЗЛОВ
# # # # # ============================================================
# # # # node_data = []
# # # # for _, row in entities.iterrows():
# # # #     node_id = str(uuid.uuid4())
# # # #     name = row['title']
# # # #     node_type = row['type']  # PERSON, ORGANIZATION, GEO, EVENT и т.д.
# # # #     properties = []  # пока пустой, можно потом заполнить
# # # #     description = row.get('description', '')
# # # #     # Если хотим сохранить частоту, можно в metadata
# # # #     metadata = {'frequency': int(row.get('frequency', 1))}
# # # #
# # # #     node_data.append((
# # # #         node_id,
# # # #         name,
# # # #         node_type,
# # # #         properties,
# # # #         description,
# # # #         json.dumps(metadata)  # или просто оставить пустым
# # # #     ))
# # # #
# # # # execute_values(cur, """
# # # #     INSERT INTO agi_evolution.knowledge_nodes
# # # #     (id, name, node_type, properties, description, metadata)
# # # #     VALUES %s
# # # # """, node_data)
# # # # print(f"✅ Импортировано {len(node_data)} узлов")
# # # #
# # # # # ============================================================
# # # # # 2. ИМПОРТ СВЯЗЕЙ
# # # # # ============================================================
# # # # # Сначала создадим словарь имя -> ID для быстрого поиска
# # # # cur.execute("SELECT id, name FROM agi_evolution.knowledge_nodes")
# # # # rows = cur.fetchall()
# # # # name_to_id = {row[1]: row[0] for row in rows}
# # # #
# # # # edge_data = []
# # # # for _, row in relationships.iterrows():
# # # #     source_name = row['source']
# # # #     target_name = row['target']
# # # #     source_id = name_to_id.get(source_name)
# # # #     target_id = name_to_id.get(target_name)
# # # #     if source_id is None:
# # # #         print(f"⚠️ Не найден узел-источник: {source_name}")
# # # #         continue
# # # #     if target_id is None:
# # # #         print(f"⚠️ Не найден узел-цель: {target_name}")
# # # #         continue
# # # #     edge_type = 'related_to'  # Можно задать что-то более конкретное, если есть информация
# # # #     weight = float(row.get('weight', 1.0))
# # # #     description = row.get('description', '')
# # # #     # Можно также добавить текстовые ID в metadata
# # # #     metadata = {}
# # # #
# # # #     edge_data.append((
# # # #         str(uuid.uuid4()),
# # # #         source_id,
# # # #         target_id,
# # # #         edge_type,
# # # #         weight,
# # # #         description,
# # # #         json.dumps(metadata)
# # # #     ))
# # # #
# # # # execute_values(cur, """
# # # #     INSERT INTO agi_evolution.knowledge_edges
# # # #     (id, source_id, target_id, edge_type, weight, description, metadata)
# # # #     VALUES %s
# # # # """, edge_data)
# # # # print(f"✅ Импортировано {len(edge_data)} связей")
# # # #
# # # # conn.commit()
# # # # cur.close()
# # # # conn.close()
# # # # print("🎉 Импорт завершён.")