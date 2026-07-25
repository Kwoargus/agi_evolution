# db/connector.py
"""
[ru] Модуль для работы с БД.
[en] Module for working with the database.
"""

import json
import psycopg2
import psycopg2.extras
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union


def get_connection(host='localhost', port=5432,
                   dbname='postgres', user='postgres',
                   password='postgres'):
    """
    [ru] Возвращает соединение с БД.
    [en] Returns a database connection.
    """
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )


def load_reflex_rules(schema='agi_evolution'):
    """
    [ru] Загружает правила рефлексов из таблицы reflex_pattern.
    [en] Loads reflex rules from the reflex_pattern table.
    """
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT sense_type, signal_type, signal_threshold, action FROM {schema}.reflex_pattern")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[ru] Ошибка загрузки рефлексов: {e}")
        print(f"[en] Error loading reflexes: {e}")
        return []


def load_instinct_patterns(schema='agi_evolution'):
    """
    [ru] Загружает паттерны инстинктов из таблицы instinct_pattern.
    [en] Loads instinct patterns from the instinct_pattern table.
    """
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT pattern, actions FROM {schema}.instinct_pattern")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        patterns = []
        for row in rows:
            pattern = row['pattern'] if isinstance(row['pattern'], dict) else json.loads(row['pattern'])
            action = row['actions'] if isinstance(row['actions'], dict) else json.loads(row['actions'])
            patterns.append({'pattern': pattern, 'action': action})
        return patterns
    except Exception as e:
        print(f"[ru] Ошибка загрузки инстинктов: {e}")
        print(f"[en] Error loading instincts: {e}")
        return []


def save_generation(generation_num, best_fitness, avg_fitness, schema='agi_evolution'):
    """
    [ru] Сохраняет поколение и возвращает его id.
    [en] Saves a generation and returns its id.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO {schema}.generations (generation_num, best_fitness, avg_fitness) VALUES (%s, %s, %s) RETURNING id",
            (generation_num, best_fitness, avg_fitness)
        )
        gen_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return gen_id
    except Exception as e:
        print(f"[ru] Ошибка сохранения поколения: {e}")
        print(f"[en] Error saving generation: {e}")
        return None


def save_genome(generation_id, genome_json, fitness, schema='agi_evolution'):
    """
    [ru] Сохраняет геном.
    [en] Saves a genome.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO {schema}.genomes (generation_id, genome_json, fitness) VALUES (%s, %s, %s)",
            (generation_id, json.dumps(genome_json), fitness)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[ru] Ошибка сохранения генома: {e}")
        print(f"[en] Error saving genome: {e}")
        return False


def load_best_genome(schema='agi_evolution'):
    """
    [ru] Загружает лучший геном из БД.
    [en] Loads the best genome from the database.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT g.genome_json, g.fitness, gen.generation_num
            FROM {schema}.genomes g
            JOIN {schema}.generations gen ON g.generation_id = gen.id
            ORDER BY g.fitness DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            genome_data = row[0]
            if isinstance(genome_data, dict):
                genome_json = genome_data
            else:
                genome_json = json.loads(genome_data)
            fitness = row[1]
            generation = row[2]
            print(f"[ru] Загружен лучший геном из поколения {generation} с фитнесом {fitness:.2f}")
            print(f"[en] Loaded best genome from generation {generation} with fitness {fitness:.2f}")
            return genome_json
        else:
            print("[ru] В БД нет сохранённых геномов.")
            print("[en] No saved genomes in the database.")
            return None
    except Exception as e:
        print(f"[ru] Ошибка загрузки лучшего генома: {e}")
        print(f"[en] Error loading best genome: {e}")
        return None


def save_generated_pattern(generation_id: int, pattern: np.ndarray,
                           score: float, rule: dict) -> bool:
    """
    [ru] Сохраняет сгенерированный паттерн в БД.
    [en] Saves a generated pattern to the database.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO agi_evolution.generated_patterns 
            (generation_id, pattern_vector, pattern_score, 
             sense_type, signal_type, signal_threshold, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            generation_id,
            json.dumps(pattern.tolist()),
            score,
            rule.get('sense_type', 'unknown'),
            rule.get('signal_type', 'unknown'),
            rule.get('threshold', 0.5),
            rule.get('action', 'unknown')
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[ru] ❌ Ошибка сохранения паттерна: {e}")
        print(f"[en] ❌ Error saving pattern: {e}")
        return False


def load_generated_patterns(limit: int = 100, schema='agi_evolution') -> List[Dict]:
    """
    [ru] Загружает сгенерированные паттерны из БД.
    [en] Loads generated patterns from the database.
    """
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(f"""
            SELECT id, generation_id, pattern_vector, pattern_score,
                   sense_type, signal_type, signal_threshold, action, created_at
            FROM {schema}.generated_patterns
            ORDER BY pattern_score DESC, created_at DESC
            LIMIT %s
        """, (limit,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        patterns = []
        for row in rows:
            patterns.append({
                'id': row['id'],
                'generation_id': row['generation_id'],
                'pattern': np.array(json.loads(row['pattern_vector'])),
                'score': row['pattern_score'],
                'sense_type': row['sense_type'],
                'signal_type': row['signal_type'],
                'threshold': row['signal_threshold'],
                'action': row['action'],
                'created_at': row['created_at']
            })

        return patterns
    except Exception as e:
        print(f"[ru] ❌ Ошибка загрузки паттернов: {e}")
        print(f"[en] ❌ Error loading patterns: {e}")
        return []



# # db/connector.py
# """
# Модуль для работы с БД.
# """
#
# import json
# import psycopg2
# import psycopg2.extras
# import numpy as np
# from typing import List, Dict, Optional, Tuple, Any, Union
#
#
# def get_connection(host='localhost', port=5432,
#                    dbname='postgres', user='postgres',
#                    password='postgres'):
#     """
#     Возвращает соединение с БД.
#     """
#     return psycopg2.connect(
#         host=host,
#         port=port,
#         dbname=dbname,
#         user=user,
#         password=password
#     )
#
#
# def load_reflex_rules(schema='agi_evolution'):
#     """
#     Загружает правила рефлексов из таблицы reflex_pattern.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#         cur.execute(f"SELECT sense_type, signal_type, signal_threshold, action FROM {schema}.reflex_pattern")
#         rows = cur.fetchall()
#         cur.close()
#         conn.close()
#         return [dict(row) for row in rows]
#     except Exception as e:
#         print(f"Ошибка загрузки рефлексов: {e}")
#         return []
#
#
# def load_instinct_patterns(schema='agi_evolution'):
#     """
#     Загружает паттерны инстинктов из таблицы instinct_pattern.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#         cur.execute(f"SELECT pattern, actions FROM {schema}.instinct_pattern")
#         rows = cur.fetchall()
#         cur.close()
#         conn.close()
#         patterns = []
#         for row in rows:
#             pattern = row['pattern'] if isinstance(row['pattern'], dict) else json.loads(row['pattern'])
#             action = row['actions'] if isinstance(row['actions'], dict) else json.loads(row['actions'])
#             patterns.append({'pattern': pattern, 'action': action})
#         return patterns
#     except Exception as e:
#         print(f"Ошибка загрузки инстинктов: {e}")
#         return []
#
#
# def save_generation(generation_num, best_fitness, avg_fitness, schema='agi_evolution'):
#     """
#     Сохраняет поколение и возвращает его id.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor()
#         cur.execute(
#             f"INSERT INTO {schema}.generations (generation_num, best_fitness, avg_fitness) VALUES (%s, %s, %s) RETURNING id",
#             (generation_num, best_fitness, avg_fitness)
#         )
#         gen_id = cur.fetchone()[0]
#         conn.commit()
#         cur.close()
#         conn.close()
#         return gen_id
#     except Exception as e:
#         print(f"Ошибка сохранения поколения: {e}")
#         return None
#
#
# def save_genome(generation_id, genome_json, fitness, schema='agi_evolution'):
#     """
#     Сохраняет геном.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor()
#         cur.execute(
#             f"INSERT INTO {schema}.genomes (generation_id, genome_json, fitness) VALUES (%s, %s, %s)",
#             (generation_id, json.dumps(genome_json), fitness)
#         )
#         conn.commit()
#         cur.close()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"Ошибка сохранения генома: {e}")
#         return False
#
#
# def load_best_genome(schema='agi_evolution'):
#     """
#     Загружает лучший геном из БД.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor()
#         cur.execute(f"""
#             SELECT g.genome_json, g.fitness, gen.generation_num
#             FROM {schema}.genomes g
#             JOIN {schema}.generations gen ON g.generation_id = gen.id
#             ORDER BY g.fitness DESC
#             LIMIT 1
#         """)
#         row = cur.fetchone()
#         cur.close()
#         conn.close()
#         if row:
#             genome_data = row[0]
#             if isinstance(genome_data, dict):
#                 genome_json = genome_data
#             else:
#                 genome_json = json.loads(genome_data)
#             fitness = row[1]
#             generation = row[2]
#             print(f"Загружен лучший геном из поколения {generation} с фитнесом {fitness:.2f}")
#             return genome_json
#         else:
#             print("В БД нет сохранённых геномов.")
#             return None
#     except Exception as e:
#         print(f"Ошибка загрузки лучшего генома: {e}")
#         return None
#
#
# def save_generated_pattern(generation_id: int, pattern: np.ndarray,
#                            score: float, rule: dict) -> bool:
#     """
#     Сохраняет сгенерированный паттерн в БД.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor()
#
#         cur.execute("""
#             INSERT INTO agi_evolution.generated_patterns
#             (generation_id, pattern_vector, pattern_score,
#              sense_type, signal_type, signal_threshold, action)
#             VALUES (%s, %s, %s, %s, %s, %s, %s)
#         """, (
#             generation_id,
#             json.dumps(pattern.tolist()),
#             score,
#             rule.get('sense_type', 'unknown'),
#             rule.get('signal_type', 'unknown'),
#             rule.get('threshold', 0.5),
#             rule.get('action', 'unknown')
#         ))
#
#         conn.commit()
#         cur.close()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"❌ Ошибка сохранения паттерна: {e}")
#         return False
#
#
# def load_generated_patterns(limit: int = 100, schema='agi_evolution') -> List[Dict]:
#     """
#     Загружает сгенерированные паттерны из БД.
#     """
#     try:
#         conn = get_connection()
#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#         cur.execute(f"""
#             SELECT id, generation_id, pattern_vector, pattern_score,
#                    sense_type, signal_type, signal_threshold, action, created_at
#             FROM {schema}.generated_patterns
#             ORDER BY pattern_score DESC, created_at DESC
#             LIMIT %s
#         """, (limit,))
#
#         rows = cur.fetchall()
#         cur.close()
#         conn.close()
#
#         patterns = []
#         for row in rows:
#             patterns.append({
#                 'id': row['id'],
#                 'generation_id': row['generation_id'],
#                 'pattern': np.array(json.loads(row['pattern_vector'])),
#                 'score': row['pattern_score'],
#                 'sense_type': row['sense_type'],
#                 'signal_type': row['signal_type'],
#                 'threshold': row['signal_threshold'],
#                 'action': row['action'],
#                 'created_at': row['created_at']
#             })
#
#         return patterns
#     except Exception as e:
#         print(f"❌ Ошибка загрузки паттернов: {e}")
#         return []
#
#
#
