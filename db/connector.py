# db/connector.py
import json
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Tuple, Any, Union

def get_connection(host='localhost', port=5432, dbname='postgres', user='postgres', password='postgres'):
    """Возвращает соединение с БД."""
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )

def load_reflex_rules(schema='agi_evolution'):
    """Загружает правила рефлексов из таблицы reflex_pattern."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT sense_type, signal_type, signal_threshold, action FROM {schema}.reflex_pattern")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Ошибка загрузки рефлексов: {e}")
        return []

def load_instinct_patterns(schema='agi_evolution'):
    """Загружает паттерны инстинктов из таблицы instinct_pattern."""
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
        print(f"Ошибка загрузки инстинктов: {e}")
        return []

def save_generation(generation_num, best_fitness, avg_fitness, schema='agi_evolution'):
    """Сохраняет поколение и возвращает его id."""
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
        print(f"Ошибка сохранения поколения: {e}")
        return None

def save_genome(generation_id, genome_json, fitness, schema='agi_evolution'):
    """Сохраняет геном."""
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
        print(f"Ошибка сохранения генома: {e}")
        return False

def load_best_genome(schema='agi_evolution'):
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
            # Если это уже dict (например, при использовании JSONB), используем как есть
            if isinstance(genome_data, dict):
                genome_json = genome_data
            else:
                # Иначе предполагаем, что это строка JSON
                genome_json = json.loads(genome_data)
            fitness = row[1]
            generation = row[2]
            print(f"Загружен лучший геном из поколения {generation} с фитнесом {fitness:.2f}")
            return genome_json
        else:
            print("В БД нет сохранённых геномов.")
            return None
    except Exception as e:
        print(f"Ошибка загрузки лучшего генома: {e}")
        return None

def save_reflex_stats(reflex_id: int, success_count: int, total_count: int, success_rate: float):
    """Сохраняет статистику рефлекса."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE agi_evolution.reflex_pattern 
            SET success_count = %s, total_count = %s, success_rate = %s
            WHERE id = %s
        """, (success_count, total_count, success_rate, reflex_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения статистики рефлекса: {e}")
        return False

def save_gan_training_log(generation_id: int, epoch: int, g_loss: float, d_loss: float,
                         pattern_diversity: float, best_score: float, avg_score: float):
    """Сохраняет лог тренировки GAN."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO agi_evolution.gan_training_log 
            (generation_id, epoch, g_loss, d_loss, pattern_diversity, 
             best_pattern_score, avg_pattern_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (generation_id, epoch, g_loss, d_loss, pattern_diversity, best_score, avg_score))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения лога GAN: {e}")
        return False


