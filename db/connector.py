# db/connector.py
import json
import psycopg2
import psycopg2.extras

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




