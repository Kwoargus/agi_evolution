import json
import uuid
import time
import re
import psycopg2
import ollama

# Список отсутствующих узлов (вставьте свой список)
MISSING_NODES = [
    "антенна",
    "арифметико-логическое устройство",
    "аэродинамические рули",
    "багор",
    "база",
    "бак с хладоагентом",
    "блок развёртки",
    "блок управления",
    "болт",
    "большое гиперболическое зеркало",
    "вал привода силового агрегата",
    "вал с зубчатыми колёсами",
    "ведро",
    "весло",
    "винт фокусировки",
    "внешний обод",
    "внутренний обод",
    "вторичный вал",
    "втулка",
    "входной вал",
    "выключатель",
    "выходной вал",
    "выходной контакт",
    "гвоздь",
    "генератор тока",
    "горшок",
    "граммофон",
    "демодулятор",
    "жернов",
    "забор",
    "заземление",
    "затвор",
    "звуковая бороздка",
    "звуковой динамик",
    "зубчатое колесо",
    "игла",
    "индикатор",
    "испаритель",
    "исток",
    "источник света",
    "источник тока",
    "казан",
    "камера сгорания",
    "карбюратор",
    "кастрюля",
    "катапульта",
    "катушка из медной проволоки",
    "катушка индуктивности",
    "клубок нити",
    "колба",
    "колебательный контур",
    "коленвал",
    "колодец",
    "компрессор",
    "конденсатор",
    "контакты",
    "костёр",
    "котёл",
    "кочерга",
    "крыльчатка",
    "кружка",
    "лампа",
    "линза",
    "лодка",
    "лопасть",
    "лопатка",
    "лыжи",
    "малое зеркало",
    "механическая нагрузка",
    "модулятор",
    "нить накаливания",
    "объектив",
    "обод",
    "оперативная память",
    "отражатель",
    "очки",
    "паровой двигатель",
    "патронник",
    "педаль",
    "первичный вал",
    "печь",
    "пластинка с записью",
    "подающее устройство",
    "подзорная труба",
    "подшипник",
    "полупроводник",
    "поршень",
    "преобразоратель тока",
    "приёмник сигнала",
    "приклад",
    "принимающая антенна",
    "принимающее устройство",
    "призма",
    "прицел",
    "провод",
    "проводник дырок",
    "проводник электронов",
    "пропеллер",
    "пушка",
    "рама",
    "рамка с опорным шпагатом",
    "рамка с током",
    "регистр AX",
    "регистр BX",
    "редуктор",
    "резистор",
    "резьбовое соединение",
    "рельс",
    "решатель",
    "розетка",
    "ротор",
    "руль",
    "рупор",
    "ручка вращения вала",
    "ручной фонарь",
    "рычаг",
    "рычаг взвода",
    "самовар",
    "светочувствительный материал",
    "свеча",
    "сепаратор",
    "скамья",
    "сопло",
    "сопло Лаваля",
    "спусковой крючёк",
    "статор",
    "ствол",
    "сток",
    "стол",
    "строение",
    "сферическое зеркало",
    "таз",
    "тёмная камера",
    "тетива",
    "топка",
    "топливный бак",
    "трансформатор",
    "триод",
    "трубка",
    "трубопровод",
    "турбина",
    "увеличительное стекло",
    "уключина",
    "упругая боевая дуга",
    "усилитель",
    "факел",
    "фильр",
    "форсунка",
    "центрифуга",
    "цепь",
    "цилиндр",
    "цоколь",
    "чайник",
    "челнок",
    "шарик",
    "шатун",
    "шестерня",
    "шкаф",
    "электродвигатель",
    "электронно-лучевая трубка",
    "ядро",
    "ящик"
]

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def generate_node_data(name):
    """Генерирует описание и свойства для узла через LLM."""
    prompt = f"""
Ты — эксперт по техническим системам и инженерии. 
Для концептуального узла «{name}» сгенерируй:
- краткое описание (1–2 предложения),
- список свойств (ключевые характеристики, 3–5 пунктов).

Ответ должен быть в формате JSON:
{{"description": "...", "properties": ["свойство1", "свойство2", ...]}}
"""
    response = ollama.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.3}
    )
    content = response['message']['content'].strip()
    # Извлекаем JSON
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    # fallback
    return {
        "description": f"Концептуальный узел, связанный с {name}",
        "properties": ["технический компонент"]
    }

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    name_to_id = {}

    # Проверяем, какие узлы уже есть
    placeholders = ','.join(['%s'] * len(MISSING_NODES))
    cur.execute(f"SELECT id, name FROM agi_evolution.knowledge_nodes WHERE name IN ({placeholders})", MISSING_NODES)
    existing = {row[1]: row[0] for row in cur.fetchall()}
    existing_names = set(existing.keys())

    # Оставляем только те, которых нет в БД
    to_add = [n for n in MISSING_NODES if n not in existing_names]
    print(f"Всего отсутствует: {len(to_add)}")

    if not to_add:
        print("Все узлы уже существуют.")
        return

    insert_statements = []
    for name in to_add:
        node_id = str(uuid.uuid4())
        data = generate_node_data(name)
        description = data.get('description', f'Узел: {name}')
        properties = data.get('properties', [])
        properties_json = json.dumps(properties, ensure_ascii=False)

        insert_sql = f"""
INSERT INTO agi_evolution.knowledge_nodes (id, name, node_type, properties, description, embedding, parameters, metadata, created_at, updated_at, theme_id)
VALUES (
    '{node_id}',
    '{name.replace("'", "''")}',
    'CONCEPT',
    '{{"properties": {properties_json}}}'::jsonb,
    '{description.replace("'", "''")}',
    NULL,
    '{{}}'::jsonb,
    '{{}}'::jsonb,
    NOW(),
    NOW(),
    NULL
);
"""
        insert_statements.append(insert_sql)
        name_to_id[name] = node_id
        print(f"Сгенерирован узел: {name} -> {node_id}")
        time.sleep(0.5)  # пауза, чтобы не перегружать LLM

    # Сохраняем INSERT-запросы в файл
    with open("insert_missing_nodes.sql", "w", encoding="utf-8") as f:
        f.write("BEGIN;\n")
        f.write("\n".join(insert_statements))
        f.write("\nCOMMIT;\n")

    # Также сохраняем словарь соответствий
    with open("name_to_id.json", "w", encoding="utf-8") as f:
        json.dump(name_to_id, f, ensure_ascii=False, indent=2)

    print(f"✅ Сгенерировано {len(insert_statements)} INSERT-запросов.")
    print("✅ Файл insert_missing_nodes.sql готов для выполнения.")
    print("✅ Словарь name→id сохранён в name_to_id.json.")

    conn.close()

if __name__ == "__main__":
    main()