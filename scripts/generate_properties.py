import re
import json
import sys
import csv
from io import StringIO

# --- Список типов, для которых генерируем свойства (в нижнем регистре) ---
# Вы можете дополнить этот список после просмотра диагностики
ALLOWED_TYPES = {'part', 'component', 'device', 'mechanism', 'material', 'system', 'model'}

# --- Вспомогательные функции (без изменений) ---
def extract_year(text):
    if not text:
        return None
    match = re.search(r'\b(1[8-9]\d{2}|20\d{2})\b', text)
    return match.group(0) if match else None

def extract_country(text):
    countries = {
        'США': ['usa', 'united states', 'america', 'american'],
        'Россия': ['russia', 'russian', 'soviet'],
        'Германия': ['germany', 'german', 'prandtl'],
        'Франция': ['france', 'french', 'eiffel'],
        'Англия': ['england', 'english', 'britain', 'british'],
        'Италия': ['italy', 'italian'],
        'Китай': ['china', 'chinese'],
        'Япония': ['japan', 'japanese']
    }
    if not text:
        return None
    text_lower = text.lower()
    for country, variants in countries.items():
        for v in variants:
            if v in text_lower:
                return country
    return None

def extract_inventor(text):
    inventors = {
        'Братья Райт': ['wright brother', 'wrights', 'orville', 'wilbur'],
        'Джордж Кейли': ['cayley'],
        'Отто Лилиенталь': ['lilienthal'],
        'Гюстав Эйфель': ['eiffel'],
        'Людвиг Прандтль': ['prandtl'],
        'Николай Жуковский': ['zhukovsky', 'jukovsky'],
        'Хайрам Максим': ['maxim'],
        'Клемент Адер': ['ader'],
        'Сэмюэл Лэнгли': ['langley'],
        'Леонардо да Винчи': ['leonardo', 'da vinci']
    }
    if not text:
        return None
    text_lower = text.lower()
    for name, keywords in inventors.items():
        for kw in keywords:
            if kw in text_lower:
                return name
    return None

def extract_material(text):
    materials = ['wood', 'metal', 'steel', 'aluminium', 'aluminum', 'iron', 'copper', 'brass',
                 'plaster', 'paper', 'fabric', 'rubber', 'glass', 'plastic', 'composite']
    if not text:
        return None
    text_lower = text.lower()
    for mat in materials:
        if mat in text_lower:
            return mat.capitalize()
    return None

def extract_component_role(name, desc):
    if not name and not desc:
        return ''
    text = (name + ' ' + (desc or '')).lower()
    if 'крыло' in text or 'wing' in text:
        return 'создание подъёмной силы'
    if 'двигатель' in text or 'engine' in text or 'motor' in text:
        return 'создание тяги'
    if 'руль' in text or 'control' in text:
        return 'управление полётом'
    if 'фюзеляж' in text or 'fuselage' in text:
        return 'несущая конструкция'
    if 'элерон' in text or 'aileron' in text:
        return 'управление креном'
    if 'стабилизатор' in text or 'stabilizer' in text:
        return 'стабилизация'
    return ''

# --- Основная функция генерации свойств ---
def generate_properties(node_type, name, description, metadata):
    node_type_lower = node_type.lower()
    if node_type_lower not in ALLOWED_TYPES:
        return None

    desc = description or ''
    meta = metadata or {}
    props = {}

    if node_type_lower == 'part':
        props['название'] = name
        props['функция'] = extract_component_role(name, desc) or (desc[:100] if desc else '')
        mat = extract_material(desc)
        if mat:
            props['материал'] = mat

    elif node_type_lower == 'component':
        props['название'] = name
        props['роль'] = extract_component_role(name, desc) or (desc[:100] if desc else '')
        mat = extract_material(desc)
        if mat:
            props['материал'] = mat
        year = extract_year(desc)
        if year:
            props['год'] = year

    elif node_type_lower == 'device':
        props['название'] = name
        props['функция'] = desc[:150] if desc else ''
        inv = extract_inventor(desc)
        if inv:
            props['изобретатель'] = inv
        year = extract_year(desc)
        if year:
            props['год'] = year
        if 'glider' in name.lower() or 'планер' in name.lower():
            props['тип'] = 'планёр'
        elif 'balloon' in name.lower() or 'воздушный шар' in name.lower():
            props['тип'] = 'воздушный шар'
        elif 'airship' in name.lower() or 'дирижабль' in name.lower():
            props['тип'] = 'дирижабль'
        elif 'helicopter' in name.lower() or 'вертолёт' in name.lower():
            props['тип'] = 'вертолёт'
        elif 'rocket' in name.lower() or 'ракета' in name.lower():
            props['тип'] = 'ракета'
        else:
            props['тип'] = 'летательный аппарат'

    elif node_type_lower == 'mechanism':
        props['название'] = name
        props['принцип'] = desc[:150] if desc else ''
        inv = extract_inventor(desc)
        if inv:
            props['изобретатель'] = inv
        year = extract_year(desc)
        if year:
            props['год'] = year

    elif node_type_lower == 'material':
        props['название'] = name
        props['свойства'] = desc[:150] if desc else ''
        if 'air' in name.lower() or 'воздух' in name.lower():
            props['применение'] = 'атмосфера, аэродинамика'
        else:
            props['применение'] = 'авиастроение'

    elif node_type_lower == 'system':
        props['назначение'] = desc[:150] if desc else ''
        if 'frequency' in meta:
            props['частота'] = str(meta.get('frequency'))
        components = []
        if 'wing' in desc.lower() or 'крыло' in desc.lower():
            components.append('крыло')
        if 'engine' in desc.lower() or 'двигатель' in desc.lower():
            components.append('двигатель')
        if 'control' in desc.lower() or 'управление' in desc.lower():
            components.append('система управления')
        if components:
            props['компоненты'] = ', '.join(components)

    elif node_type_lower == 'model':
        props['тип'] = name
        props['область'] = 'аэродинамика'
        author = extract_inventor(desc)
        if author:
            props['автор'] = author
        year = extract_year(desc)
        if year:
            props['год'] = year
        props['описание'] = desc[:150] if desc else ''

    # Удаляем пустые значения
    props = {k: v for k, v in props.items() if v}
    return props if props else None

# --- Парсинг SQL-дампа (надёжный) ---
def parse_inserts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"INSERT INTO agi_evolution\.knowledge_nodes \(id,\"name\",node_type,properties,description,embedding,parameters,metadata,created_at,updated_at,theme_id\) VALUES\s*\((.*?)\);"
    matches = re.findall(pattern, content, re.DOTALL)
    nodes = []
    for match in matches:
        try:
            reader = csv.reader(StringIO(match), quotechar="'", skipinitialspace=True)
            parts = next(reader)
        except:
            # fallback
            parts = []
            current = ''
            in_quotes = False
            for ch in match:
                if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
                    in_quotes = not in_quotes
                if ch == ',' and not in_quotes:
                    parts.append(current.strip())
                    current = ''
                else:
                    current += ch
            parts.append(current.strip())
        if len(parts) < 11:
            continue
        def clean(s):
            s = s.strip()
            if s.startswith("'") and s.endswith("'"):
                s = s[1:-1]
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            if s == 'NULL':
                return None
            return s
        node = {
            'id': clean(parts[0]),
            'name': clean(parts[1]),
            'node_type': clean(parts[2]),
            'properties': clean(parts[3]),
            'description': clean(parts[4]),
            'embedding': clean(parts[5]),
            'parameters': clean(parts[6]),
            'metadata': clean(parts[7]),
            'created_at': clean(parts[8]),
            'updated_at': clean(parts[9]),
            'theme_id': clean(parts[10])
        }
        try:
            if node['metadata']:
                node['metadata'] = json.loads(node['metadata'])
            else:
                node['metadata'] = {}
        except:
            node['metadata'] = {}
        nodes.append(node)
    return nodes

def main():
    # Имя файла дампа можно передать аргументом
    dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
    nodes = parse_inserts(dump_file)

    # Диагностика: выводим все найденные типы
    type_counts = {}
    for node in nodes:
        nt = node['node_type']
        if nt is None:
            nt = 'NULL'
        type_counts[nt] = type_counts.get(nt, 0) + 1
    print("-- Types found in dump:", file=sys.stderr)
    for t, cnt in sorted(type_counts.items()):
        print(f"--   '{t}': {cnt}", file=sys.stderr)
    print("--", file=sys.stderr)

    print("-- Generated UPDATE statements for knowledge_nodes properties", file=sys.stderr)
    print("-- Only for node types: part, component, device, mechanism, material, system, model", file=sys.stderr)
    print("BEGIN;")
    count = 0
    for node in nodes:
        node_type = node['node_type']
        if not node_type:
            continue
        if node_type.lower() not in ALLOWED_TYPES:
            continue
        props = generate_properties(
            node_type,
            node['name'],
            node['description'],
            node['metadata']
        )
        if not props:
            continue
        json_str = json.dumps(props, ensure_ascii=False)
        # Экранируем кавычки для SQL (одиночные кавычки)
        json_str_escaped = json_str.replace("'", "''")
        # Формируем массив с одним элементом (строка JSON)
        # Используем синтаксис ARRAY['...']::text[]
        print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
        count += 1
    print("COMMIT;")
    print(f"-- Total {count} nodes updated.", file=sys.stderr)

if __name__ == '__main__':
    main()



# import re
# import json
# import sys
# import csv
# from io import StringIO
#
# # --- Список типов, для которых генерируем свойства (в нижнем регистре) ---
# # Вы можете дополнить этот список после просмотра диагностики
# ALLOWED_TYPES = {'part', 'component', 'device', 'mechanism', 'material', 'system', 'model'}
#
# # --- Вспомогательные функции (без изменений) ---
# def extract_year(text):
#     if not text:
#         return None
#     match = re.search(r'\b(1[8-9]\d{2}|20\d{2})\b', text)
#     return match.group(0) if match else None
#
# def extract_country(text):
#     countries = {
#         'США': ['usa', 'united states', 'america', 'american'],
#         'Россия': ['russia', 'russian', 'soviet'],
#         'Германия': ['germany', 'german', 'prandtl'],
#         'Франция': ['france', 'french', 'eiffel'],
#         'Англия': ['england', 'english', 'britain', 'british'],
#         'Италия': ['italy', 'italian'],
#         'Китай': ['china', 'chinese'],
#         'Япония': ['japan', 'japanese']
#     }
#     if not text:
#         return None
#     text_lower = text.lower()
#     for country, variants in countries.items():
#         for v in variants:
#             if v in text_lower:
#                 return country
#     return None
#
# def extract_inventor(text):
#     inventors = {
#         'Братья Райт': ['wright brother', 'wrights', 'orville', 'wilbur'],
#         'Джордж Кейли': ['cayley'],
#         'Отто Лилиенталь': ['lilienthal'],
#         'Гюстав Эйфель': ['eiffel'],
#         'Людвиг Прандтль': ['prandtl'],
#         'Николай Жуковский': ['zhukovsky', 'jukovsky'],
#         'Хайрам Максим': ['maxim'],
#         'Клемент Адер': ['ader'],
#         'Сэмюэл Лэнгли': ['langley'],
#         'Леонардо да Винчи': ['leonardo', 'da vinci']
#     }
#     if not text:
#         return None
#     text_lower = text.lower()
#     for name, keywords in inventors.items():
#         for kw in keywords:
#             if kw in text_lower:
#                 return name
#     return None
#
# def extract_material(text):
#     materials = ['wood', 'metal', 'steel', 'aluminium', 'aluminum', 'iron', 'copper', 'brass',
#                  'plaster', 'paper', 'fabric', 'rubber', 'glass', 'plastic', 'composite']
#     if not text:
#         return None
#     text_lower = text.lower()
#     for mat in materials:
#         if mat in text_lower:
#             return mat.capitalize()
#     return None
#
# def extract_component_role(name, desc):
#     if not name and not desc:
#         return ''
#     text = (name + ' ' + (desc or '')).lower()
#     if 'крыло' in text or 'wing' in text:
#         return 'создание подъёмной силы'
#     if 'двигатель' in text or 'engine' in text or 'motor' in text:
#         return 'создание тяги'
#     if 'руль' in text or 'control' in text:
#         return 'управление полётом'
#     if 'фюзеляж' in text or 'fuselage' in text:
#         return 'несущая конструкция'
#     if 'элерон' in text or 'aileron' in text:
#         return 'управление креном'
#     if 'стабилизатор' in text or 'stabilizer' in text:
#         return 'стабилизация'
#     return ''
#
# # --- Основная функция генерации свойств ---
# def generate_properties(node_type, name, description, metadata):
#     node_type_lower = node_type.lower()
#     if node_type_lower not in ALLOWED_TYPES:
#         return None
#
#     desc = description or ''
#     meta = metadata or {}
#     props = {}
#
#     if node_type_lower == 'part':
#         props['название'] = name
#         props['функция'] = extract_component_role(name, desc) or (desc[:100] if desc else '')
#         mat = extract_material(desc)
#         if mat:
#             props['материал'] = mat
#
#     elif node_type_lower == 'component':
#         props['название'] = name
#         props['роль'] = extract_component_role(name, desc) or (desc[:100] if desc else '')
#         mat = extract_material(desc)
#         if mat:
#             props['материал'] = mat
#         year = extract_year(desc)
#         if year:
#             props['год'] = year
#
#     elif node_type_lower == 'device':
#         props['название'] = name
#         props['функция'] = desc[:150] if desc else ''
#         inv = extract_inventor(desc)
#         if inv:
#             props['изобретатель'] = inv
#         year = extract_year(desc)
#         if year:
#             props['год'] = year
#         if 'glider' in name.lower() or 'планер' in name.lower():
#             props['тип'] = 'планёр'
#         elif 'balloon' in name.lower() or 'воздушный шар' in name.lower():
#             props['тип'] = 'воздушный шар'
#         elif 'airship' in name.lower() or 'дирижабль' in name.lower():
#             props['тип'] = 'дирижабль'
#         elif 'helicopter' in name.lower() or 'вертолёт' in name.lower():
#             props['тип'] = 'вертолёт'
#         elif 'rocket' in name.lower() or 'ракета' in name.lower():
#             props['тип'] = 'ракета'
#         else:
#             props['тип'] = 'летательный аппарат'
#
#     elif node_type_lower == 'mechanism':
#         props['название'] = name
#         props['принцип'] = desc[:150] if desc else ''
#         inv = extract_inventor(desc)
#         if inv:
#             props['изобретатель'] = inv
#         year = extract_year(desc)
#         if year:
#             props['год'] = year
#
#     elif node_type_lower == 'material':
#         props['название'] = name
#         props['свойства'] = desc[:150] if desc else ''
#         if 'air' in name.lower() or 'воздух' in name.lower():
#             props['применение'] = 'атмосфера, аэродинамика'
#         else:
#             props['применение'] = 'авиастроение'
#
#     elif node_type_lower == 'system':
#         props['назначение'] = desc[:150] if desc else ''
#         if 'frequency' in meta:
#             props['частота'] = str(meta.get('frequency'))
#         components = []
#         if 'wing' in desc.lower() or 'крыло' in desc.lower():
#             components.append('крыло')
#         if 'engine' in desc.lower() or 'двигатель' in desc.lower():
#             components.append('двигатель')
#         if 'control' in desc.lower() or 'управление' in desc.lower():
#             components.append('система управления')
#         if components:
#             props['компоненты'] = ', '.join(components)
#
#     elif node_type_lower == 'model':
#         props['тип'] = name
#         props['область'] = 'аэродинамика'
#         author = extract_inventor(desc)
#         if author:
#             props['автор'] = author
#         year = extract_year(desc)
#         if year:
#             props['год'] = year
#         props['описание'] = desc[:150] if desc else ''
#
#     props = {k: v for k, v in props.items() if v}
#     return props if props else None
#
# # --- Парсинг SQL-дампа с использованием csv для надёжности ---
# def parse_inserts(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
#     # Ищем все VALUES
#     pattern = r"INSERT INTO agi_evolution\.knowledge_nodes \(id,\"name\",node_type,properties,description,embedding,parameters,metadata,created_at,updated_at,theme_id\) VALUES\s*\((.*?)\);"
#     matches = re.findall(pattern, content, re.DOTALL)
#     nodes = []
#     for match in matches:
#         # Используем csv для разбора полей, учитывая кавычки
#         try:
#             reader = csv.reader(StringIO(match), quotechar="'", skipinitialspace=True)
#             parts = next(reader)
#         except:
#             # fallback: простой разбор
#             parts = []
#             current = ''
#             in_quotes = False
#             for ch in match:
#                 if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
#                     in_quotes = not in_quotes
#                 if ch == ',' and not in_quotes:
#                     parts.append(current.strip())
#                     current = ''
#                 else:
#                     current += ch
#             parts.append(current.strip())
#         if len(parts) < 11:
#             continue
#         def clean(s):
#             s = s.strip()
#             if s.startswith("'") and s.endswith("'"):
#                 s = s[1:-1]
#             if s.startswith('"') and s.endswith('"'):
#                 s = s[1:-1]
#             if s == 'NULL':
#                 return None
#             return s
#         node = {
#             'id': clean(parts[0]),
#             'name': clean(parts[1]),
#             'node_type': clean(parts[2]),
#             'properties': clean(parts[3]),
#             'description': clean(parts[4]),
#             'embedding': clean(parts[5]),
#             'parameters': clean(parts[6]),
#             'metadata': clean(parts[7]),
#             'created_at': clean(parts[8]),
#             'updated_at': clean(parts[9]),
#             'theme_id': clean(parts[10])
#         }
#         try:
#             if node['metadata']:
#                 node['metadata'] = json.loads(node['metadata'])
#             else:
#                 node['metadata'] = {}
#         except:
#             node['metadata'] = {}
#         nodes.append(node)
#     return nodes
#
# def main():
#     nodes = parse_inserts('dump.sql')
#     # --- Диагностика: выводим все найденные типы ---
#     type_counts = {}
#     for node in nodes:
#         nt = node['node_type']
#         if nt is None:
#             nt = 'NULL'
#         type_counts[nt] = type_counts.get(nt, 0) + 1
#     print("-- Types found in dump:", file=sys.stderr)
#     for t, cnt in sorted(type_counts.items()):
#         print(f"--   '{t}': {cnt}", file=sys.stderr)
#     print("--", file=sys.stderr)
#
#     print("-- Generated UPDATE statements for knowledge_nodes properties", file=sys.stderr)
#     print("-- Only for node types: part, component, device, mechanism, material, system, model", file=sys.stderr)
#     print("BEGIN;")
#     count = 0
#     for node in nodes:
#         node_type = node['node_type']
#         if not node_type:
#             continue
#         # Проверяем, соответствует ли тип (без учёта регистра)
#         if node_type.lower() not in ALLOWED_TYPES:
#             continue
#         props = generate_properties(
#             node_type,
#             node['name'],
#             node['description'],
#             node['metadata']
#         )
#         if not props:
#             continue
#         json_str = json.dumps(props, ensure_ascii=False)
#         json_str_escaped = json_str.replace("'", "''")
#         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = '{json_str_escaped}'::jsonb WHERE id = '{node['id']}';")
#         count += 1
#     print("COMMIT;")
#     print(f"-- Total {count} nodes updated.", file=sys.stderr)
#
# if __name__ == '__main__':
#     main()