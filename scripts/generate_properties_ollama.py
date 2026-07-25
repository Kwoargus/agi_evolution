import re
import json
import sys
import csv
from io import StringIO
import time

ALLOWED_TYPES = {
    'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
    'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
    'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
}

def parse_inserts(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    nodes = []
    pos = 0
    found_inserts = 0
    while True:
        start = content.find("INSERT INTO agi_evolution.knowledge_nodes", pos)
        if start == -1:
            break
        found_inserts += 1
        # Ищем VALUES (
        values_pos = content.find("VALUES (", start)
        if values_pos == -1:
            print(f"INSERT {found_inserts}: не найдено VALUES (", file=sys.stderr)
            pos = start + 1
            continue
        print(f"INSERT {found_inserts}: VALUES ( на позиции {values_pos}", file=sys.stderr)
        i = values_pos + len("VALUES (")
        paren_level = 1
        in_single_quote = False
        in_double_quote = False
        escape = False
        while i < len(content):
            ch = content[i]
            if escape:
                escape = False
                i += 1
                continue
            if ch == '\\':
                escape = True
                i += 1
                continue
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if ch == '(':
                    paren_level += 1
                elif ch == ')':
                    paren_level -= 1
                    if paren_level == 0:
                        end = i
                        break
            i += 1
        else:
            print(f"INSERT {found_inserts}: не найдена закрывающая скобка", file=sys.stderr)
            pos = i + 1
            continue

        values_str = content[values_pos + len("VALUES ("):end]
        print(f"INSERT {found_inserts}: values_str длина {len(values_str)}, начало: {repr(values_str[:100])}", file=sys.stderr)
        tuples = split_tuples(values_str)
        print(f"INSERT {found_inserts}: извлечено {len(tuples)} кортежей", file=sys.stderr)
        for tup in tuples:
            node = parse_tuple(tup)
            if node:
                nodes.append(node)
        pos = end + 1

    print(f"Всего найдено INSERT-запросов: {found_inserts}", file=sys.stderr)
    return nodes

def split_tuples(values_str):
    tuples = []
    current = ''
    in_single_quote = False
    in_double_quote = False
    paren_level = 0
    i = 0
    while i < len(values_str):
        ch = values_str[i]
        if ch == "'" and (i == 0 or values_str[i-1] != '\\'):
            in_single_quote = not in_single_quote
            current += ch
            i += 1
            continue
        if ch == '"' and (i == 0 or values_str[i-1] != '\\'):
            in_double_quote = not in_double_quote
            current += ch
            i += 1
            continue
        if not in_single_quote and not in_double_quote:
            if ch == '(':
                paren_level += 1
            elif ch == ')':
                paren_level -= 1
            if ch == ',' and paren_level == 0:
                # разделитель между кортежами
                if current.strip():
                    tuples.append(current.strip())
                current = ''
                i += 1
                continue
        current += ch
        i += 1
    if current.strip():
        tuples.append(current.strip())
    return tuples

# def split_tuples(values_str):
#     tuples = []
#     current = ''
#     in_single_quote = False
#     in_double_quote = False
#     escape = False
#     paren_level = 0
#     for ch in values_str:
#         if escape:
#             escape = False
#             current += ch
#             continue
#         if ch == '\\':
#             escape = True
#             current += ch
#             continue
#         if ch == "'" and not in_double_quote:
#             in_single_quote = not in_single_quote
#             current += ch
#             continue
#         if ch == '"' and not in_single_quote:
#             in_double_quote = not in_double_quote
#             current += ch
#             continue
#         if not in_single_quote and not in_double_quote:
#             if ch == '(':
#                 paren_level += 1
#                 current += ch
#                 continue
#             elif ch == ')':
#                 paren_level -= 1
#                 current += ch
#                 if paren_level == 0:
#                     tuples.append(current.strip())
#                     current = ''
#                 continue
#             elif ch == ',' and paren_level == 0:
#                 if current.strip():
#                     tuples.append(current.strip())
#                     current = ''
#                 continue
#         current += ch
#     if current.strip():
#         tuples.append(current.strip())
#     return tuples

def parse_tuple(tup_str):
    tup_str = tup_str.strip()
    if tup_str.startswith('(') and tup_str.endswith(')'):
        tup_str = tup_str[1:-1]
    try:
        reader = csv.reader(StringIO(tup_str), quotechar="'", skipinitialspace=True)
        parts = next(reader)
    except:
        parts = []
        current = ''
        in_quotes = False
        for ch in tup_str:
            if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
                in_quotes = not in_quotes
            if ch == ',' and not in_quotes:
                parts.append(current.strip())
                current = ''
            else:
                current += ch
        parts.append(current.strip())
    if len(parts) < 11:
        return None

    def clean(s):
        s = s.strip()
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        if s == 'NULL':
            return None
        if '::' in s:
            s = s.split('::')[0]
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
        node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
    except:
        node['metadata'] = {}
    return node

# Заглушка для LLM
def generate_properties_with_llm(name, node_type, description):
    return {"название": name, "описание": description[:100] if description else ""}

def main():
    dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
    nodes = parse_inserts(dump_file)
    print(f"Всего распарсено узлов: {len(nodes)}", file=sys.stderr)

    print("BEGIN;")
    updated = 0
    skipped = 0
    errors = 0

    for node in nodes:
        nt = node['node_type']
        if not nt:
            skipped += 1
            continue
        if nt.lower() not in ALLOWED_TYPES:
            skipped += 1
            continue
        if not node['description']:
            skipped += 1
            continue

        props = None
        for attempt in range(3):
            try:
                props = generate_properties_with_llm(node['name'], nt, node['description'])
                if props:
                    break
            except Exception as e:
                print(f"Error for {node['name']} (attempt {attempt+1}): {e}", file=sys.stderr)
                time.sleep(1)

        if not props:
            errors += 1
            continue

        json_str = json.dumps(props, ensure_ascii=False)
        json_str_escaped = json_str.replace("'", "''")
        print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
        updated += 1

    print("COMMIT;")
    print(f"-- Updated: {updated}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)

if __name__ == '__main__':
    main()









# import re
# import json
# import sys
# import csv
# from io import StringIO
# import time
#
# ALLOWED_TYPES = {
#     'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
#     'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
#     'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
# }
#
# def parse_inserts(filepath):
#
#     print(f"DEBUG: Начало парсинга, файл {filepath}")
#     with open(filepath, 'r', encoding='utf-8-sig') as f:
#         content = f.read()
#     print(f"DEBUG: Прочитано {len(content)} символов")
#     print(f"DEBUG: Первые 200 символов: {repr(content[:200])}")
#     print(f"DEBUG: Поиск 'INSERT INTO ...'...")
#     start = content.find("INSERT INTO agi_evolution.knowledge_nodes")
#     print(f"DEBUG: Позиция первого INSERT: {start}")
#
#     with open(filepath, 'r', encoding='utf-8-sig') as f:
#         content = f.read()
#
#     nodes = []
#     pos = 0
#     while True:
#         # Ищем начало INSERT
#         start = content.find("INSERT INTO agi_evolution.knowledge_nodes", pos)
#         if start == -1:
#             break
#         # Ищем VALUES (
#         values_pos = content.find("VALUES (", start)
#         if values_pos == -1:
#             break
#         # Начинаем с позиции после "VALUES ("
#         i = values_pos + len("VALUES (")
#         paren_level = 1
#         in_single_quote = False
#         in_double_quote = False
#         escape = False
#         while i < len(content):
#             ch = content[i]
#             if escape:
#                 escape = False
#                 i += 1
#                 continue
#             if ch == '\\':
#                 escape = True
#                 i += 1
#                 continue
#             if ch == "'" and not in_double_quote:
#                 in_single_quote = not in_single_quote
#             elif ch == '"' and not in_single_quote:
#                 in_double_quote = not in_double_quote
#             elif not in_single_quote and not in_double_quote:
#                 if ch == '(':
#                     paren_level += 1
#                 elif ch == ')':
#                     paren_level -= 1
#                     if paren_level == 0:
#                         end = i
#                         break
#             i += 1
#         else:
#             # Не нашли закрывающую скобку
#             pos = i + 1
#             continue
#
#         values_str = content[values_pos + len("VALUES ("):end]
#         tuples = split_tuples(values_str)
#         for tup in tuples:
#             node = parse_tuple(tup)
#             if node:
#                 nodes.append(node)
#
#         pos = end + 1
#
#     return nodes
#
# def split_tuples(values_str):
#     tuples = []
#     current = ''
#     in_single_quote = False
#     in_double_quote = False
#     escape = False
#     paren_level = 0
#     for ch in values_str:
#         if escape:
#             escape = False
#             current += ch
#             continue
#         if ch == '\\':
#             escape = True
#             current += ch
#             continue
#         if ch == "'" and not in_double_quote:
#             in_single_quote = not in_single_quote
#             current += ch
#             continue
#         if ch == '"' and not in_single_quote:
#             in_double_quote = not in_double_quote
#             current += ch
#             continue
#         if not in_single_quote and not in_double_quote:
#             if ch == '(':
#                 paren_level += 1
#                 current += ch
#                 continue
#             elif ch == ')':
#                 paren_level -= 1
#                 current += ch
#                 if paren_level == 0:
#                     tuples.append(current.strip())
#                     current = ''
#                 continue
#             elif ch == ',' and paren_level == 0:
#                 if current.strip():
#                     tuples.append(current.strip())
#                     current = ''
#                 continue
#         current += ch
#     if current.strip():
#         tuples.append(current.strip())
#     return tuples
#
# def parse_tuple(tup_str):
#     tup_str = tup_str.strip()
#     if tup_str.startswith('(') and tup_str.endswith(')'):
#         tup_str = tup_str[1:-1]
#     try:
#         reader = csv.reader(StringIO(tup_str), quotechar="'", skipinitialspace=True)
#         parts = next(reader)
#     except:
#         parts = []
#         current = ''
#         in_quotes = False
#         for ch in tup_str:
#             if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
#                 in_quotes = not in_quotes
#             if ch == ',' and not in_quotes:
#                 parts.append(current.strip())
#                 current = ''
#             else:
#                 current += ch
#         parts.append(current.strip())
#     if len(parts) < 11:
#         return None
#
#     def clean(s):
#         s = s.strip()
#         if s.startswith("'") and s.endswith("'"):
#             s = s[1:-1]
#         if s.startswith('"') and s.endswith('"'):
#             s = s[1:-1]
#         if s == 'NULL':
#             return None
#         # Убираем ::uuid и подобное
#         if '::' in s:
#             s = s.split('::')[0]
#         return s
#
#     node = {
#         'id': clean(parts[0]),
#         'name': clean(parts[1]),
#         'node_type': clean(parts[2]),
#         'properties': clean(parts[3]),
#         'description': clean(parts[4]),
#         'embedding': clean(parts[5]),
#         'parameters': clean(parts[6]),
#         'metadata': clean(parts[7]),
#         'created_at': clean(parts[8]),
#         'updated_at': clean(parts[9]),
#         'theme_id': clean(parts[10])
#     }
#     try:
#         node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
#     except:
#         node['metadata'] = {}
#     return node
#
# # --- Функция генерации свойств (замените на вашу) ---
# def generate_properties_with_llm(name, node_type, description):
#     # Заглушка
#     return {"название": name, "описание": description[:100] if description else ""}
#
# def main():
#     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
#     nodes = parse_inserts(dump_file)
#     print(f"Всего распарсено узлов: {len(nodes)}", file=sys.stderr)
#
#     print("BEGIN;")
#     updated = 0
#     skipped = 0
#     errors = 0
#
#     for node in nodes:
#         nt = node['node_type']
#         if not nt:
#             skipped += 1
#             continue
#         if nt.lower() not in ALLOWED_TYPES:
#             skipped += 1
#             continue
#         if not node['description']:
#             skipped += 1
#             continue
#
#         props = None
#         for attempt in range(3):
#             try:
#                 props = generate_properties_with_llm(node['name'], nt, node['description'])
#                 if props:
#                     break
#             except Exception as e:
#                 print(f"Error for {node['name']} (attempt {attempt+1}): {e}", file=sys.stderr)
#                 time.sleep(1)
#
#         if not props:
#             errors += 1
#             continue
#
#         json_str = json.dumps(props, ensure_ascii=False)
#         json_str_escaped = json_str.replace("'", "''")
#         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
#         updated += 1
#
#     print("COMMIT;")
#     print(f"-- Updated: {updated}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)
#
# if __name__ == '__main__':
#     main()
#
#
#
#
#
#
# # import re
# # import json
# # import sys
# # import csv
# # from io import StringIO
# # import time
# #
# # ALLOWED_TYPES = {
# #     'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
# #     'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
# #     'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
# # }
# #
# # def parse_inserts(filepath):
# #     with open(filepath, 'r', encoding='utf-8') as f:
# #         content = f.read()
# #
# #     # Ищем все INSERT ... VALUES
# #     # Регулярное выражение для поиска начала INSERT
# #     pattern = re.compile(r"INSERT\s+INTO\s+agi_evolution\.knowledge_nodes\s*\([^)]*\)\s+VALUES\s*\(", re.IGNORECASE | re.DOTALL)
# #
# #     nodes = []
# #     pos = 0
# #     while True:
# #         match = pattern.search(content, pos)
# #         if not match:
# #             break
# #         start = match.end()  # позиция после '('
# #         # Теперь идём по символам, балансируя скобки
# #         i = start
# #         paren_level = 1
# #         in_single_quote = False
# #         in_double_quote = False
# #         escape = False
# #         while i < len(content):
# #             ch = content[i]
# #             if escape:
# #                 escape = False
# #                 i += 1
# #                 continue
# #             if ch == '\\':
# #                 escape = True
# #                 i += 1
# #                 continue
# #             if ch == "'" and not in_double_quote:
# #                 in_single_quote = not in_single_quote
# #             elif ch == '"' and not in_single_quote:
# #                 in_double_quote = not in_double_quote
# #             elif not in_single_quote and not in_double_quote:
# #                 if ch == '(':
# #                     paren_level += 1
# #                 elif ch == ')':
# #                     paren_level -= 1
# #                     if paren_level == 0:
# #                         # Конец VALUES
# #                         end = i
# #                         break
# #             i += 1
# #         else:
# #             # Не нашли закрывающую скобку
# #             break
# #
# #         # Извлекаем строку VALUES без внешних скобок
# #         values_str = content[start:end]
# #         # Разбиваем на кортежи
# #         tuples = split_tuples(values_str)
# #         for tup in tuples:
# #             node = parse_tuple(tup)
# #             if node:
# #                 nodes.append(node)
# #
# #         pos = end + 1  # продолжаем после ')'
# #
# #     return nodes
# #
# #
# # def split_tuples(values_str):
# #     # Разбиваем строку на кортежи: разделитель ),(
# #     tuples = []
# #     current = ''
# #     in_single_quote = False
# #     in_double_quote = False
# #     escape = False
# #     paren_level = 0
# #     for ch in values_str:
# #         if escape:
# #             escape = False
# #             current += ch
# #             continue
# #         if ch == '\\':
# #             escape = True
# #             current += ch
# #             continue
# #         if ch == "'" and not in_double_quote:
# #             in_single_quote = not in_single_quote
# #             current += ch
# #             continue
# #         if ch == '"' and not in_single_quote:
# #             in_double_quote = not in_double_quote
# #             current += ch
# #             continue
# #         if not in_single_quote and not in_double_quote:
# #             if ch == '(':
# #                 paren_level += 1
# #                 current += ch
# #                 continue
# #             elif ch == ')':
# #                 paren_level -= 1
# #                 current += ch
# #                 if paren_level == 0:
# #                     # Конец кортежа
# #                     tuples.append(current.strip())
# #                     current = ''
# #                 continue
# #             elif ch == ',' and paren_level == 0:
# #                 # Разделитель между кортежами (на верхнем уровне)
# #                 # current уже содержит предыдущий кортеж
# #                 tuples.append(current.strip())
# #                 current = ''
# #                 continue
# #         current += ch
# #     if current.strip():
# #         tuples.append(current.strip())
# #     return tuples
# #
# #
# # def parse_tuple(tup_str):
# #     # Убираем внешние скобки, если есть
# #     tup_str = tup_str.strip()
# #     if tup_str.startswith('(') and tup_str.endswith(')'):
# #         tup_str = tup_str[1:-1]
# #     try:
# #         reader = csv.reader(StringIO(tup_str), quotechar="'", skipinitialspace=True)
# #         parts = next(reader)
# #     except:
# #         # fallback
# #         parts = []
# #         current = ''
# #         in_quotes = False
# #         for ch in tup_str:
# #             if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# #                 in_quotes = not in_quotes
# #             if ch == ',' and not in_quotes:
# #                 parts.append(current.strip())
# #                 current = ''
# #             else:
# #                 current += ch
# #         parts.append(current.strip())
# #     if len(parts) < 11:
# #         return None
# #
# #     def clean(s):
# #         s = s.strip()
# #         if s.startswith("'") and s.endswith("'"):
# #             s = s[1:-1]
# #         if s.startswith('"') and s.endswith('"'):
# #             s = s[1:-1]
# #         if s == 'NULL':
# #             return None
# #         return s
# #
# #     node = {
# #         'id': clean(parts[0]),
# #         'name': clean(parts[1]),
# #         'node_type': clean(parts[2]),
# #         'properties': clean(parts[3]),
# #         'description': clean(parts[4]),
# #         'embedding': clean(parts[5]),
# #         'parameters': clean(parts[6]),
# #         'metadata': clean(parts[7]),
# #         'created_at': clean(parts[8]),
# #         'updated_at': clean(parts[9]),
# #         'theme_id': clean(parts[10])
# #     }
# #     try:
# #         node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
# #     except:
# #         node['metadata'] = {}
# #     return node
# #
# # # --- Функция генерации свойств (здесь ваша реализация с LLM) ---
# # def generate_properties_with_llm(name, node_type, description):
# #     # Замените на реальный вызов вашей локальной модели
# #     # Пока заглушка
# #     return {"название": name, "описание": description[:100] if description else ""}
# #
# # def main():
# #     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
# #     nodes = parse_inserts(dump_file)
# #     print(f"Всего распарсено узлов: {len(nodes)}", file=sys.stderr)
# #
# #     print("BEGIN;")
# #     updated = 0
# #     skipped = 0
# #     errors = 0
# #
# #     for node in nodes:
# #         nt = node['node_type']
# #         if not nt:
# #             skipped += 1
# #             continue
# #         if nt.lower() not in ALLOWED_TYPES:
# #             skipped += 1
# #             continue
# #         if not node['description']:
# #             skipped += 1
# #             continue
# #
# #         props = None
# #         for attempt in range(3):
# #             try:
# #                 props = generate_properties_with_llm(node['name'], nt, node['description'])
# #                 if props:
# #                     break
# #             except Exception as e:
# #                 print(f"Error for {node['name']} (attempt {attempt+1}): {e}", file=sys.stderr)
# #                 time.sleep(1)
# #
# #         if not props:
# #             errors += 1
# #             continue
# #
# #         json_str = json.dumps(props, ensure_ascii=False)
# #         json_str_escaped = json_str.replace("'", "''")
# #         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
# #         updated += 1
# #
# #     print("COMMIT;")
# #     print(f"-- Updated: {updated}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)
# #
# # if __name__ == '__main__':
# #     main()
# #
# #
# #
# # # import re
# # # import json
# # # import sys
# # # import csv
# # # from io import StringIO
# # # from collections import defaultdict
# # #
# # # ALLOWED_TYPES = {
# # #     'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
# # #     'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
# # #     'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
# # # }
# # #
# # # def parse_all_nodes(filepath):
# # #     with open(filepath, 'r', encoding='utf-8') as f:
# # #         content = f.read()
# # #
# # #     blocks = re.split(r';\s*', content)
# # #     nodes = []
# # #     for block in blocks:
# # #         if not block.strip().startswith('INSERT'):
# # #             continue
# # #         values_match = re.search(r'VALUES\s*\((.*)\)\s*$', block, re.DOTALL | re.IGNORECASE)
# # #         if not values_match:
# # #             continue
# # #         values_str = values_match.group(1)
# # #         tuples = []
# # #         current = ''
# # #         in_quotes = False
# # #         paren_level = 0
# # #         for ch in values_str:
# # #             if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # #                 in_quotes = not in_quotes
# # #             if not in_quotes:
# # #                 if ch == '(':
# # #                     paren_level += 1
# # #                 elif ch == ')':
# # #                     paren_level -= 1
# # #             if not in_quotes and ch == ',' and paren_level == 1:
# # #                 tuples.append(current.strip())
# # #                 current = ''
# # #                 continue
# # #             current += ch
# # #         if current.strip():
# # #             tuples.append(current.strip())
# # #
# # #         for tup in tuples:
# # #             if tup.startswith('(') and tup.endswith(')'):
# # #                 tup = tup[1:-1]
# # #             try:
# # #                 reader = csv.reader(StringIO(tup), quotechar="'", skipinitialspace=True)
# # #                 parts = next(reader)
# # #             except:
# # #                 parts = []
# # #                 current = ''
# # #                 in_quotes = False
# # #                 for ch in tup:
# # #                     if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # #                         in_quotes = not in_quotes
# # #                     if ch == ',' and not in_quotes:
# # #                         parts.append(current.strip())
# # #                         current = ''
# # #                     else:
# # #                         current += ch
# # #                 parts.append(current.strip())
# # #             if len(parts) < 11:
# # #                 continue
# # #             def clean(s):
# # #                 s = s.strip()
# # #                 if s.startswith("'") and s.endswith("'"):
# # #                     s = s[1:-1]
# # #                 if s.startswith('"') and s.endswith('"'):
# # #                     s = s[1:-1]
# # #                 if s == 'NULL':
# # #                     return None
# # #                 return s
# # #             node = {
# # #                 'id': clean(parts[0]),
# # #                 'name': clean(parts[1]),
# # #                 'node_type': clean(parts[2]),
# # #                 'properties': clean(parts[3]),
# # #                 'description': clean(parts[4]),
# # #                 'embedding': clean(parts[5]),
# # #                 'parameters': clean(parts[6]),
# # #                 'metadata': clean(parts[7]),
# # #                 'created_at': clean(parts[8]),
# # #                 'updated_at': clean(parts[9]),
# # #                 'theme_id': clean(parts[10])
# # #             }
# # #             try:
# # #                 node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
# # #             except:
# # #                 node['metadata'] = {}
# # #             nodes.append(node)
# # #     return nodes
# # #
# # # def main():
# # #     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
# # #     nodes = parse_all_nodes(dump_file)
# # #     total = len(nodes)
# # #     print(f"Всего распарсено узлов: {total}", file=sys.stderr)
# # #
# # #     type_counts = defaultdict(int)
# # #     allowed_type_counts = defaultdict(int)
# # #     allowed_with_desc = defaultdict(int)
# # #     generated_count = 0
# # #     skipped_no_desc = 0
# # #     skipped_type = 0
# # #     generated_by_type = defaultdict(int)
# # #
# # #     # Здесь мы имитируем логику генерации (но без реального вызова LLM)
# # #     # Просто проверяем, что узел прошёл бы фильтры
# # #     for node in nodes:
# # #         nt = node['node_type']
# # #         if not nt:
# # #             continue
# # #         type_counts[nt] += 1
# # #         if nt.lower() in ALLOWED_TYPES:
# # #             allowed_type_counts[nt] += 1
# # #             if node['description']:
# # #                 allowed_with_desc[nt] += 1
# # #                 # Считаем, что генерация бы удалась (мы не вызываем LLM)
# # #                 generated_by_type[nt] += 1
# # #                 generated_count += 1
# # #             else:
# # #                 skipped_no_desc += 1
# # #         else:
# # #             skipped_type += 1
# # #
# # #     print(f"Узлы разрешённых типов: {sum(allowed_type_counts.values())}", file=sys.stderr)
# # #     print(f"Из них с непустым описанием: {sum(allowed_with_desc.values())}", file=sys.stderr)
# # #     print(f"Пропущено из-за пустого описания: {skipped_no_desc}", file=sys.stderr)
# # #     print(f"Пропущено из-за типа (не в ALLOWED_TYPES): {skipped_type}", file=sys.stderr)
# # #     print(f"Потенциально могли быть обработаны (генерация): {generated_count}", file=sys.stderr)
# # #     print("\nРаспределение по типам:", file=sys.stderr)
# # #     for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
# # #         allowed = "✅" if t.lower() in ALLOWED_TYPES else "❌"
# # #         desc_count = allowed_with_desc.get(t, 0)
# # #         gen_count = generated_by_type.get(t, 0)
# # #         print(f"  {allowed} {t}: {cnt} (с описанием: {desc_count}, обработано бы: {gen_count})", file=sys.stderr)
# # #
# # #     print("\nЕсли количество 'обработано бы' совпадает с количеством сгенерированных UPDATE,", file=sys.stderr)
# # #     print("то проблема не в фильтрации, а в работе самой LLM (ошибки генерации).", file=sys.stderr)
# # #
# # # if __name__ == '__main__':
# # #     main()
# # #
# # #
# # # # import re
# # # # import json
# # # # import sys
# # # # import csv
# # # # from io import StringIO
# # # # from collections import defaultdict
# # # # import time
# # # #
# # # # # Разрешаем все типы, которые есть в дампе
# # # # ALLOWED_TYPES = {
# # # #     'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
# # # #     'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
# # # #     'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
# # # # }
# # # #
# # # # def parse_inserts(filepath):
# # # #     with open(filepath, 'r', encoding='utf-8') as f:
# # # #         content = f.read()
# # # #
# # # #     # Находим все блоки INSERT
# # # #     insert_pattern = r"INSERT INTO agi_evolution\.knowledge_nodes \(id,\"name\",node_type,properties,description,embedding,parameters,metadata,created_at,updated_at,theme_id\) VALUES\s*"
# # # #     # Ищем всё до точки с запятой, но с учётом вложенности скобок
# # # #     # Проще: разбиваем по ");" и ищем начало INSERT
# # # #     blocks = re.split(r';\s*', content)
# # # #     nodes = []
# # # #     for block in blocks:
# # # #         if not block.strip().startswith('INSERT'):
# # # #             continue
# # # #         # Извлекаем часть после VALUES
# # # #         values_match = re.search(r'VALUES\s*\((.*)\)\s*$', block, re.DOTALL | re.IGNORECASE)
# # # #         if not values_match:
# # # #             continue
# # # #         values_str = values_match.group(1)
# # # #         # Теперь разбиваем на отдельные кортежи: разделитель "),("
# # # #         # Используем csv-парсер, но сначала нужно разделить кортежи
# # # #         tuples = []
# # # #         current = ''
# # # #         in_quotes = False
# # # #         paren_level = 0
# # # #         for ch in values_str:
# # # #             if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # # #                 in_quotes = not in_quotes
# # # #             if not in_quotes:
# # # #                 if ch == '(':
# # # #                     paren_level += 1
# # # #                 elif ch == ')':
# # # #                     paren_level -= 1
# # # #             if not in_quotes and ch == ',' and paren_level == 1:
# # # #                 # Это разделитель между кортежами на верхнем уровне
# # # #                 tuples.append(current.strip())
# # # #                 current = ''
# # # #                 continue
# # # #             current += ch
# # # #         if current.strip():
# # # #             tuples.append(current.strip())
# # # #
# # # #         # Теперь каждый кортеж парсим через csv
# # # #         for tup in tuples:
# # # #             # Убираем внешние скобки
# # # #             if tup.startswith('(') and tup.endswith(')'):
# # # #                 tup = tup[1:-1]
# # # #             try:
# # # #                 reader = csv.reader(StringIO(tup), quotechar="'", skipinitialspace=True)
# # # #                 parts = next(reader)
# # # #             except:
# # # #                 # fallback: простой разбор
# # # #                 parts = []
# # # #                 current = ''
# # # #                 in_quotes = False
# # # #                 for ch in tup:
# # # #                     if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # # #                         in_quotes = not in_quotes
# # # #                     if ch == ',' and not in_quotes:
# # # #                         parts.append(current.strip())
# # # #                         current = ''
# # # #                     else:
# # # #                         current += ch
# # # #                 parts.append(current.strip())
# # # #             if len(parts) < 11:
# # # #                 continue
# # # #             def clean(s):
# # # #                 s = s.strip()
# # # #                 if s.startswith("'") and s.endswith("'"):
# # # #                     s = s[1:-1]
# # # #                 if s.startswith('"') and s.endswith('"'):
# # # #                     s = s[1:-1]
# # # #                 if s == 'NULL':
# # # #                     return None
# # # #                 return s
# # # #             node = {
# # # #                 'id': clean(parts[0]),
# # # #                 'name': clean(parts[1]),
# # # #                 'node_type': clean(parts[2]),
# # # #                 'properties': clean(parts[3]),
# # # #                 'description': clean(parts[4]),
# # # #                 'embedding': clean(parts[5]),
# # # #                 'parameters': clean(parts[6]),
# # # #                 'metadata': clean(parts[7]),
# # # #                 'created_at': clean(parts[8]),
# # # #                 'updated_at': clean(parts[9]),
# # # #                 'theme_id': clean(parts[10])
# # # #             }
# # # #             try:
# # # #                 node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
# # # #             except:
# # # #                 node['metadata'] = {}
# # # #             nodes.append(node)
# # # #     return nodes
# # # #
# # # # # Остальная часть скрипта (generate_properties_with_llm и main) остаётся без изменений
# # # # # ...
# # # #
# # # # def generate_properties_with_llm(name, node_type, description):
# # # #     # Здесь ваш код для вызова локальной модели
# # # #     # ...
# # # #     return {"название": name, "описание": description[:100] if description else ""}
# # # #
# # # # def main():
# # # #     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
# # # #     nodes = parse_inserts(dump_file)
# # # #     print(f"Всего распарсено узлов: {len(nodes)}", file=sys.stderr)
# # # #
# # # #     print("BEGIN;")
# # # #     updated = 0
# # # #     skipped = 0
# # # #     errors = 0
# # # #
# # # #     for node in nodes:
# # # #         nt = node['node_type']
# # # #         if not nt:
# # # #             skipped += 1
# # # #             continue
# # # #         if nt.lower() not in ALLOWED_TYPES:
# # # #             skipped += 1
# # # #             continue
# # # #         if not node['description']:
# # # #             skipped += 1
# # # #             continue
# # # #
# # # #         props = None
# # # #         for attempt in range(3):
# # # #             try:
# # # #                 props = generate_properties_with_llm(node['name'], nt, node['description'])
# # # #                 if props:
# # # #                     break
# # # #             except Exception as e:
# # # #                 print(f"Error for {node['name']} (attempt {attempt+1}): {e}", file=sys.stderr)
# # # #                 time.sleep(1)
# # # #
# # # #         if not props:
# # # #             errors += 1
# # # #             continue
# # # #
# # # #         json_str = json.dumps(props, ensure_ascii=False)
# # # #         json_str_escaped = json_str.replace("'", "''")
# # # #         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
# # # #         updated += 1
# # # #
# # # #     print("COMMIT;")
# # # #     print(f"-- Updated: {updated}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)
# # # #
# # # # if __name__ == '__main__':
# # # #     main()
# # # #
# # # #
# # # #
# # # #
# # # # # import json
# # # # # import re
# # # # # import sys
# # # # # import time
# # # # # import csv
# # # # # from io import StringIO
# # # # # from collections import defaultdict
# # # # #
# # # # # # Для локальной модели (используйте свою)
# # # # # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # # # # или замените на вашу функцию generate_with_llm
# # # # #
# # # # # # Разрешаем все типы, которые есть в дампе
# # # # # ALLOWED_TYPES = {
# # # # #     'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
# # # # #     'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
# # # # #     'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
# # # # # }
# # # # #
# # # # # def generate_properties_with_llm(name, node_type, description):
# # # # #     """
# # # # #     Здесь вставьте ваш код для вызова локальной модели.
# # # # #     Возвращает словарь свойств или None при ошибке.
# # # # #     """
# # # # #     # Пример заглушки – замените на реальный вызов модели
# # # # #     # ...
# # # # #     # Для демонстрации возвращаем базовый набор
# # # # #     return {
# # # # #         "название": name,
# # # # #         "описание": description[:100] if description else ""
# # # # #     }
# # # # #
# # # # # def parse_inserts(filepath):
# # # # #     with open(filepath, 'r', encoding='utf-8') as f:
# # # # #         content = f.read()
# # # # #     pattern = r"INSERT INTO agi_evolution\.knowledge_nodes \(id,\"name\",node_type,properties,description,embedding,parameters,metadata,created_at,updated_at,theme_id\) VALUES\s*\((.*?)\);"
# # # # #     matches = re.findall(pattern, content, re.DOTALL)
# # # # #     nodes = []
# # # # #     for match in matches:
# # # # #         try:
# # # # #             reader = csv.reader(StringIO(match), quotechar="'", skipinitialspace=True)
# # # # #             parts = next(reader)
# # # # #         except:
# # # # #             parts = []
# # # # #             current = ''
# # # # #             in_quotes = False
# # # # #             for ch in match:
# # # # #                 if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # # # #                     in_quotes = not in_quotes
# # # # #                 if ch == ',' and not in_quotes:
# # # # #                     parts.append(current.strip())
# # # # #                     current = ''
# # # # #                 else:
# # # # #                     current += ch
# # # # #             parts.append(current.strip())
# # # # #         if len(parts) < 11:
# # # # #             continue
# # # # #         def clean(s):
# # # # #             s = s.strip()
# # # # #             if s.startswith("'") and s.endswith("'"):
# # # # #                 s = s[1:-1]
# # # # #             if s.startswith('"') and s.endswith('"'):
# # # # #                 s = s[1:-1]
# # # # #             if s == 'NULL':
# # # # #                 return None
# # # # #             return s
# # # # #         node = {
# # # # #             'id': clean(parts[0]),
# # # # #             'name': clean(parts[1]),
# # # # #             'node_type': clean(parts[2]),
# # # # #             'properties': clean(parts[3]),
# # # # #             'description': clean(parts[4]),
# # # # #             'embedding': clean(parts[5]),
# # # # #             'parameters': clean(parts[6]),
# # # # #             'metadata': clean(parts[7]),
# # # # #             'created_at': clean(parts[8]),
# # # # #             'updated_at': clean(parts[9]),
# # # # #             'theme_id': clean(parts[10])
# # # # #         }
# # # # #         try:
# # # # #             node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
# # # # #         except:
# # # # #             node['metadata'] = {}
# # # # #         nodes.append(node)
# # # # #     return nodes
# # # # #
# # # # # def main():
# # # # #     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
# # # # #     nodes = parse_inserts(dump_file)
# # # # #
# # # # #     print("BEGIN;")
# # # # #     updated = 0
# # # # #     skipped = 0
# # # # #     errors = 0
# # # # #
# # # # #     for node in nodes:
# # # # #         nt = node['node_type']
# # # # #         if not nt:
# # # # #             skipped += 1
# # # # #             continue
# # # # #         if nt.lower() not in ALLOWED_TYPES:
# # # # #             skipped += 1
# # # # #             continue
# # # # #         if not node['description']:
# # # # #             skipped += 1
# # # # #             continue
# # # # #
# # # # #         # Пытаемся сгенерировать свойства (до 3 попыток)
# # # # #         props = None
# # # # #         for attempt in range(3):
# # # # #             try:
# # # # #                 props = generate_properties_with_llm(node['name'], nt, node['description'])
# # # # #                 if props:
# # # # #                     break
# # # # #             except Exception as e:
# # # # #                 print(f"Error for {node['name']} (attempt {attempt+1}): {e}", file=sys.stderr)
# # # # #                 time.sleep(1)
# # # # #
# # # # #         if not props:
# # # # #             errors += 1
# # # # #             continue
# # # # #
# # # # #         json_str = json.dumps(props, ensure_ascii=False)
# # # # #         json_str_escaped = json_str.replace("'", "''")
# # # # #         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
# # # # #         updated += 1
# # # # #
# # # # #     print("COMMIT;")
# # # # #     print(f"-- Updated: {updated}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)
# # # # #
# # # # # if __name__ == '__main__':
# # # # #     main()
# # # # #
# # # # #
# # # # #
# # # # #
# # # # #
# # # # #
# # # # # # import requests
# # # # # # import json
# # # # # # import re
# # # # # # import sys
# # # # # # import time
# # # # # # import csv
# # # # # # from io import StringIO
# # # # # #
# # # # # # # Настройки
# # # # # # OLLAMA_URL = "http://localhost:11434/api/generate"
# # # # # # MODEL_NAME = "qwen2.5:7b"  # или любая другая
# # # # # # ALLOWED_TYPES = {'part', 'component', 'device', 'mechanism', 'material', 'system', 'model'}
# # # # # #
# # # # # # def generate_properties_ollama(name, node_type, description):
# # # # # #     prompt = f"""Ты — эксперт по истории авиации и аэродинамике. Для узла знания:
# # # # # # - Имя: {name}
# # # # # # - Тип: {node_type}
# # # # # # - Описание: {description}
# # # # # #
# # # # # # Сгенерируй свойства в виде JSON-объекта. Ключи зависят от типа:
# # # # # # - PERSON: имя, роль, вклад, период, страна
# # # # # # - DEVICE: название, тип, функция, изобретатель, год
# # # # # # - SYSTEM: назначение, компоненты, принцип
# # # # # # - MODEL: тип, область, автор, год
# # # # # # - COMPONENT: название, роль, материал
# # # # # # - PART: название, функция, материал
# # # # # # - MECHANISM: название, принцип, изобретатель
# # # # # # - MATERIAL: название, свойства, применение
# # # # # #
# # # # # # Ответ должен быть только JSON, без пояснений.
# # # # # # """
# # # # # #     payload = {
# # # # # #         "model": MODEL_NAME,
# # # # # #         "prompt": prompt,
# # # # # #         "stream": False,
# # # # # #         "temperature": 0.2,
# # # # # #         "max_tokens": 500,
# # # # # #         "format": "json"  # если модель поддерживает (у qwen не всегда)
# # # # # #     }
# # # # # #     try:
# # # # # #         response = requests.post(OLLAMA_URL, json=payload, timeout=120)
# # # # # #         if response.status_code == 200:
# # # # # #             data = response.json()
# # # # # #             content = data.get("response", "")
# # # # # #             json_match = re.search(r'\{.*\}', content, re.DOTALL)
# # # # # #             if json_match:
# # # # # #                 try:
# # # # # #                     return json.loads(json_match.group())
# # # # # #                 except json.JSONDecodeError:
# # # # # #                     return None
# # # # # #             else:
# # # # # #                 return None
# # # # # #         else:
# # # # # #             return None
# # # # # #     except Exception as e:
# # # # # #         print(f"Error for {name}: {e}", file=sys.stderr)
# # # # # #         return None
# # # # # #
# # # # # # def parse_inserts(filepath):
# # # # # #     with open(filepath, 'r', encoding='utf-8') as f:
# # # # # #         content = f.read()
# # # # # #     pattern = r"INSERT INTO agi_evolution\.knowledge_nodes \(id,\"name\",node_type,properties,description,embedding,parameters,metadata,created_at,updated_at,theme_id\) VALUES\s*\((.*?)\);"
# # # # # #     matches = re.findall(pattern, content, re.DOTALL)
# # # # # #     nodes = []
# # # # # #     for match in matches:
# # # # # #         try:
# # # # # #             reader = csv.reader(StringIO(match), quotechar="'", skipinitialspace=True)
# # # # # #             parts = next(reader)
# # # # # #         except:
# # # # # #             parts = []
# # # # # #             current = ''
# # # # # #             in_quotes = False
# # # # # #             for ch in match:
# # # # # #                 if ch == "'" and (len(current) == 0 or current[-1] != '\\'):
# # # # # #                     in_quotes = not in_quotes
# # # # # #                 if ch == ',' and not in_quotes:
# # # # # #                     parts.append(current.strip())
# # # # # #                     current = ''
# # # # # #                 else:
# # # # # #                     current += ch
# # # # # #             parts.append(current.strip())
# # # # # #         if len(parts) < 11:
# # # # # #             continue
# # # # # #         def clean(s):
# # # # # #             s = s.strip()
# # # # # #             if s.startswith("'") and s.endswith("'"):
# # # # # #                 s = s[1:-1]
# # # # # #             if s.startswith('"') and s.endswith('"'):
# # # # # #                 s = s[1:-1]
# # # # # #             if s == 'NULL':
# # # # # #                 return None
# # # # # #             return s
# # # # # #         node = {
# # # # # #             'id': clean(parts[0]),
# # # # # #             'name': clean(parts[1]),
# # # # # #             'node_type': clean(parts[2]),
# # # # # #             'properties': clean(parts[3]),
# # # # # #             'description': clean(parts[4]),
# # # # # #             'embedding': clean(parts[5]),
# # # # # #             'parameters': clean(parts[6]),
# # # # # #             'metadata': clean(parts[7]),
# # # # # #             'created_at': clean(parts[8]),
# # # # # #             'updated_at': clean(parts[9]),
# # # # # #             'theme_id': clean(parts[10])
# # # # # #         }
# # # # # #         try:
# # # # # #             node['metadata'] = json.loads(node['metadata']) if node['metadata'] else {}
# # # # # #         except:
# # # # # #             node['metadata'] = {}
# # # # # #         nodes.append(node)
# # # # # #     return nodes
# # # # # #
# # # # # # def main():
# # # # # #     dump_file = sys.argv[1] if len(sys.argv) > 1 else 'dump.sql'
# # # # # #     nodes = parse_inserts(dump_file)
# # # # # #
# # # # # #     # Выводим диагностику
# # # # # #     type_counts = {}
# # # # # #     for node in nodes:
# # # # # #         nt = node['node_type']
# # # # # #         if nt is None:
# # # # # #             nt = 'NULL'
# # # # # #         type_counts[nt] = type_counts.get(nt, 0) + 1
# # # # # #     print("-- Types found in dump:", file=sys.stderr)
# # # # # #     for t, cnt in sorted(type_counts.items()):
# # # # # #         print(f"--   '{t}': {cnt}", file=sys.stderr)
# # # # # #     print("--", file=sys.stderr)
# # # # # #
# # # # # #     print("BEGIN;")
# # # # # #     count = 0
# # # # # #     for node in nodes:
# # # # # #         nt = node['node_type']
# # # # # #         if not nt or nt.lower() not in ALLOWED_TYPES:
# # # # # #             continue
# # # # # #         print(f"-- Processing {node['name']} ({nt})", file=sys.stderr)
# # # # # #         props = generate_properties_ollama(node['name'], nt, node['description'])
# # # # # #         if not props:
# # # # # #             print(f"-- Failed to generate for {node['name']}, skipping", file=sys.stderr)
# # # # # #             continue
# # # # # #         json_str = json.dumps(props, ensure_ascii=False)
# # # # # #         json_str_escaped = json_str.replace("'", "''")
# # # # # #         print(f"UPDATE agi_evolution.knowledge_nodes SET properties = ARRAY['{json_str_escaped}']::text[] WHERE id = '{node['id']}';")
# # # # # #         count += 1
# # # # # #         time.sleep(0.5)  # небольшая задержка
# # # # # #     print("COMMIT;")
# # # # # #     print(f"-- Updated {count} nodes.", file=sys.stderr)
# # # # # #
# # # # # # if __name__ == '__main__':
# # # # # #     main()