import json
import sys
import time
import sqlparse

ALLOWED_TYPES = {
    'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
    'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
    'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
}

def split_fields(tup_str):
    """Разбивает строку одного кортежа на отдельные поля."""
    fields = []
    current = ''
    in_single_quote = False
    in_double_quote = False
    escape = False
    paren_level = 0
    for ch in tup_str:
        if escape:
            escape = False
            current += ch
            continue
        if ch == '\\':
            escape = True
            current += ch
            continue
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current += ch
            continue
        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current += ch
            continue
        if not in_single_quote and not in_double_quote:
            if ch == '(':
                paren_level += 1
                current += ch
                continue
            elif ch == ')':
                paren_level -= 1
                current += ch
                continue
            elif ch == ',' and paren_level == 0:
                fields.append(current.strip())
                current = ''
                continue
        current += ch
    if current.strip():
        fields.append(current.strip())
    return fields

def parse_tuple(tup_str):
    """Парсит один кортеж и возвращает словарь узла."""
    tup_str = tup_str.strip()
    # Убираем внешние скобки, если они есть (они будут, так как мы извлекаем кортежи с ними)
    if tup_str.startswith('(') and tup_str.endswith(')'):
        tup_str = tup_str[1:-1]

    parts = split_fields(tup_str)
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

def extract_tuples(values_str):
    """Извлекает все кортежи из строки VALUES, используя балансировку скобок."""
    tuples = []
    i = 0
    n = len(values_str)
    while i < n:
        if values_str[i] == '(':
            start = i
            paren_level = 0
            in_single = False
            in_double = False
            escape = False
            while i < n:
                ch = values_str[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if ch == '\\':
                    escape = True
                    i += 1
                    continue
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == '(':
                        paren_level += 1
                    elif ch == ')':
                        paren_level -= 1
                        if paren_level == 0:
                            # Нашли конец кортежа
                            tuples.append(values_str[start:i+1].strip())
                            i += 1
                            break
                i += 1
            else:
                # Не нашли закрывающую скобку — выходим
                break
        else:
            i += 1
    return tuples

def parse_inserts(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    statements = sqlparse.split(content)
    nodes = []
    for stmt in statements:
        if not stmt.strip().upper().startswith('INSERT'):
            continue
        values_pos = stmt.find("VALUES")
        if values_pos == -1:
            continue
        open_paren = stmt.find("(", values_pos)
        if open_paren == -1:
            continue
        # Балансируем скобки, чтобы найти конец внешнего списка VALUES
        i = open_paren + 1
        paren_level = 1
        in_single_quote = False
        in_double_quote = False
        escape = False
        while i < len(stmt):
            ch = stmt[i]
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
            continue
        values_str = stmt[open_paren + 1:end]
        tuples = extract_tuples(values_str)
        for tup in tuples:
            node = parse_tuple(tup)
            if node:
                nodes.append(node)
    return nodes

def generate_properties_with_llm(name, node_type, description):
    # Замените на ваш реальный вызов локальной модели
    # Пока заглушка
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
                print(f"Error for {node['name']} (attempt {attempt + 1}): {e}", file=sys.stderr)
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