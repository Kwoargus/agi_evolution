import json
import sys
import time
import sqlparse
from sqlparse.sql import Values, Parenthesis, Identifier
from sqlparse.tokens import DML

ALLOWED_TYPES = {
    'part', 'component', 'device', 'mechanism', 'material', 'system', 'model',
    'person', 'method', 'property', 'state', 'group', 'facility', 'phenomenon',
    'geo', 'force', 'parameter', 'principle', 'numeric', 'segment', 'principal stresses'
}

def split_fields(tup_str):
    """Разбивает строку кортежа на поля, учитывая кавычки и вложенные скобки."""
    fields = []
    current = ''
    in_single = False
    in_double = False
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
        if ch == "'" and not in_double:
            in_single = not in_single
            current += ch
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current += ch
            continue
        if not in_single and not in_double:
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
    """Парсит строку кортежа и возвращает словарь узла."""
    tup_str = tup_str.strip()
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

def parse_inserts(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    statements = sqlparse.split(content)
    nodes = []

    for stmt in statements:
        if not stmt.strip().upper().startswith('INSERT'):
            continue

        parsed = sqlparse.parse(stmt)[0]
        values_token = None
        # Ищем токен VALUES
        for token in parsed.tokens:
            if token.ttype is DML and token.value.upper() == 'VALUES':
                values_token = token
                break
            # Иногда VALUES представлен как Parenthesis
            if isinstance(token, Values):
                values_token = token
                break
            # Ищем вложенные Parenthesis
            if isinstance(token, Parenthesis):
                # Проверяем, не является ли этот Parenthesis списком VALUES
                if token.value.upper().startswith('VALUES'):
                    values_token = token
                    break

        if not values_token:
            # Попробуем найти Parenthesis на верхнем уровне
            for token in parsed.tokens:
                if isinstance(token, Parenthesis):
                    values_token = token
                    break

        if not values_token:
            continue

        # Получаем строку значений
        if isinstance(values_token, Values):
            # У Values есть метод get_identifiers()?
            if hasattr(values_token, 'get_identifiers'):
                identifiers = values_token.get_identifiers()
            else:
                # Берём весь текст и разбираем вручную
                value_str = values_token.value
                if value_str.upper().startswith('VALUES'):
                    value_str = value_str[len('VALUES'):].strip()
                # Извлекаем кортежи с помощью extract_tuples
                tuples = extract_tuples(value_str)
                for tup in tuples:
                    node = parse_tuple(tup)
                    if node:
                        nodes.append(node)
                continue
        else:
            # Это Parenthesis – берём его содержимое
            value_str = values_token.value
            if value_str.startswith('(') and value_str.endswith(')'):
                value_str = value_str[1:-1]
            tuples = extract_tuples(value_str)
            for tup in tuples:
                node = parse_tuple(tup)
                if node:
                    nodes.append(node)
            continue

        # Если есть identifiers, обрабатываем их
        for ident in identifiers:
            if isinstance(ident, Identifier):
                # Токен идентификатора может содержать кортеж
                tup_str = ident.value
                node = parse_tuple(tup_str)
                if node:
                    nodes.append(node)
            elif isinstance(ident, Parenthesis):
                tup_str = ident.value
                node = parse_tuple(tup_str)
                if node:
                    nodes.append(node)
            else:
                # Просто строка
                tup_str = str(ident)
                node = parse_tuple(tup_str)
                if node:
                    nodes.append(node)

    return nodes

def extract_tuples(values_str):
    """Извлекает все кортежи из строки, балансируя скобки."""
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
                            tuples.append(values_str[start:i+1].strip())
                            i += 1
                            break
                i += 1
            else:
                break
        else:
            i += 1
    return tuples

def generate_properties_with_llm(name, node_type, description):
    # Замените на реальный вызов вашей модели
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