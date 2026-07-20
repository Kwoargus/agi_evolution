import json
import sys
import sqlparse

def extract_tuples(values_str):
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

def debug_parse_inserts(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    statements = sqlparse.split(content)
    for stmt in statements:
        if not stmt.strip().upper().startswith('INSERT'):
            continue
        values_pos = stmt.find("VALUES")
        if values_pos == -1:
            continue
        open_paren = stmt.find("(", values_pos)
        if open_paren == -1:
            continue
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
        print(f"DEBUG: values_str length = {len(values_str)}")
        print(f"DEBUG: values_str = {repr(values_str[:500])}")
        tuples = extract_tuples(values_str)
        print(f"DEBUG: найдено кортежей: {len(tuples)}")
        if tuples:
            print(f"Первый кортеж: {repr(tuples[0][:200])}")
        break

if __name__ == '__main__':
    debug_parse_inserts('dump.sql')