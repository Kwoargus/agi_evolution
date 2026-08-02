import json

with open('training_data_inventions.json', 'r') as f:
    data = json.load(f)

with open('name_to_id.json', 'r') as f:
    name_to_id = json.load(f)

def extract_names(combo):
    if isinstance(combo, list):
        # Если список, то каждый элемент может быть строкой или словарём с ключом "analogy"
        result = []
        for item in combo:
            if isinstance(item, dict) and 'analogy' in item:
                result.extend(item['analogy'])
            elif isinstance(item, list):
                result.extend(extract_names(item))
            else:
                result.append(item)
        return result
    elif isinstance(combo, dict) and 'analogy' in combo:
        return combo['analogy']
    else:
        return [combo]  # fallback

normalized = []
for item in data:
    task = item.get('task')
    score = item.get('score')
    if score is None:
        # может быть вложенный score
        if isinstance(item.get('combination'), dict) and 'score' in item['combination']:
            score = item['combination']['score']
        else:
            continue
    names = extract_names(item.get('combination', []))
    # Заменяем названия на ID
    ids = [name_to_id.get(name, name) for name in names]  # если нет в словаре, оставляем как есть
    # Создаём новую запись с плоским списком ID
    normalized.append({
        'task': task,
        'combination': ids,
        'score': score,
        'source': item.get('source', 'unknown')
    })

with open('training_data_inventions_normalized.json', 'w') as f:
    json.dump(normalized, f, indent=2, ensure_ascii=False)

print(f"Обработано {len(normalized)} записей.")