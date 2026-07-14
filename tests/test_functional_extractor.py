# tests/test_functional_extractor.py
"""
Тест извлечения функциональных свойств.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thinking.functional_extractor import FunctionalExtractor


def test_extractor():
    print("\n" + "=" * 70)
    print("🧠 ТЕСТ ИЗВЛЕЧЕНИЯ ФУНКЦИОНАЛЬНЫХ СВОЙСТВ (СЛОВАРЬ+ШАБЛОНЫ)")
    print("=" * 70)

    extractor = FunctionalExtractor(use_llm=False)  # Отключаем LLM для чистоты теста
    print("DEBUG: keyword_to_function keys:", list(extractor.keyword_to_function.keys()))


    test_cases = [
        {
            'task': "Нужен летательный аппарат для перевозки грузов",
            'expected': ['создавать подъёмную силу', 'поднимать груз']
        },
        {
            'task': "Требуется разработать систему управления движением робота",
            'expected': ['управлять полётом', 'исполнять команды', 'принимать решения']
        },
        {
            'task': "Необходимо создать механизм для подъема тяжелых предметов",
            'expected': ['поднимать груз', 'удерживать нагрузку']
        },
        {
            'task': "Нужен двигатель для преобразования тепловой энергии в механическую",
            'expected': ['преобразовывать энергию', 'создавать тягу']
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 ЗАДАЧА {i}: {test_case['task']}")
        print("=" * 70)

        # Извлекаем свойства
        props = extractor.extract(test_case['task'])

        print(f"\n📊 Извлечено свойств: {len(props)}")
        for prop in props:
            print(f"   - {prop.name} (кат: {prop.category}, источник: {prop.source}, уверенность: {prop.confidence:.2f})")

        # Проверяем ожидаемые свойства
        found_expected = []
        for expected in test_case['expected']:
            found = any(expected == p.name for p in props)
            found_expected.append((expected, found))

        print(f"\n🔍 Проверка ожидаемых свойств:")
        for expected, found in found_expected:
            status = "✅" if found else "❌"
            print(f"   {status} {expected}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")


def test_extractor_with_llm():
    print("\n" + "=" * 70)
    print("🧠 ТЕСТ ИЗВЛЕЧЕНИЯ С LLM (ГИБРИДНЫЙ)")
    print("=" * 70)

    # Создаём экстрактор с LLM (qwen7B)
    extractor = FunctionalExtractor(use_llm=True, llm_model="qwen7B")

    # Сложная задача, где словарь не поможет
    task = "Нужно устройство, которое может парить в воздухе, перемещаться в любом направлении, поднимать грузы до 100 кг и работать от солнечной энергии"

    print(f"\n📝 ЗАДАЧА: {task}")

    # Извлекаем свойства
    props = extractor.extract(task)

    print(f"\n📊 Извлечено свойств: {len(props)}")
    for prop in props:
        print(f"   - {prop.name} (кат: {prop.category}, источник: {prop.source}, уверенность: {prop.confidence:.2f})")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ С LLM ЗАВЕРШЕН")


if __name__ == "__main__":
    # Сначала тест без LLM
    test_extractor()

    # Затем тест с LLM (если есть доступ к Ollama)
    # Раскомментируй следующую строку, если Ollama запущен
    # test_extractor_with_llm()