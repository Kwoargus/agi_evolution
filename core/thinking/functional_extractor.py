# core/thinking/functional_extractor.py
"""
[ru] Извлечение функциональных свойств из текста задачи. ГИБРИДНЫЙ подход: словарь + шаблоны + LLM (как fallback).
[en] Extracting functional properties from task text. HYBRID approach: dictionary + patterns + LLM (as fallback).
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
import time


class FunctionalProperty:
    """
    [ru] Функциональное свойство с метаданными.
    [en] Functional property with metadata.
    """

    def __init__(self, name: str, category: str, confidence: float = 1.0,
                 source: str = "extracted", synonyms: List[str] = None):
        self.name = name
        self.category = category  # 'physics', 'mechanics', 'control', 'energy', etc.
        self.confidence = confidence
        self.source = source
        self.synonyms = synonyms or []

    def __repr__(self):
        return f"FunctionalProperty({self.name}, cat={self.category}, conf={self.confidence:.2f})"


class FunctionalExtractor:
    """
    [ru] Извлекает функциональные свойства из текста задачи. ГИБРИДНЫЙ подход: словарь + шаблоны + LLM (как fallback).
    [en] Extracts functional properties from task text. HYBRID approach: dictionary + patterns + LLM (as fallback).
    """

    # ============================================================
    # [ru] БАЗА ЗНАНИЙ: ФУНКЦИОНАЛЬНЫЕ СВОЙСТВА → КЛЮЧЕВЫЕ СЛОВА
    # [en] KNOWLEDGE BASE: FUNCTIONAL PROPERTIES → KEYWORDS
    # ============================================================

    FUNCTIONAL_MAP = {
        # ============================================================
        # [ru] Аэродинамика и полёт
        # [en] Aerodynamics and flight
        # ============================================================
        'создавать подъёмную силу': {
            'keywords': ['подъём', 'подъем', 'лететь', 'взлететь', 'парить',
                         'подниматься', 'летательный', 'летать', 'воздушный',
                         'аэродинамический', 'крыло'],
            'category': 'aerodynamics',
            'synonyms': ['подъёмная сила', 'lift', 'аэродинамическая сила']
        },

        'создавать тягу': {
            'keywords': ['тяга', 'движение', 'разгон', 'ускорение', 'толкать',
                         'пропеллер', 'винт', 'реактивный', 'двигатель', 'двигателя', 'двигателю', 'двигателем', 'двигателе', 'мотор'],
            'category': 'propulsion',
            'synonyms': ['тяговое усилие', 'двигательная сила']
        },

        'преобразовывать энергию': {
            'keywords': ['преобразовывать', 'энергия', 'конвертировать', 'тепловой',
                         'механический', 'электрический', 'солнечный', 'конверсия'],
            'category': 'energy',
            'synonyms': ['конверсия энергии', 'трансформация']
        },
        'управлять полётом': {
            'keywords': ['управлять', 'направлять', 'маневрировать', 'контролировать',
                         'рулить', 'автопилот', 'стабилизация', 'навигация'],
            'category': 'control',
            'synonyms': ['навигация', 'управление']
        },

        # ============================================================
        # [ru] Механика и робототехника
        # [en] Mechanics and robotics
        # ============================================================
        'передавать движение': {
            'keywords': ['передавать', 'вращение', 'крутящий момент', 'трансмиссия',
                         'редуктор', 'вал', 'привод'],
            'category': 'mechanics',
            'synonyms': ['трансмиссия', 'привод']
        },
        'поднимать груз': {
            'keywords': ['поднимать', 'подъем', 'подъём', 'грузить', 'перевозить',
                         'транспортировать', 'перевозка', 'груз', 'транспортировка',
                         'доставлять', 'тяжелый', 'грузоподъемность'],
            'category': 'mechanics',
            'synonyms': ['грузоподъёмность', 'подъём']
        },
        'удерживать нагрузку': {
            'keywords': ['удерживать', 'фиксировать', 'крепление', 'захват',
                         'держать', 'стопор', 'зажим'],
            'category': 'mechanics',
            'synonyms': ['захват', 'фиксация']
        },
        'двигаться': {
            'keywords': ['двигаться', 'движение', 'перемещаться', 'ход', 'ездить'],
            'category': 'mechanics',
            'synonyms': ['передвижение', 'мобильность']
        },
        'поворачивать': {
            'keywords': ['поворачивать', 'поворот', 'вращать', 'маневр'],
            'category': 'mechanics',
            'synonyms': ['поворот', 'вращение']
        },

        # ============================================================
        # [ru] Управление и ИИ
        # [en] Control and AI
        # ============================================================
        'управлять системой': {
            'keywords': ['управлять', 'управление', 'регулировать', 'контролировать',
                         'система', 'автоматизация'],
            'category': 'control',
            'synonyms': ['управление', 'регулировка']
        },
        'обрабатывать данные': {
            'keywords': ['обрабатывать', 'данные', 'информация', 'сигнал', 'сенсор',
                         'датчик', 'анализировать', 'измерять'],
            'category': 'control',
            'synonyms': ['обработка сигналов', 'анализ данных']
        },
        'принимать решения': {
            'keywords': ['решение', 'алгоритм', 'логика', 'выбор', 'оптимизация',
                         'планировать', 'рассчитывать', 'интеллект'],
            'category': 'intelligence',
            'synonyms': ['принятие решений', 'планирование']
        },
        'исполнять команды': {
            'keywords': ['исполнять', 'команда', 'действие', 'реакция', 'отклик',
                         'выполнять', 'реализовывать', 'актуатор'],
            'category': 'control',
            'synonyms': ['выполнение', 'реализация']
        },
        'воспринимать окружение': {
            'keywords': ['воспринимать', 'сенсор', 'датчик', 'зрение', 'локализация',
                         'карта', 'обнаружение'],
            'category': 'perception',
            'synonyms': ['сенсорика', 'восприятие']
        },

        # ============================================================
        # [ru] Физика и энергия
        # [en] Physics and energy
        # ============================================================
        'создавать давление': {
            'keywords': ['давление', 'сжатие', 'расширение', 'газ', 'жидкость',
                         'гидравлика', 'пневматика'],
            'category': 'physics',
            'synonyms': ['гидравлика', 'пневматика']
        },
        'нагревать или охлаждать': {
            'keywords': ['нагревать', 'охлаждать', 'температура', 'тепло', 'холод',
                         'терморегуляция', 'теплообмен'],
            'category': 'thermodynamics',
            'synonyms': ['терморегуляция', 'теплообмен']
        },

        # ============================================================
        # [ru] Логистика и транспорт
        # [en] Logistics and transport
        # ============================================================
        'перевозить': {
            'keywords': ['перевозить', 'перевозка', 'транспортировать', 'доставлять',
                         'груз', 'транспорт', 'логистика'],
            'category': 'logistics',
            'synonyms': ['транспортировка', 'грузоперевозки']
        },
        'поднимать': {
            'keywords': ['поднимать', 'подъем', 'подъём', 'лебедка', 'кран', 'такелаж'],
            'category': 'mechanics',
            'synonyms': ['подъём', 'поднятие']
        },

        # ============================================================
        # [ru] Робототехника
        # [en] Robotics
        # ============================================================
        'двигаться по поверхности': {
            'keywords': ['ходьба', 'шагать', 'колесо', 'гусеница', 'передвижение'],
            'category': 'robotics',
            'synonyms': ['локомоция', 'передвижение']
        },
        'манипулировать объектами': {
            'keywords': ['манипулировать', 'захват', 'рука', 'схватить', 'манипулятор'],
            'category': 'robotics',
            'synonyms': ['манипуляция', 'схват']
        },
    }

    # ============================================================
    # [ru] ШАБЛОНЫ ДЛЯ ИЗВЛЕЧЕНИЯ
    # [en] PATTERNS FOR EXTRACTION
    # ============================================================

    PATTERNS = [
        (r'для\s+([а-яА-Яa-zA-Z\s]+)', 'purpose'),
        (r'чтобы\s+([а-яА-Яa-zA-Z\s]+)', 'purpose'),
        (r'который\s+([а-яА-Яa-zA-Z]+)', 'verb'),
        (r'способность\s+([а-яА-Яa-zA-Z]+)', 'ability'),
        (r'обеспечивать\s+([а-яА-Яa-zA-Z]+)', 'provide'),
        (r'может\s+([а-яА-Яa-zA-Z]+)', 'can'),
    ]

    def __init__(self, use_llm: bool = True, llm_model: str = "qwen7B"):
        """
        [ru]
        Args:
            use_llm: Использовать LLM как fallback
            llm_model: Модель для использования ('llama7B', 'qwen7B', 'mistral7B')
        [en]
        Args:
            use_llm: Use LLM as fallback
            llm_model: Model to use ('llama7B', 'qwen7B', 'mistral7B')
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._build_indexes()

        # [ru] Кэш для результатов LLM
        # [en] Cache for LLM results
        self.llm_cache = {}

    def _build_indexes(self):
        """
        [ru] Строит индексы для быстрого поиска.
        [en] Builds indexes for fast search.
        """
        self.keyword_to_function = {}
        self.synonym_to_function = {}

        for function, data in self.FUNCTIONAL_MAP.items():
            for keyword in data['keywords']:
                self.keyword_to_function[keyword] = function
            for synonym in data.get('synonyms', []):
                self.synonym_to_function[synonym] = function

    def extract(self, text: str, use_patterns: bool = True) -> List[FunctionalProperty]:
        """
        [ru] Извлекает функциональные свойства из текста. ГИБРИДНЫЙ метод: сначала словарь, если мало → LLM.
        [en] Extracts functional properties from text. HYBRID method: dictionary first, if few → LLM.
        """
        properties = []
        seen = set()
        text_lower = text.lower()

        # ============================================================
        # [ru] ШАГ 1: СЛОВАРЬ + ШАБЛОНЫ (быстрый путь)
        # [en] STEP 1: DICTIONARY + PATTERNS (fast path)
        # ============================================================

        # [ru] 1. Поиск по ключевым словам
        # [en] 1. Search by keywords
        for keyword, function in self.keyword_to_function.items():
            if keyword in text_lower and function not in seen:
                print(f"DEBUG: найдено ключевое слово '{keyword}' → функция '{function}'")
                data = self.FUNCTIONAL_MAP[function]
                prop = FunctionalProperty(
                    name=function,
                    category=data['category'],
                    confidence=0.9,
                    source='keyword'
                )
                properties.append(prop)
                seen.add(function)

        # [ru] 2. Поиск по синонимам
        # [en] 2. Search by synonyms
        for synonym, function in self.synonym_to_function.items():
            if synonym in text_lower and function not in seen:
                data = self.FUNCTIONAL_MAP[function]
                prop = FunctionalProperty(
                    name=function,
                    category=data['category'],
                    confidence=0.85,
                    source='synonym'
                )
                properties.append(prop)
                seen.add(function)

        # [ru] 3. Поиск по шаблонам
        # [en] 3. Search by patterns
        if use_patterns:
            extracted_phrases = self._extract_by_patterns(text_lower)
            for phrase, pattern_type in extracted_phrases:
                for function, data in self.FUNCTIONAL_MAP.items():
                    if function in seen:
                        continue
                    for keyword in data['keywords']:
                        if keyword in phrase or phrase in keyword:
                            prop = FunctionalProperty(
                                name=function,
                                category=data['category'],
                                confidence=0.7,
                                source=f'pattern_{pattern_type}'
                            )
                            properties.append(prop)
                            seen.add(function)
                            break

        # ============================================================
        # [ru] ШАГ 2: LLM (если свойств мало И включена LLM)
        # [en] STEP 2: LLM (if few properties AND LLM is enabled)
        # ============================================================

        if self.use_llm and len(properties) < 3:
            print(f"[ru]  Извлечено только {len(properties)} свойств, используем LLM...")
            print(f"[en]  Extracted only {len(properties)} properties, we use LLM...")
            llm_props = self._extract_with_llm(text)

            for prop in llm_props:
                if prop.name not in seen:
                    properties.append(prop)
                    seen.add(prop.name)

        # [ru] Сортируем по уверенности
        # [en] Sort by confidence
        properties.sort(key=lambda x: x.confidence, reverse=True)
        return properties

    def _extract_by_patterns(self, text: str) -> List[Tuple[str, str]]:
        """
        [ru] Извлекает фразы по шаблонам.
        [en] Extracts phrases by patterns.
        """
        results = []
        for pattern, pattern_type in self.PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                phrase = match.strip().lower()
                if phrase and len(phrase) > 2:
                    results.append((phrase, pattern_type))
        return results

    def _extract_with_llm(self, text: str) -> List[FunctionalProperty]:
        """
        [ru] Извлекает функциональные свойства с помощью локальной LLM.
        [en] Extracts functional properties using a local LLM.
        """
        # [ru] Проверяем кэш
        # [en] Check cache
        cache_key = text[:100]
        if cache_key in self.llm_cache:
            print(f" [ru] Используем кэш LLM")
            print(f" [en] Use cache of LLM")
            return self.llm_cache[cache_key]

        try:
            prompt = self._build_llm_prompt(text)
            response = self._call_llm(prompt)
            properties = self._parse_llm_response(response)
            self.llm_cache[cache_key] = properties
            return properties
        except Exception as e:
            print(f" [ru] Ошибка LLM: {e}")
            print(f" [en] Error of LLM: {e}")
            return []

    def _build_llm_prompt(self, text: str) -> str:
        """
        [ru] Строит промпт для LLM.
        [en] Builds prompt for LLM.
        """
        return f"""
Ты — инженер-аналитик. Извлеки функциональные свойства из следующей задачи.

Задача: {text}

Функциональные свойства — это действия или способности, которые должно выполнять устройство/система.
Например:
- "создавать подъёмную силу"
- "преобразовывать энергию"
- "передавать движение"
- "управлять полётом"

Верни ТОЛЬКО список функциональных свойств в формате JSON, без пояснений:
["свойство1", "свойство2", ...]

Если свойств нет, верни [].
"""

    def _call_llm(self, prompt: str) -> str:
        """
        [ru] Вызывает локальную LLM через Ollama.
        [en] Calls local LLM via Ollama.
        """
        import requests

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.llm_model,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.1,
                    'max_tokens': 200
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '[]')
        except Exception as e:
            print(f" [ru] Ollama не доступен: {e}")
            print(f" [en] Ollama not available: {e}")

        return '[]'

    def _parse_llm_response(self, response: str) -> List[FunctionalProperty]:
        """
        [ru] Парсит ответ LLM.
        [en] Parses LLM response.
        """
        try:
            response = response.strip()
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                if isinstance(data, list):
                    properties = []
                    for item in data:
                        if isinstance(item, str):
                            category = 'unknown'
                            for func, info in self.FUNCTIONAL_MAP.items():
                                if item in func or func in item:
                                    category = info['category']
                                    break
                            prop = FunctionalProperty(
                                name=item,
                                category=category,
                                confidence=0.8,
                                source='llm'
                            )
                            properties.append(prop)
                    return properties
        except Exception as e:
            print(f" [ru] Ошибка парсинга LLM: {e}")
            print(f" [en] Parce error of LLM: {e}")

        return []

    def extract_as_properties(self, text: str) -> List[str]:
        """
        [ru] Извлекает функциональные свойства как список строк.
        [en] Extracts functional properties as a list of strings.
        """
        props = self.extract(text)
        return [p.name for p in props]

    def get_functional_requirements(self, text: str) -> Dict[str, List[FunctionalProperty]]:
        """
        [ru] Группирует функциональные требования по категориям.
        [en] Groups functional requirements by categories.
        """
        props = self.extract(text)
        grouped = {}
        for prop in props:
            if prop.category not in grouped:
                grouped[prop.category] = []
            grouped[prop.category].append(prop)
        return grouped




# # core/thinking/functional_extractor.py
# """
# Извлечение функциональных свойств из текста задачи.
# ГИБРИДНЫЙ подход: словарь + шаблоны + LLM (как fallback).
# """
#
# import re
# import json
# from typing import List, Dict, Set, Tuple, Optional
# import time
#
#
# class FunctionalProperty:
#     """
#     Функциональное свойство с метаданными.
#     """
#
#     def __init__(self, name: str, category: str, confidence: float = 1.0,
#                  source: str = "extracted", synonyms: List[str] = None):
#         self.name = name
#         self.category = category  # 'physics', 'mechanics', 'control', 'energy', etc.
#         self.confidence = confidence
#         self.source = source
#         self.synonyms = synonyms or []
#
#     def __repr__(self):
#         return f"FunctionalProperty({self.name}, cat={self.category}, conf={self.confidence:.2f})"
#
#
# class FunctionalExtractor:
#     """
#     Извлекает функциональные свойства из текста задачи. ГИБРИДНЫЙ подход: словарь + шаблоны + LLM (как fallback).
#     """
#
#     # ============================================================
#     # БАЗА ЗНАНИЙ: ФУНКЦИОНАЛЬНЫЕ СВОЙСТВА → КЛЮЧЕВЫЕ СЛОВА
#     # ============================================================
#
#     FUNCTIONAL_MAP = {
#         # ============================================================
#         # Аэродинамика и полёт
#         # ============================================================
#         'создавать подъёмную силу': {
#             'keywords': ['подъём', 'подъем', 'лететь', 'взлететь', 'парить',
#                          'подниматься', 'летательный', 'летать', 'воздушный',
#                          'аэродинамический', 'крыло'],
#             'category': 'aerodynamics',
#             'synonyms': ['подъёмная сила', 'lift', 'аэродинамическая сила']
#         },
#
#         'создавать тягу': {
#             'keywords': ['тяга', 'движение', 'разгон', 'ускорение', 'толкать',
#                          'пропеллер', 'винт', 'реактивный', 'двигатель', 'двигателя', 'двигателю', 'двигателем', 'двигателе', 'мотор'],
#             'category': 'propulsion',
#             'synonyms': ['тяговое усилие', 'двигательная сила']
#         },
#
#         'преобразовывать энергию': {
#             'keywords': ['преобразовывать', 'энергия', 'конвертировать', 'тепловой',
#                          'механический', 'электрический', 'солнечный', 'конверсия'],
#             'category': 'energy',
#             'synonyms': ['конверсия энергии', 'трансформация']
#         },
#         'управлять полётом': {
#             'keywords': ['управлять', 'направлять', 'маневрировать', 'контролировать',
#                          'рулить', 'автопилот', 'стабилизация', 'навигация'],
#             'category': 'control',
#             'synonyms': ['навигация', 'управление']
#         },
#
#         # ============================================================
#         # Механика и робототехника
#         # ============================================================
#         'передавать движение': {
#             'keywords': ['передавать', 'вращение', 'крутящий момент', 'трансмиссия',
#                          'редуктор', 'вал', 'привод'],
#             'category': 'mechanics',
#             'synonyms': ['трансмиссия', 'привод']
#         },
#         'поднимать груз': {
#             'keywords': ['поднимать', 'подъем', 'подъём', 'грузить', 'перевозить',
#                          'транспортировать', 'перевозка', 'груз', 'транспортировка',
#                          'доставлять', 'тяжелый', 'грузоподъемность'],
#             'category': 'mechanics',
#             'synonyms': ['грузоподъёмность', 'подъём']
#         },
#         'удерживать нагрузку': {
#             'keywords': ['удерживать', 'фиксировать', 'крепление', 'захват',
#                          'держать', 'стопор', 'зажим'],
#             'category': 'mechanics',
#             'synonyms': ['захват', 'фиксация']
#         },
#         'двигаться': {
#             'keywords': ['двигаться', 'движение', 'перемещаться', 'ход', 'ездить'],
#             'category': 'mechanics',
#             'synonyms': ['передвижение', 'мобильность']
#         },
#         'поворачивать': {
#             'keywords': ['поворачивать', 'поворот', 'вращать', 'маневр'],
#             'category': 'mechanics',
#             'synonyms': ['поворот', 'вращение']
#         },
#
#         # ============================================================
#         # Управление и ИИ
#         # ============================================================
#         'управлять системой': {
#             'keywords': ['управлять', 'управление', 'регулировать', 'контролировать',
#                          'система', 'автоматизация'],
#             'category': 'control',
#             'synonyms': ['управление', 'регулировка']
#         },
#         'обрабатывать данные': {
#             'keywords': ['обрабатывать', 'данные', 'информация', 'сигнал', 'сенсор',
#                          'датчик', 'анализировать', 'измерять'],
#             'category': 'control',
#             'synonyms': ['обработка сигналов', 'анализ данных']
#         },
#         'принимать решения': {
#             'keywords': ['решение', 'алгоритм', 'логика', 'выбор', 'оптимизация',
#                          'планировать', 'рассчитывать', 'интеллект'],
#             'category': 'intelligence',
#             'synonyms': ['принятие решений', 'планирование']
#         },
#         'исполнять команды': {
#             'keywords': ['исполнять', 'команда', 'действие', 'реакция', 'отклик',
#                          'выполнять', 'реализовывать', 'актуатор'],
#             'category': 'control',
#             'synonyms': ['выполнение', 'реализация']
#         },
#         'воспринимать окружение': {
#             'keywords': ['воспринимать', 'сенсор', 'датчик', 'зрение', 'локализация',
#                          'карта', 'обнаружение'],
#             'category': 'perception',
#             'synonyms': ['сенсорика', 'восприятие']
#         },
#
#         # ============================================================
#         # Физика и энергия
#         # ============================================================
#         'создавать давление': {
#             'keywords': ['давление', 'сжатие', 'расширение', 'газ', 'жидкость',
#                          'гидравлика', 'пневматика'],
#             'category': 'physics',
#             'synonyms': ['гидравлика', 'пневматика']
#         },
#         'нагревать или охлаждать': {
#             'keywords': ['нагревать', 'охлаждать', 'температура', 'тепло', 'холод',
#                          'терморегуляция', 'теплообмен'],
#             'category': 'thermodynamics',
#             'synonyms': ['терморегуляция', 'теплообмен']
#         },
#
#         # ============================================================
#         # Логистика и транспорт
#         # ============================================================
#         'перевозить': {
#             'keywords': ['перевозить', 'перевозка', 'транспортировать', 'доставлять',
#                          'груз', 'транспорт', 'логистика'],
#             'category': 'logistics',
#             'synonyms': ['транспортировка', 'грузоперевозки']
#         },
#         'поднимать': {
#             'keywords': ['поднимать', 'подъем', 'подъём', 'лебедка', 'кран', 'такелаж'],
#             'category': 'mechanics',
#             'synonyms': ['подъём', 'поднятие']
#         },
#
#         # ============================================================
#         # Робототехника
#         # ============================================================
#         'двигаться по поверхности': {
#             'keywords': ['ходьба', 'шагать', 'колесо', 'гусеница', 'передвижение'],
#             'category': 'robotics',
#             'synonyms': ['локомоция', 'передвижение']
#         },
#         'манипулировать объектами': {
#             'keywords': ['манипулировать', 'захват', 'рука', 'схватить', 'манипулятор'],
#             'category': 'robotics',
#             'synonyms': ['манипуляция', 'схват']
#         },
#     }
#
#     # ============================================================
#     # ШАБЛОНЫ ДЛЯ ИЗВЛЕЧЕНИЯ
#     # ============================================================
#
#     PATTERNS = [
#         (r'для\s+([а-яА-Яa-zA-Z\s]+)', 'purpose'),
#         (r'чтобы\s+([а-яА-Яa-zA-Z\s]+)', 'purpose'),
#         (r'который\s+([а-яА-Яa-zA-Z]+)', 'verb'),
#         (r'способность\s+([а-яА-Яa-zA-Z]+)', 'ability'),
#         (r'обеспечивать\s+([а-яА-Яa-zA-Z]+)', 'provide'),
#         (r'может\s+([а-яА-Яa-zA-Z]+)', 'can'),
#     ]
#
#     def __init__(self, use_llm: bool = True, llm_model: str = "qwen7B"):
#         """
#         Args:
#             use_llm: Использовать LLM как fallback
#             llm_model: Модель для использования ('llama7B', 'qwen7B', 'mistral7B')
#         """
#         self.use_llm = use_llm
#         self.llm_model = llm_model
#         self._build_indexes()
#
#         # Кэш для результатов LLM
#         self.llm_cache = {}
#
#     def _build_indexes(self):
#         """
#         Строит индексы для быстрого поиска.
#         """
#         self.keyword_to_function = {}
#         self.synonym_to_function = {}
#
#         for function, data in self.FUNCTIONAL_MAP.items():
#             for keyword in data['keywords']:
#                 self.keyword_to_function[keyword] = function
#             for synonym in data.get('synonyms', []):
#                 self.synonym_to_function[synonym] = function
#
#     def extract(self, text: str, use_patterns: bool = True) -> List[FunctionalProperty]:
#         """
#         Извлекает функциональные свойства из текста.
#         ГИБРИДНЫЙ метод: сначала словарь, если мало → LLM.
#         """
#         properties = []
#         seen = set()
#         text_lower = text.lower()
#
#         # ============================================================
#         # ШАГ 1: СЛОВАРЬ + ШАБЛОНЫ (быстрый путь)
#         # ============================================================
#
#         # 1. Поиск по ключевым словам
#         for keyword, function in self.keyword_to_function.items():
#             if keyword in text_lower and function not in seen:
#                 print(f"DEBUG: найдено ключевое слово '{keyword}' → функция '{function}'")
#                 data = self.FUNCTIONAL_MAP[function]
#                 prop = FunctionalProperty(
#                     name=function,
#                     category=data['category'],
#                     confidence=0.9,
#                     source='keyword'
#                 )
#                 properties.append(prop)
#                 seen.add(function)
#
#         # 2. Поиск по синонимам
#         for synonym, function in self.synonym_to_function.items():
#             if synonym in text_lower and function not in seen:
#                 data = self.FUNCTIONAL_MAP[function]
#                 prop = FunctionalProperty(
#                     name=function,
#                     category=data['category'],
#                     confidence=0.85,
#                     source='synonym'
#                 )
#                 properties.append(prop)
#                 seen.add(function)
#
#         # 3. Поиск по шаблонам
#         if use_patterns:
#             extracted_phrases = self._extract_by_patterns(text_lower)
#             for phrase, pattern_type in extracted_phrases:
#                 for function, data in self.FUNCTIONAL_MAP.items():
#                     if function in seen:
#                         continue
#                     for keyword in data['keywords']:
#                         if keyword in phrase or phrase in keyword:
#                             prop = FunctionalProperty(
#                                 name=function,
#                                 category=data['category'],
#                                 confidence=0.7,
#                                 source=f'pattern_{pattern_type}'
#                             )
#                             properties.append(prop)
#                             seen.add(function)
#                             break
#
#         # ============================================================
#         # ШАГ 2: LLM (если свойств мало И включена LLM)
#         # ============================================================
#
#         if self.use_llm and len(properties) < 3:
#             print(f"  Извлечено только {len(properties)} свойств, используем LLM...")
#             print(f"  Извлечено только {len(properties)} свойств, используем LLM...")
#             llm_props = self._extract_with_llm(text)
#
#             for prop in llm_props:
#                 if prop.name not in seen:
#                     properties.append(prop)
#                     seen.add(prop.name)
#
#         # Сортируем по уверенности
#         properties.sort(key=lambda x: x.confidence, reverse=True)
#         return properties
#
#     def _extract_by_patterns(self, text: str) -> List[Tuple[str, str]]:
#         """
#         Извлекает фразы по шаблонам.
#         """
#         results = []
#         for pattern, pattern_type in self.PATTERNS:
#             matches = re.findall(pattern, text)
#             for match in matches:
#                 phrase = match.strip().lower()
#                 if phrase and len(phrase) > 2:
#                     results.append((phrase, pattern_type))
#         return results
#
#     def _extract_with_llm(self, text: str) -> List[FunctionalProperty]:
#         """
#         Извлекает функциональные свойства с помощью локальной LLM.
#         """
#         # Проверяем кэш
#         cache_key = text[:100]
#         if cache_key in self.llm_cache:
#             print(f"   📦 Используем кэш LLM")
#             return self.llm_cache[cache_key]
#
#         try:
#             prompt = self._build_llm_prompt(text)
#             response = self._call_llm(prompt)
#             properties = self._parse_llm_response(response)
#             self.llm_cache[cache_key] = properties
#             return properties
#         except Exception as e:
#             print(f"  Ошибка LLM: {e}")
#             return []
#
#     def _build_llm_prompt(self, text: str) -> str:
#         """
#         Строит промпт для LLM.
#         """
#         return f"""
# Ты — инженер-аналитик. Извлеки функциональные свойства из следующей задачи.
#
# Задача: {text}
#
# Функциональные свойства — это действия или способности, которые должно выполнять устройство/система.
# Например:
# - "создавать подъёмную силу"
# - "преобразовывать энергию"
# - "передавать движение"
# - "управлять полётом"
#
# Верни ТОЛЬКО список функциональных свойств в формате JSON, без пояснений:
# ["свойство1", "свойство2", ...]
#
# Если свойств нет, верни [].
# """
#
#     def _call_llm(self, prompt: str) -> str:
#         """
#         Вызывает локальную LLM через Ollama.
#         """
#         import requests
#
#         try:
#             response = requests.post(
#                 'http://localhost:11434/api/generate',
#                 json={
#                     'model': self.llm_model,
#                     'prompt': prompt,
#                     'stream': False,
#                     'temperature': 0.1,
#                     'max_tokens': 200
#                 },
#                 timeout=30
#             )
#             if response.status_code == 200:
#                 result = response.json()
#                 return result.get('response', '[]')
#         except Exception as e:
#             print(f"  Ollama не доступен: {e}")
#
#         return '[]'
#
#     def _parse_llm_response(self, response: str) -> List[FunctionalProperty]:
#         """
#         Парсит ответ LLM.
#         """
#         try:
#             response = response.strip()
#             import re
#             json_match = re.search(r'\[.*\]', response, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group()
#                 data = json.loads(json_str)
#                 if isinstance(data, list):
#                     properties = []
#                     for item in data:
#                         if isinstance(item, str):
#                             category = 'unknown'
#                             for func, info in self.FUNCTIONAL_MAP.items():
#                                 if item in func or func in item:
#                                     category = info['category']
#                                     break
#                             prop = FunctionalProperty(
#                                 name=item,
#                                 category=category,
#                                 confidence=0.8,
#                                 source='llm'
#                             )
#                             properties.append(prop)
#                     return properties
#         except Exception as e:
#             print(f"   ⚠️ Ошибка парсинга LLM: {e}")
#
#         return []
#
#     def extract_as_properties(self, text: str) -> List[str]:
#         """
#         Извлекает функциональные свойства как список строк.
#         """
#         props = self.extract(text)
#         return [p.name for p in props]
#
#     def get_functional_requirements(self, text: str) -> Dict[str, List[FunctionalProperty]]:
#         """
#         Группирует функциональные требования по категориям.
#         """
#         props = self.extract(text)
#         grouped = {}
#         for prop in props:
#             if prop.category not in grouped:
#                 grouped[prop.category] = []
#             grouped[prop.category].append(prop)
#         return grouped