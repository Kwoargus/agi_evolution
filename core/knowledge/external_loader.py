# core/knowledge/external_loader.py

class ExternalKnowledgeLoader:
    """
    Загрузчик внешних графов знаний с фильтрацией.
    """

    def __init__(self):
        # Белый список типов узлов
        self.allowed_types = [
            'object', 'process', 'system', 'device',
            'material', 'concept', 'physical_object'
        ]

        # Белый список типов отношений
        self.allowed_edges = [
            'PartOf', 'HasProperty', 'UsedFor', 'Causes',
            'CreatedBy', 'HasFunction', 'HasPart', 'IsA'
        ]

    def load_conceptnet(self, limit: int = 10000):
        """
        Загружает данные из ConceptNet с фильтрацией.
        """
        import requests

        # ConceptNet API
        url = "https://api.conceptnet.io/query"

        # Запрос только на нужные типы
        params = {
            'rel': ['/r/IsA', '/r/PartOf', '/r/UsedFor', '/r/Causes'],
            'limit': limit,
            'offset': 0
        }

        # ... загрузка и фильтрация
        pass

    def filter_knowledge(self, data):
        """
        Фильтрует только релевантные знания.
        """
        filtered = []
        for item in data:
            # Проверяем тип
            if self._is_relevant_type(item):
                # Проверяем связь
                if self._is_relevant_edge(item):
                    filtered.append(item)
        return filtered

    def _is_relevant_type(self, item):
        """
        Проверяет, релевантен ли тип узла.
        """
        # Проверяем по ключевым словам
        relevant_keywords = [
            'device', 'machine', 'tool', 'system',
            'object', 'material', 'process', 'mechanism'
        ]

        # Упрощенная проверка
        text = str(item).lower()
        return any(kw in text for kw in relevant_keywords)

    def _is_relevant_edge(self, item):
        """
        Проверяет, релевантна ли связь.
        """
        relevant_relations = [
            'PartOf', 'HasProperty', 'UsedFor', 'Causes',
            'HasFunction', 'HasPart', 'IsA', 'MadeOf'
        ]

        # ... проверка
        return True