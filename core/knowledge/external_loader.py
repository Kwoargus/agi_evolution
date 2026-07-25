# core/knowledge/external_loader.py

class ExternalKnowledgeLoader:
    """
    [ru] Загрузчик внешних графов знаний с фильтрацией.
    [en] Loader for external knowledge graphs with filtering.
    """

    def __init__(self):
        # [ru] Белый список типов узлов
        # [en] Whitelist of node types
        self.allowed_types = [
            'object', 'process', 'system', 'device',
            'material', 'concept', 'physical_object'
        ]

        # [ru] Белый список типов отношений
        # [en] Whitelist of relation types
        self.allowed_edges = [
            'PartOf', 'HasProperty', 'UsedFor', 'Causes',
            'CreatedBy', 'HasFunction', 'HasPart', 'IsA'
        ]

    def load_conceptnet(self, limit: int = 10000):
        """
        [ru] Загружает данные из ConceptNet с фильтрацией.
        [en] Loads data from ConceptNet with filtering.
        """
        import requests

        # ConceptNet API
        url = "https://api.conceptnet.io/query"

        # [ru] Запрос только на нужные типы
        # [en] Request only specific types
        params = {
            'rel': ['/r/IsA', '/r/PartOf', '/r/UsedFor', '/r/Causes'],
            'limit': limit,
            'offset': 0
        }

        # [ru] ... загрузка и фильтрация
        # [en] ... loading and filtering
        pass

    def filter_knowledge(self, data):
        """
        [ru] Фильтрует только релевантные знания.
        [en] Filters only relevant knowledge.
        """
        filtered = []
        for item in data:
            # [ru] Проверяем тип
            # [en] Check type
            if self._is_relevant_type(item):
                # [ru] Проверяем связь
                # [en] Check relation
                filtered.append(item)
        return filtered

    def _is_relevant_type(self, item):
        """
        [ru] Проверяет, релевантен ли тип узла.
        [en] Checks if the node type is relevant.
        """
        # [ru] Проверяем по ключевым словам
        # [en] Check by keywords
        relevant_keywords = [
            'device', 'machine', 'tool', 'system',
            'object', 'material', 'process', 'mechanism'
        ]

        # [ru] Упрощенная проверка
        # [en] Simplified check
        text = str(item).lower()
        return any(kw in text for kw in relevant_keywords)

    def _is_relevant_edge(self, item):
        """
        [ru] Проверяет, релевантна ли связь.
        [en] Checks if the relation is relevant.
        """
        relevant_relations = [
            'PartOf', 'HasProperty', 'UsedFor', 'Causes',
            'HasFunction', 'HasPart', 'IsA', 'MadeOf'
        ]

        # [ru] ... проверка ?!!! где код?
        # [en] ... check ?!!! where is the code?
        return True




# # core/knowledge/external_loader.py
#
# class ExternalKnowledgeLoader:
#     """
#     Загрузчик внешних графов знаний с фильтрацией.
#     """
#
#     def __init__(self):
#         # Белый список типов узлов
#         self.allowed_types = [
#             'object', 'process', 'system', 'device',
#             'material', 'concept', 'physical_object'
#         ]
#
#         # Белый список типов отношений
#         self.allowed_edges = [
#             'PartOf', 'HasProperty', 'UsedFor', 'Causes',
#             'CreatedBy', 'HasFunction', 'HasPart', 'IsA'
#         ]
#
#     def load_conceptnet(self, limit: int = 10000):
#         """
#         Загружает данные из ConceptNet с фильтрацией.
#         """
#         import requests
#
#         # ConceptNet API
#         url = "https://api.conceptnet.io/query"
#
#         # Запрос только на нужные типы
#         params = {
#             'rel': ['/r/IsA', '/r/PartOf', '/r/UsedFor', '/r/Causes'],
#             'limit': limit,
#             'offset': 0
#         }
#
#         # ... загрузка и фильтрация
#         pass
#
#     def filter_knowledge(self, data):
#         """
#         Фильтрует только релевантные знания.
#         """
#         filtered = []
#         for item in data:
#             # Проверяем тип
#             if self._is_relevant_type(item):
#                 # Проверяем связь
#                 if self._is_relevant_edge(item):
#                     filtered.append(item)
#         return filtered
#
#     def _is_relevant_type(self, item):
#         """
#         Проверяет, релевантен ли тип узла.
#         """
#         # Проверяем по ключевым словам
#         relevant_keywords = [
#             'device', 'machine', 'tool', 'system',
#             'object', 'material', 'process', 'mechanism'
#         ]
#
#         # Упрощенная проверка
#         text = str(item).lower()
#         return any(kw in text for kw in relevant_keywords)
#
#     def _is_relevant_edge(self, item):
#         """
#         Проверяет, релевантна ли связь.
#         """
#         relevant_relations = [
#             'PartOf', 'HasProperty', 'UsedFor', 'Causes',
#             'HasFunction', 'HasPart', 'IsA', 'MadeOf'
#         ]
#
#         # ... проверка ?!!! где код?
#         return True