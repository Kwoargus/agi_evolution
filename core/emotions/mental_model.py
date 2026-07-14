# core/emotions/mental_model.py
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from .emotion_base import MentalModel



class MentalModelManager:
    """
    Управление ментальными моделями.

    Функции:
    1. Создание и хранение моделей
    2. Сравнение моделей (синхронизация)
    3. Обновление моделей на основе опыта
    4. Прогнозирование на основе моделей
    """

    def __init__(self):
        self.models: Dict[str, MentalModel] = {}
        self.model_embeddings: List[np.ndarray] = []

        # База знаний
        self.knowledge_base = {}

        print("✅ Менеджер ментальных моделей инициализирован")

    def create_model(self, name: str, model_type: str,
                     attributes: Dict[str, float]) -> MentalModel:
        """Создает новую ментальную модель."""
        model_id = f"{model_type}_{name}_{len(self.models)}"

        # Создаем эмбеддинг модели
        embedding = self._model_to_embedding(name, model_type, attributes)

        model = MentalModel(
            id=model_id,
            name=name,
            type=model_type,
            embedding=embedding,
            attributes=attributes,
            related_models=[],
            predictions=[]
        )

        self.models[model_id] = model
        self.model_embeddings.append(embedding)

        print(f"✅ Создана модель: {name} ({model_type})")
        return model

    def compare_models(self, model1_id: str, model2_id: str) -> Dict:
        """
        Сравнивает две ментальные модели.

        Returns:
            dict: {
                'similarity': float,
                'differences': dict,
                'synchronization_score': float
            }
        """
        model1 = self.models.get(model1_id)
        model2 = self.models.get(model2_id)

        if not model1 or not model2:
            return {'error': 'Model not found'}

        # Косинусное сходство эмбеддингов
        sim = np.dot(model1.embedding, model2.embedding) / (
                np.linalg.norm(model1.embedding) * np.linalg.norm(model2.embedding)
        )

        # Сравнение атрибутов
        differences = {}
        all_keys = set(model1.attributes.keys()) | set(model2.attributes.keys())
        for key in all_keys:
            val1 = model1.attributes.get(key)
            val2 = model2.attributes.get(key)
            if val1 is not None and val2 is not None and val1 != val2:
                differences[key] = (val1, val2)

        # Оценка синхронизации
        sync_score = sim * (1.0 - len(differences) / (len(all_keys) + 1))

        return {
            'similarity': sim,
            'differences': differences,
            'synchronization_score': sync_score,
            'is_synchronized': sync_score > 0.7
        }

    def synchronize_models(self, model1_id: str, model2_id: str) -> MentalModel:
        """
        Синхронизирует две ментальные модели.
        Создает третью модель, объединяющую обе.
        """
        model1 = self.models.get(model1_id)
        model2 = self.models.get(model2_id)

        if not model1 or not model2:
            return None

        # Создаем объединенную модель
        merged_attributes = {}
        for key in set(model1.attributes.keys()) | set(model2.attributes.keys()):
            val1 = model1.attributes.get(key)
            val2 = model2.attributes.get(key)
            if val1 is not None and val2 is not None:
                # Усредняем с учетом весов
                merged_attributes[key] = (val1 + val2) / 2
            elif val1 is not None:
                merged_attributes[key] = val1
            else:
                merged_attributes[key] = val2

        merged_model = self.create_model(
            name=f"synchronized_{model1.name}_{model2.name}",
            model_type="merged",
            attributes=merged_attributes
        )

        merged_model.related_models = [model1_id, model2_id]

        print(f"✅ Создана синхронизированная модель: {merged_model.name}")
        return merged_model

    def update_model_from_experience(self, model_id: str,
                                     experience: Dict) -> MentalModel:
        """
        Обновляет модель на основе опыта.
        """
        model = self.models.get(model_id)
        if not model:
            return None

        # Обновляем атрибуты
        for key, value in experience.get('attributes', {}).items():
            if key in model.attributes:
                # Экспоненциальное скользящее среднее
                alpha = 0.3
                model.attributes[key] = alpha * value + (1 - alpha) * model.attributes[key]

        # Обновляем предсказания
        if 'new_prediction' in experience:
            model.predictions.append(experience['new_prediction'])
            if len(model.predictions) > 100:
                model.predictions = model.predictions[-100:]

        # Обновляем эмбеддинг
        model.embedding = self._model_to_embedding(
            model.name, model.type, model.attributes
        )

        print(f"✅ Модель обновлена: {model.name}")
        return model

    def predict_with_model(self, model_id: str,
                           context: Dict) -> Dict:
        """
        Делает прогноз на основе ментальной модели.
        """
        model = self.models.get(model_id)
        if not model:
            return {'error': 'Model not found'}

        # Простейший прогноз: среднее по атрибутам
        predictions = {}
        for key, value in model.attributes.items():
            # Добавляем небольшой шум для разнообразия
            predictions[key] = value + np.random.randn() * 0.1

        return predictions

    def _model_to_embedding(self, name: str, model_type: str,
                            attributes: Dict[str, float]) -> np.ndarray:
        """Преобразует модель в векторное представление."""
        embedding = np.zeros(128)

        # Кодируем тип модели
        type_idx = hash(model_type) % 8
        embedding[type_idx] = 1.0

        # Кодируем имя
        name_hash = hash(name) % 32
        embedding[8:40] = np.random.randn(32) * 0.1 + name_hash / 32.0

        # Кодируем атрибуты
        for i, (key, value) in enumerate(attributes.items()):
            idx = 40 + (i % 20)
            embedding[idx] = value

        # Нормализуем
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm





# import numpy as np
# from typing import List, Dict, Optional, Tuple, Any, Union
# from .emotion_base import MentalModel
#
#
#
# class MentalModelManager:
#     """
#     Управление ментальными моделями.
#
#     Функции:
#     1. Создание и хранение моделей
#     2. Сравнение моделей (синхронизация)
#     3. Обновление моделей на основе опыта
#     4. Прогнозирование на основе моделей
#     """
#
#     def __init__(self):
#         self.models: Dict[str, MentalModel] = {}
#         self.model_embeddings: List[np.ndarray] = []
#
#         # База знаний
#         self.knowledge_base = {}
#
#         print("✅ Менеджер ментальных моделей инициализирован")
#
#     def create_model(self, name: str, model_type: str,
#                      attributes: Dict[str, float]) -> MentalModel:
#         """Создает новую ментальную модель."""
#         model_id = f"{model_type}_{name}_{len(self.models)}"
#
#         # Создаем эмбеддинг модели
#         embedding = self._model_to_embedding(name, model_type, attributes)
#
#         model = MentalModel(
#             id=model_id,
#             name=name,
#             type=model_type,
#             embedding=embedding,
#             attributes=attributes,
#             related_models=[],
#             predictions=[]
#         )
#
#         self.models[model_id] = model
#         self.model_embeddings.append(embedding)
#
#         print(f"✅ Создана модель: {name} ({model_type})")
#         return model
#
#     def compare_models(self, model1_id: str, model2_id: str) -> Dict:
#         """
#         Сравнивает две ментальные модели.
#
#         Returns:
#             dict: {
#                 'similarity': float,
#                 'differences': dict,
#                 'synchronization_score': float
#             }
#         """
#         model1 = self.models.get(model1_id)
#         model2 = self.models.get(model2_id)
#
#         if not model1 or not model2:
#             return {'error': 'Model not found'}
#
#         # Косинусное сходство эмбеддингов
#         sim = np.dot(model1.embedding, model2.embedding) / (
#                 np.linalg.norm(model1.embedding) * np.linalg.norm(model2.embedding)
#         )
#
#         # Сравнение атрибутов
#         differences = {}
#         all_keys = set(model1.attributes.keys()) | set(model2.attributes.keys())
#         for key in all_keys:
#             val1 = model1.attributes.get(key)
#             val2 = model2.attributes.get(key)
#             if val1 is not None and val2 is not None and val1 != val2:
#                 differences[key] = (val1, val2)
#
#         # Оценка синхронизации
#         sync_score = sim * (1.0 - len(differences) / (len(all_keys) + 1))
#
#         return {
#             'similarity': sim,
#             'differences': differences,
#             'synchronization_score': sync_score,
#             'is_synchronized': sync_score > 0.7
#         }
#
#     def synchronize_models(self, model1_id: str, model2_id: str) -> MentalModel:
#         """
#         Синхронизирует две ментальные модели.
#         Создает третью модель, объединяющую обе.
#         """
#         model1 = self.models.get(model1_id)
#         model2 = self.models.get(model2_id)
#
#         if not model1 or not model2:
#             return None
#
#         # Создаем объединенную модель
#         merged_attributes = {}
#         for key in set(model1.attributes.keys()) | set(model2.attributes.keys()):
#             val1 = model1.attributes.get(key)
#             val2 = model2.attributes.get(key)
#             if val1 is not None and val2 is not None:
#                 # Усредняем с учетом весов
#                 merged_attributes[key] = (val1 + val2) / 2
#             elif val1 is not None:
#                 merged_attributes[key] = val1
#             else:
#                 merged_attributes[key] = val2
#
#         merged_model = self.create_model(
#             name=f"synchronized_{model1.name}_{model2.name}",
#             model_type="merged",
#             attributes=merged_attributes
#         )
#
#         merged_model.related_models = [model1_id, model2_id]
#
#         print(f"✅ Создана синхронизированная модель: {merged_model.name}")
#         return merged_model
#
#     def update_model_from_experience(self, model_id: str,
#                                      experience: Dict) -> MentalModel:
#         """
#         Обновляет модель на основе опыта.
#         """
#         model = self.models.get(model_id)
#         if not model:
#             return None
#
#         # Обновляем атрибуты
#         for key, value in experience.get('attributes', {}).items():
#             if key in model.attributes:
#                 # Экспоненциальное скользящее среднее
#                 alpha = 0.3
#                 model.attributes[key] = alpha * value + (1 - alpha) * model.attributes[key]
#
#         # Обновляем предсказания
#         if 'new_prediction' in experience:
#             model.predictions.append(experience['new_prediction'])
#             if len(model.predictions) > 100:
#                 model.predictions = model.predictions[-100:]
#
#         # Обновляем эмбеддинг
#         model.embedding = self._model_to_embedding(
#             model.name, model.type, model.attributes
#         )
#
#         print(f"✅ Модель обновлена: {model.name}")
#         return model
#
#     def predict_with_model(self, model_id: str,
#                            context: Dict) -> Dict:
#         """
#         Делает прогноз на основе ментальной модели.
#         """
#         model = self.models.get(model_id)
#         if not model:
#             return {'error': 'Model not found'}
#
#         # Простейший прогноз: среднее по атрибутам
#         predictions = {}
#         for key, value in model.attributes.items():
#             # Добавляем небольшой шум для разнообразия
#             predictions[key] = value + np.random.randn() * 0.1
#
#         return predictions
#
#     def _model_to_embedding(self, name: str, model_type: str,
#                             attributes: Dict[str, float]) -> np.ndarray:
#         """Преобразует модель в векторное представление."""
#         embedding = np.zeros(128)
#
#         # Кодируем тип модели
#         type_idx = hash(model_type) % 8
#         embedding[type_idx] = 1.0
#
#         # Кодируем имя
#         name_hash = hash(name) % 32
#         embedding[8:40] = np.random.randn(32) * 0.1 + name_hash / 32.0
#
#         # Кодируем атрибуты
#         for i, (key, value) in enumerate(attributes.items()):
#             idx = 40 + (i % 20)
#             embedding[idx] = value
#
#         # Нормализуем
#         norm = np.linalg.norm(embedding) + 1e-8
#         return embedding / norm