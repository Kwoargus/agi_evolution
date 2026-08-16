# core/emotions/emotion_system.py
import os
import json
import pickle
import numpy as np
import torch
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from .emotion_base import EmotionalResponse, EmotionType
from typing import List, Dict, Optional
import difflib

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

class EmotionMLP(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class EmotionSystem:
    def __init__(self):
        self.current_emotions = []
        self.emotion_history = []
        self.insight_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Загружаем MLP-классификатор
        self.classifier = None
        self.label_encoder = None
        self._load_classifier()

        # Загружаем эмбеддинги эмоций для fallback (если MLP не загрузится)
        self._load_emotion_embeddings()
        self._load_event_embeddings()

    def _load_classifier(self):
        model_path = os.path.join(os.path.dirname(__file__), "emotion_mlp.pth")
        encoder_path = os.path.join(os.path.dirname(__file__), "emotion_label_encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            print("⚠️ MLP-классификатор не найден, используем fallback.")
            return

        try:
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            num_classes = len(self.label_encoder.classes_)
            self.classifier = EmotionMLP(num_classes=num_classes).to(self.device)
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            self.classifier.eval()
            print(f"✅ MLP-классификатор загружен, {num_classes} классов.")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки MLP-классификатора: {e}")
            self.classifier = None
            self.label_encoder = None

    def _load_emotion_embeddings(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT type, embedding FROM agi_evolution.emotion_respons WHERE embedding IS NOT NULL")
            self.emotion_embeddings = {}
            for row in cur.fetchall():
                emb = np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
                self.emotion_embeddings[row['type']] = emb
            cur.close()
            conn.close()
            print(f"✅ Загружено {len(self.emotion_embeddings)} эмоций с эмбеддингами.")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки эмбеддингов эмоций: {e}")
            self.emotion_embeddings = {}

    def _load_event_embeddings(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT description, embedding FROM agi_evolution.trigger_event WHERE embedding IS NOT NULL")
            self.event_embeddings = {}
            for row in cur.fetchall():
                emb = np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
                self.event_embeddings[row['description']] = emb
            cur.close()
            conn.close()
            print(f"✅ Загружено {len(self.event_embeddings)} событий с эмбеддингами.")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки эмбеддингов событий: {e}")
            self.event_embeddings = {}

    def get_event_embedding_from_db(self, description: str) -> Optional[np.ndarray]:
        if not self.event_embeddings:
            return None
        if description in self.event_embeddings:
            return self.event_embeddings[description]
        # Частичное совпадение
        for key, emb in self.event_embeddings.items():
            if description in key or key in description:
                return emb
        return None

    def predict_emotion(self, event_embedding: np.ndarray) -> Dict:
        # Используем MLP, если он загружен
        if self.classifier is not None and self.label_encoder is not None:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(event_embedding).unsqueeze(0).to(self.device)
                logits = self.classifier(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                emotion_type = self.label_encoder.inverse_transform([pred_idx])[0]
                probability = probs[0, pred_idx].item()
                return {
                    'type': emotion_type,
                    'intensity': probability,
                    'probability': probability,
                    'embedding': None  # не используем эмбеддинг для MLP
                }

        # Fallback: косинусное сходство с эмбеддингами эмоций
        if self.emotion_embeddings:
            best_emotion = None
            best_sim = -1.0
            for em_type, emb in self.emotion_embeddings.items():
                sim = np.dot(event_embedding, emb) / (np.linalg.norm(event_embedding) * np.linalg.norm(emb) + 1e-8)
                if sim > best_sim:
                    best_sim = sim
                    best_emotion = em_type
            if best_emotion:
                return {
                    'type': best_emotion,
                    'intensity': min(1.0, best_sim * 1.2),
                    'probability': best_sim
                }

        return {'type': 'neutral', 'intensity': 0.1, 'probability': 0.5}

    def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
        context = sensory_data.get('context', {})
        event_description = context.get('event_description', '')

        # Пытаемся получить готовый эмбеддинг события из БД
        event_embedding = self.get_event_embedding_from_db(event_description)

        if event_embedding is None:
            # Если не нашли — генерируем из текста
            event_embedding = self._sensory_to_embedding(sensory_data)

        prediction = self.predict_emotion(event_embedding)

        # Маппинг типа эмоции
        emotion_type_str = prediction.get('type', 'neutral')
        try:
            emotion_type = EmotionType[emotion_type_str.upper()]
        except KeyError:
            emotion_type = EmotionType.NEUTRAL

        response = EmotionalResponse(
            emotion_type=emotion_type,
            intensity=prediction.get('intensity', 0.5),
            valence=0.0,
            arousal=0.0,
            embedding=prediction.get('embedding', np.zeros(64))
        )
        self.current_emotions = [response]
        return [response]

    def _sensory_to_embedding(self, sensory_data: Dict) -> np.ndarray:
        # fallback — если не удалось найти в БД
        context = sensory_data.get('context', {})
        nearby = context.get('nearby_object', {})
        other_bots = context.get('other_bots', [])
        bots_info = ", ".join([f"бот на расстоянии {b['distance']:.1f} со здоровьем {b['health']}" for b in other_bots[:3]])
        text = f"Сенсорные данные: рядом объект {nearby}. Другие боты: {bots_info}."
        return self.embedder.encode(text)





# class EmotionSystem:
#     def __init__(self):
#         self.current_emotions = []
#         self.emotion_history = []
#         self.insight_history = []
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
#         self._load_generator()
#         self._load_emotion_embeddings()
#         self._load_event_embeddings()  # новое
#
#     def _load_generator(self):
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(current_dir, "emotion_generator_best.pth")
#         if not os.path.exists(model_path):
#             print(f"⚠️ Модель генератора не найдена по пути: {model_path}")
#             return
#         # Используем ту же архитектуру, что и при обучении (hidden_dim=512)
#         from .train_emotion_gan_with_ga import Generator
#         self.generator = Generator(hidden_dim=512).to(self.device)  # <-- 512
#         self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.generator.eval()
#         print(f"✅ Генератор эмоций загружен из {model_path}")
#
#     # def _load_generator(self):
#     #     current_dir = os.path.dirname(os.path.abspath(__file__))
#     #     model_path = os.path.join(current_dir, "emotion_generator_best.pth")
#     #     if not os.path.exists(model_path):
#     #         print(f"⚠️ Модель генератора не найдена по пути: {model_path}")
#     #         return
#     #     # Импортируем архитектуру
#     #     from .train_emotion_gan_with_ga import Generator
#     #     self.generator = Generator(hidden_dim=128).to(self.device)
#     #     self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
#     #     self.generator.eval()
#     #     print(f"✅ Генератор эмоций загружен из {model_path}")
#
#     def _load_emotion_embeddings(self):
#         try:
#             conn = psycopg2.connect(**DB_CONFIG)
#             cur = conn.cursor(cursor_factory=RealDictCursor)
#             cur.execute("SELECT type, embedding FROM agi_evolution.emotion_respons WHERE embedding IS NOT NULL")
#             self.emotion_embeddings = {}
#             for row in cur.fetchall():
#                 emb = np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
#                 self.emotion_embeddings[row['type']] = emb
#             cur.close()
#             conn.close()
#             print(f"✅ Загружено {len(self.emotion_embeddings)} эмоций с эмбеддингами.")
#         except Exception as e:
#             print(f"⚠️ Ошибка загрузки эмбеддингов эмоций: {e}")
#             self.emotion_embeddings = {}
#
#     def _load_event_embeddings(self):
#         """Загружает эмбеддинги всех событий из БД для поиска по описанию."""
#         try:
#             conn = psycopg2.connect(**DB_CONFIG)
#             cur = conn.cursor(cursor_factory=RealDictCursor)
#             cur.execute("SELECT description, embedding FROM agi_evolution.trigger_event WHERE embedding IS NOT NULL")
#             self.event_embeddings = {}
#             for row in cur.fetchall():
#                 emb = np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
#                 self.event_embeddings[row['description']] = emb
#             cur.close()
#             conn.close()
#             print(f"✅ Загружено {len(self.event_embeddings)} событий с эмбеддингами.")
#         except Exception as e:
#             print(f"⚠️ Ошибка загрузки эмбеддингов событий: {e}")
#             self.event_embeddings = {}
#
#     def get_event_embedding_from_db(self, description: str) -> Optional[np.ndarray]:
#         if not self.event_embeddings:
#             return None
#         # Прямое совпадение
#         if description in self.event_embeddings:
#             return self.event_embeddings[description]
#         # Частичное совпадение (ищем подстроку)
#         for key, emb in self.event_embeddings.items():
#             if description in key or key in description:
#                 return emb
#         # Поиск по похожести (если не нашли)
#         best_match = difflib.get_close_matches(description, self.event_embeddings.keys(), n=1, cutoff=0.6)
#         if best_match:
#             return self.event_embeddings[best_match[0]]
#         return None
#
#     # def get_event_embedding_from_db(self, description: str) -> Optional[np.ndarray]:
#     #     if not self.event_embeddings:
#     #         return None
#     #     # Точное совпадение
#     #     if description in self.event_embeddings:
#     #         return self.event_embeddings[description]
#     #     # Поиск по частичному совпадению (содержит подстроку)
#     #     for key in self.event_embeddings:
#     #         if description in key or key in description:
#     #             print(f"[DEBUG] Найдено частичное совпадение: {key} для {description}")
#     #             return self.event_embeddings[key]
#     #     # Поиск ближайшего совпадения (difflib)
#     #     import difflib
#     #     best_match = difflib.get_close_matches(description, self.event_embeddings.keys(), n=1, cutoff=0.6)
#     #     if best_match:
#     #         print(f"[DEBUG] Найдено близкое совпадение: {best_match[0]} для {description}")
#     #         return self.event_embeddings[best_match[0]]
#     #     return None
#
#     # def get_event_embedding_from_db(self, description: str) -> Optional[np.ndarray]:
#     #     """Возвращает эмбеддинг события по описанию (ближайшее совпадение)."""
#     #     if not self.event_embeddings:
#     #         return None
#     #     # Точное совпадение
#     #     if description in self.event_embeddings:
#     #         return self.event_embeddings[description]
#     #     # Поиск ближайшего совпадения
#     #     best_match = difflib.get_close_matches(description, self.event_embeddings.keys(), n=1, cutoff=0.6)
#     #     if best_match:
#     #         return self.event_embeddings[best_match[0]]
#     #     return None
#
#     def predict_emotion(self, event_embedding: np.ndarray) -> Dict:
#         if self.generator is None or not self.emotion_embeddings:
#             return {'type': 'trust', 'intensity': 0.1, 'probability': 0.5}
#
#         event_tensor = torch.FloatTensor(event_embedding).unsqueeze(0).to(self.device)
#         noise = torch.zeros(1, 64).to(self.device)
#         with torch.no_grad():
#             generated_emb = self.generator(event_tensor, noise).cpu().numpy()[0]
#
#         best_emotion = None
#         best_sim = -1.0
#         for em_type, emb in self.emotion_embeddings.items():
#             sim = np.dot(generated_emb, emb) / (np.linalg.norm(generated_emb) * np.linalg.norm(emb) + 1e-8)
#             if sim > best_sim:
#                 best_sim = sim
#                 best_emotion = em_type
#
#         return {
#             'type': best_emotion or 'trust',
#             'intensity': min(1.0, best_sim * 1.2),
#             'probability': best_sim,
#             'embedding': generated_emb
#         }
#
#     def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
#         context = sensory_data.get('context', {})
#         event_description = context.get('event_description', '')
#         print(f"[DEBUG] event_description: {event_description}")
#
#         # Пытаемся получить готовый эмбеддинг события из БД
#         event_embedding = self.get_event_embedding_from_db(event_description)
#         print(f"[DEBUG] event_embedding found: {event_embedding is not None}")
#
#         if event_embedding is not None:
#             print(f"[DEBUG] event_embedding found: True")
#         else:
#             print(f"[DEBUG] event_embedding found: False, generating from text")
#             event_embedding = self._sensory_to_embedding(sensory_data)
#
#         prediction = self.predict_emotion(event_embedding)
#
#         # if event_embedding is None:
#         #     # Если не нашли — генерируем из текста
#         #     event_embedding = self._sensory_to_embedding(sensory_data)
#         #
#         # prediction = self.predict_emotion(event_embedding)
#
#         if prediction is None:
#             return [EmotionalResponse(emotion_type=EmotionType.TRUST, intensity=0.1, valence=0.0, arousal=0.0, embedding=np.zeros(64))]
#
#         emotion_type = EmotionType(prediction['type']) if prediction['type'] in EmotionType.__members__ else EmotionType.TRUST
#         response = EmotionalResponse(
#             emotion_type=emotion_type,
#             intensity=prediction.get('intensity', 0.5),
#             valence=0.0,
#             arousal=0.0,
#             embedding=prediction.get('embedding', np.zeros(64))
#         )
#         self.current_emotions = [response]
#         return [response]
#
#     def _sensory_to_embedding(self, sensory_data: Dict) -> np.ndarray:
#         # fallback — если не удалось найти в БД
#         context = sensory_data.get('context', {})
#         nearby = context.get('nearby_object', {})
#         text = f"Событие: {json.dumps(nearby, ensure_ascii=False)}"
#         return self.embedder.encode(text)



# # core/emotions/emotion_system.py
# import numpy as np
# import torch
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from sentence_transformers import SentenceTransformer
# from .emotion_engine import EmotionEngine
# from .emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# from typing import List, Dict, Optional
# import json
# import os
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'postgres'
# }
#
# class EmotionSystem:
#     def __init__(self):
#         self.engine = EmotionEngine()
#         self.current_emotions = []
#         self.emotion_history = []
#         self.insight_history = []
#
#         # Загружаем модели GAN
#         self.generator = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self._load_generator()
#
#         # Загружаем эмбеддинги всех эмоций из БД
#         self.emotion_embeddings = {}
#         self._load_emotion_embeddings()
#
#         # Модель для эмбеддингов событий
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
#
#     def get_event_embedding_from_db(self, event_description: str) -> Optional[np.ndarray]:
#         """Возвращает эмбеддинг события из БД по его описанию (точное совпадение)."""
#         conn = psycopg2.connect(**DB_CONFIG)
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute("SELECT embedding FROM agi_evolution.trigger_event WHERE description = %s", (event_description,))
#         row = cur.fetchone()
#         cur.close()
#         conn.close()
#         if row and row['embedding']:
#             return np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
#         return None
#
#     def _load_generator(self):
#         """Загружает обученный генератор из файла."""
#
#         # Определяем директорию, где находится этот файл
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(current_dir, "emotion_generator_best.pth")
#
#         if not os.path.exists(model_path):
#             print(f"⚠️ Модель генератора не найдена по пути: {model_path}")
#             return
#         else:
#             print(f"Модель генератора найдена по пути: {model_path}")
#
#         # model_path = "emotion_generator_best.pth"
#         # if not os.path.exists(model_path):
#         #     print("⚠️ Модель генератора не найдена. Используем rule-based подход.")
#         #     return
#
#         # Импортируем архитектуру генератора из train_emotion_gan_with_ga.py
#         from .train_emotion_gan_with_ga import Generator
#         self.generator = Generator(hidden_dim=128).to(self.device)
#         self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.generator.eval()
#         print("✅ Генератор эмоций загружен.")
#
#     def _load_emotion_embeddings(self):
#         """Загружает эмбеддинги всех эмоций из БД."""
#         try:
#             conn = psycopg2.connect(**DB_CONFIG)
#             cur = conn.cursor(cursor_factory=RealDictCursor)
#             cur.execute("SELECT id, type, embedding FROM agi_evolution.emotion_respons WHERE embedding IS NOT NULL")
#             for row in cur.fetchall():
#                 emb = np.array(row['embedding']) if isinstance(row['embedding'], list) else np.array(json.loads(row['embedding']))
#                 self.emotion_embeddings[row['type']] = emb
#             cur.close()
#             conn.close()
#             print(f"✅ Загружено {len(self.emotion_embeddings)} эмоций с эмбеддингами.")
#         except Exception as e:
#             print(f"⚠️ Ошибка загрузки эмбеддингов эмоций: {e}")
#
#     def predict_emotion(self, event_embedding: np.ndarray) -> Dict:
#         """
#         Использует обученный генератор для предсказания эмоции по эмбеддингу события.
#         Возвращает словарь: {'type': str, 'intensity': float, 'probability': float}
#         """
#         if self.generator is None or not self.emotion_embeddings:
#             # fallback: если модель не загружена, возвращаем доверие (trust) с низкой интенсивностью
#             return {'type': 'trust', 'intensity': 0.1, 'probability': 0.5}
#
#         # Преобразуем в тензор и добавляем шум (нулевой для детерминированности)
#         event_tensor = torch.FloatTensor(event_embedding).unsqueeze(0).to(self.device)
#         noise = torch.zeros(1, 64).to(self.device)  # нулевой шум для стабильности
#
#         with torch.no_grad():
#             generated_emotion_emb = self.generator(event_tensor, noise).cpu().numpy()[0]
#
#         # Находим ближайшую эмоцию по косинусному сходству
#         best_emotion = None
#         best_sim = -1.0
#         for em_type, emb in self.emotion_embeddings.items():
#             sim = np.dot(generated_emotion_emb, emb) / (np.linalg.norm(generated_emotion_emb) * np.linalg.norm(emb) + 1e-8)
#             if sim > best_sim:
#                 best_sim = sim
#                 best_emotion = em_type
#
#         # Интенсивность и вероятность можно получить из свойств эмоции (или задать по умолчанию)
#         # Для простоты используем фиксированные значения
#         return {
#             'type': best_emotion,
#             'intensity': min(1.0, best_sim * 1.2),
#             'probability': best_sim,
#             'embedding': generated_emotion_emb
#         }
#
#     def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
#         """
#         Обрабатывает сенсорные данные и генерирует эмоциональные реакции,
#         используя обученный предиктор (GAN).
#         """
#         # Создаём эмбеддинг события из сенсорных данных
#         event_embedding = self._sensory_to_embedding(sensory_data)
#
#         # Предсказываем эмоцию
#         prediction = self.predict_emotion(event_embedding)
#
#         if prediction is None:
#             # fallback: нейтральная эмоция
#             return [EmotionalResponse(
#                 emotion_type=EmotionType.TRUST,
#                 intensity=0.1,
#                 valence=0.0,
#                 arousal=0.0,
#                 embedding=np.zeros(64)
#             )]
#
#         # Создаём EmotionalResponse на основе предсказания
#         emotion_type = EmotionType(prediction['type']) if prediction['type'] in EmotionType.__members__ else EmotionType.TRUST
#         response = EmotionalResponse(
#             emotion_type=emotion_type,
#             intensity=prediction.get('intensity', 0.5),
#             valence=0.0,  # можно вычислить из свойств эмоции
#             arousal=0.0,
#             embedding=prediction.get('embedding', np.zeros(64))
#         )
#         self.current_emotions = [response]
#         return [response]
#
#     def _sensory_to_embedding(self, sensory_data: Dict) -> np.ndarray:
#         context = sensory_data.get('context', {})
#         nearby = context.get('nearby_object', {})
#         # Получаем описание события из контекста (если есть)
#         event_description = context.get('event_description', 'неизвестное событие')
#         # Формируем текст как при обучении
#         text = f"{event_description} (свойства: {json.dumps(nearby, ensure_ascii=False)})"
#         return self.embedder.encode(text)
#
#     def _sensory_to_embedding_prev(self, sensory_data: Dict) -> np.ndarray:
#         """Преобразует сенсорные данные в эмбеддинг (как в обучении)."""
#         # Формируем текст для эмбеддинга
#         context = sensory_data.get('context', {})
#         nearby = context.get('nearby_object', {})
#         other_bots = context.get('other_bots', [])
#         bots_info = ", ".join([f"бот на расстоянии {b['distance']:.1f} со здоровьем {b['health']}" for b in other_bots[:3]])
#         text = f"Сенсорные данные: рядом объект {nearby}. Другие боты: {bots_info}."
#         return self.embedder.encode(text)
#
#
#
# # core/emotions/emotion_system.py
# # """
# # [ru] Главный модуль эмоциональной системы.
# #
# # Интегрирует все компоненты:
# # - Биграф событий/эмоций
# # - Движок эмоциональных реакций
# # - Ментальные модели
# # - Интуицию
# # - Эволюцию эмоций
# #
# # [en] The main module of the emotional system.
# #
# # Integrates all components:
# # - Event/emotion bigraph
# # - Emotional response engine
# # - Mental models
# # - Intuition
# # - Emotional evolution
# # """
# #
# # import numpy as np
# # from typing import List, Dict, Optional, Tuple, Any, Union
# # from .emotion_engine import EmotionEngine
# # from .mental_model import MentalModelManager
# # from core.emotions.emotion_base import MentalModel
# # from .intuition_engine import IntuitionEngine
# # from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# # from core.emotions.emotion_graph import EmotionGraph
# #
# #
# # class EmotionSystem:
# #     """
# #     [ru] Полная эмоциональная подсистема AGI.
# #     Интеграция всех компонентов эмоционального восприятия и реагирования.
# #
# #     [en]AGI's complete emotional subsystem.
# #     Integration of all components of emotional perception and response.
# #     """
# #
# #     def __init__(self):
# #         # Основные компоненты
# #         self.engine = EmotionEngine()
# #         self.models = MentalModelManager()
# #         self.intuition = IntuitionEngine(self.engine.graph)
# #
# #         # Состояние системы
# #         self.current_emotions: List[EmotionalResponse] = []
# #         self.emotion_history: List[Dict] = []
# #         self.insight_history: List[Dict] = []
# #
# #         # Связь с другими подсистемами
# #         self.reflex_system = None  # Будет подключен позже
# #         self.instinct_system = None  # Будет подключен позже
# #
# #         print("[ru] Полная эмоциональная подсистема инициализирована")
# #         print("[en] The complete emotional subsystem is initialized")
# #
# #     def synchronize_with_knowledge_graph(self, model_id: str) -> bool:
# #         """
# #         [ru] Синхронизирует ментальную модель с глобальным графом знаний.
# #         [en] Synchronizes the mental model with the global knowledge graph.
# #         """
# #         model = self.models.get_model(model_id)
# #         if not model:
# #             return False
# #
# #         try:
# #             from core.knowledge.knowledge_node import KnowledgeNode
# #             from db.knowledge_db import KnowledgeDB
# #
# #             db = KnowledgeDB()
# #
# #             # Создаём узел в ГЗ
# #             # [ru]
# #             # [en]
# #             node = KnowledgeNode(
# #                 id=model.id,
# #                 name=model.name,
# #                 node_type="mental_model",
# #                 properties=list(model.attributes.keys()),
# #                 description=f"Ментальная модель: {model.name} (тип: {model.type})"
# #             )
# #
# #             # [ru] Добавляем эмбеддинг
# #             # [en] Adding embedding
# #             if hasattr(model, 'embedding') and model.embedding is not None:
# #                 node.embedding = model.embedding
# #
# #             # [ru] Сохраняем
# #             # [en] Save
# #             db.save_node(node)
# #             print(f"✅ Модель {model.name} синхронизирована с ГЗ")
# #             print(f"✅ Model {model.name} synchronized with the KG")
# #             return True
# #
# #         except Exception as e:
# #             print(f"Ошибка синхронизации модели {model.name}: {e}")
# #             print(f"Model synchronization error {model.name}: {e}")
# #             return False
# #
# #     def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
# #         """
# #         [ru] Обрабатывает сенсорные данные и генерирует эмоциональные реакции.
# #         [en] Processes sensory data and generates emotional responses.
# #         """
# #         # [ru] 1. Создаем событие из сенсорных данных
# #         # [en] 1. Create an event from sensor data
# #         event = self._sensory_to_event(sensory_data)
# #
# #         # [ru] 2. Генерируем эмоциональные реакции
# #         # [en] 2. Generate emotional reactions
# #         responses = self.engine.process_event(event)
# #
# #         # [ru] 3. Обновляем текущее состояние
# #         # [en] 3. Update the current state
# #         self.current_emotions = responses
# #
# #         # [ru] 4. Сохраняем в историю
# #         # [en] 4. Save to history
# #         self.emotion_history.append({
# #             'timestamp': event.timestamp,
# #             'event': event,
# #             'responses': responses
# #         })
# #
# #         # [ru] 5. Проверяем интуитивные инсайты (с обработкой ошибок)
# #         # [en] 5. Testing intuitive insights (with error handling)
# #         try:
# #             insight = self.intuition.get_insight(event)
# #             if insight and insight.get('confidence', 0) > 0.7:
# #                 self.insight_history.append(insight)
# #                 print(f"[ru] Интуитивный инсайт: {insight.get('explanation', '')}")
# #                 print(f"[en] Intuitive insight: {insight.get('explanation', '')}")
# #         except Exception as e:
# #             # [ru] Игнорируем ошибки интуиции, чтобы не прерывать основной процесс
# #             # [en] Ignore intuition errors to avoid interrupting the main process
# #             pass
# #
# #         return responses
# #
# #
# #     def get_emotional_state(self) -> Dict:
# #         """
# #         [ru] Возвращает текущее эмоциональное состояние.
# #         [en] Returns the current emotional state.
# #         """
# #         if not self.current_emotions:
# #             return {'state': 'neutral', 'intensity': 0.0}
# #
# #         # [ru] Определяем доминирующую эмоцию
# #         # [en] Identifying the dominant emotion
# #         dominant = max(self.current_emotions, key=lambda x: x.intensity)
# #
# #         return {
# #             'dominant_emotion': dominant.emotion_type.value,
# #             'intensity': dominant.intensity,
# #             'valence': dominant.valence,
# #             'arousal': dominant.arousal,
# #             'all_emotions': [{
# #                 'type': e.emotion_type.value,
# #                 'intensity': e.intensity,
# #                 'valence': e.valence
# #             } for e in self.current_emotions]
# #         }
# #
# #     def trace_emotional_chain(self, depth: int = 10) -> List[Dict]:
# #         """
# #         [ru] Трассирует цепочку эмоциональных реакций.
# #         [en] Traces the chain of emotional reactions.
# #         """
# #         if not self.emotion_history:
# #             return []
# #
# #         # Начинаем с последней реакции
# #         # [ru]
# #         # [en]
# #         last_response = self.emotion_history[-1]['responses'][0]
# #         chain = self.engine.trace_response_chain(last_response)
# #
# #         return chain[:depth]
# #
# #     def predict_emotional_development(self, emotion_type: EmotionType,
# #                                       max_depth: int = 5) -> List[List[str]]:
# #         """
# #         [ru] Прогнозирует развитие эмоциональной цепочки.
# #         [en] Predicts the development of the emotional chain.
# #         """
# #         return self.engine.predict_emotion_chain(emotion_type, max_depth)
# #
# #     def compare_mental_models(self, model1_name: str, model2_name: str) -> Dict:
# #         """
# #         [ru] Сравнивает две ментальные модели.
# #         [en] Compares two mental models.
# #
# #         """
# #         # [ru] Находим модели по имени
# #         # [en] Find models by name
# #         model1 = None
# #         model2 = None
# #
# #         for m in self.models.models.values():
# #             if m.name == model1_name:
# #                 model1 = m
# #             if m.name == model2_name:
# #                 model2 = m
# #
# #         if not model1 or not model2:
# #             return {'error': 'Model not found'}
# #
# #         return self.models.compare_models(model1.id, model2.id)
# #
# #     def synchronize_mental_models(self, model1_name: str, model2_name: str):
# #         """
# #         [ru] Синхронизирует две ментальные модели.
# #         [en] Synchronizes two mental models.
# #         """
# #
# #         # [ru] Находим модели по имени
# #         # [en] Find models by name
# #         model1 = None
# #         model2 = None
# #
# #         for m in self.models.models.values():
# #             if m.name == model1_name:
# #                 model1 = m
# #             if m.name == model2_name:
# #                 model2 = m
# #
# #         if not model1 or not model2:
# #             print(f"[ru] Модели не найдены: {model1_name}, {model2_name}")
# #             print(f"[en] No models found: {model1_name}, {model2_name}")
# #             return None
# #
# #         return self.models.synchronize_models(model1.id, model2.id)
# #
# #     def _sensory_to_event(self, sensory_data: Dict) -> EmotionalEvent:
# #         """
# #         [ru] Преобразует сенсорные данные в событие.
# #         [en] Converts sensor data into an event.
# #         """
# #         import time
# #
# #         # Создаём эмбеддинг из сенсорных данных
# #         embedding = self._sensory_to_embedding(sensory_data)
# #
# #         # Проверяем, есть ли раненый бот в контексте
# #         context = sensory_data.get('context', {})
# #         other_bots = context.get('other_bots', [])
# #         injured_bot_detected = False
# #         for bot_info in other_bots:
# #             if bot_info.get('health', 100) < 30:
# #                 injured_bot_detected = True
# #                 break
# #
# #         # Если есть раненый бот, используем специальный ID
# #         event_id = f"event_{len(self.emotion_history)}"
# #         if injured_bot_detected:
# #             event_id = 'bot_injured'
# #             print(f"[ru] Обнаружен раненый бот! Событие: bot_injured")
# #             print(f"[en] Injured bot detected! Event: bot_injured")
# #
# #         event = EmotionalEvent(
# #             id=event_id,
# #             description=self._describe_sensory(sensory_data),
# #             timestamp=time.time(),
# #             context=sensory_data.get('context', {}),
# #             participants=sensory_data.get('participants', []),
# #             embedding=embedding
# #         )
# #
# #         return event
# #
# #
# #     # def _sensory_to_event(self, sensory_data: Dict) -> EmotionalEvent:
# #     #     """
# #     #     [ru] Преобразует сенсорные данные в событие.
# #     #     [en] Converts sensor data into an event.
# #     #
# #     #     """
# #     #     import time
# #     #
# #     #     # [ru] Создаем эмбеддинг из сенсорных данных
# #     #     # [en] Creating an embedding from sensory data
# #     #     embedding = self._sensory_to_embedding(sensory_data)
# #     #
# #     #     event = EmotionalEvent(
# #     #         id=f"event_{len(self.emotion_history)}",
# #     #         description=self._describe_sensory(sensory_data),
# #     #         timestamp=time.time(),
# #     #         context=sensory_data.get('context', {}),
# #     #         participants=sensory_data.get('participants', []),
# #     #         embedding=embedding
# #     #     )
# #     #
# #     #     return event
# #
# #     def _sensory_to_embedding(self, sensory_data: Dict) -> np.ndarray:
# #         """
# #         [ru] Преобразует сенсорные данные в эмбеддинг.
# #         [en] Converts sensory data into embedding.
# #         """
# #         embedding = np.zeros(128)
# #
# #         # [ru] Зрение
# #         # [en] Vision
# #         if 'vision' in sensory_data:
# #             vision = sensory_data['vision']
# #             if len(vision) >= 64:
# #                 embedding[:64] = vision[:64]
# #             else:
# #                 embedding[:len(vision)] = vision
# #
# #         # [ru] Слух
# #         # [en] Sound
# #         if 'sound' in sensory_data:
# #             sound = sensory_data['sound']
# #             if len(sound) >= 32:
# #                 embedding[64:96] = sound[:32]
# #             else:
# #                 embedding[64:64 + len(sound)] = sound
# #
# #         # [ru] Запах
# #         # [en] Smell
# #         if 'smell' in sensory_data:
# #             smell = sensory_data['smell']
# #             if len(smell) >= 32:
# #                 embedding[96:128] = smell[:32]
# #             else:
# #                 embedding[96:96 + len(smell)] = smell
# #
# #         # [ru] Нормализуем
# #         # [en] Let's normalize
# #         norm = np.linalg.norm(embedding) + 1e-8
# #         return embedding / norm
# #
# #     def _describe_sensory(self, sensory_data: Dict) -> str:
# #         """
# #         [ru] Генерирует текстовое описание сенсорных данных.
# #         [en] Generates a text description of sensory data.
# #         """
# #         parts = []
# #         if 'vision' in sensory_data:
# #             parts.append("визуальный стимул")
# #         if 'sound' in sensory_data:
# #             parts.append("звуковой стимул")
# #         if 'smell' in sensory_data:
# #             parts.append("запаховой стимул")
# #
# #         return f"Событие: {', '.join(parts)}"
# #
# #     def connect_systems(self, reflex_system, instinct_system):
# #         """
# #         [ru] Подключает эмоциональную систему к рефлексам и инстинктам.
# #         [en] Connects the emotional system to reflexes and instincts.
# #         """
# #         self.reflex_system = reflex_system
# #         self.instinct_system = instinct_system
# #         print("[ru] Эмоциональная система подключена к рефлексам и инстинктам")
# #         print("[en] The emotional system is connected to reflexes and instincts.")
# #
# #     def influence_reflexes(self):
# #         """
# #         [ru] Эмоции влияют на рефлексы. Например: страх усиливает рефлекс убегания.
# #         [en] Emotions influence reflexes. For example, fear increases the flight reflex.
# #         """
# #         if not self.reflex_system:
# #             return
# #
# #         state = self.get_emotional_state()
# #
# #         # [ru] Пример: страх усиливает рефлексы
# #         # [en] Example: fear enhances reflexes
# #         if state['dominant_emotion'] == EmotionType.FEAR.value:
# #             self.reflex_system.boost_reflex('run_away', state['intensity'])
# #
# #         # [ru] Гнев усиливает агрессивные рефлексы
# #         # [en] Anger increases aggressive reflexes
# #         if state['dominant_emotion'] == EmotionType.ANGER.value:
# #             self.reflex_system.boost_reflex('attack', state['intensity'])
# #
# #     def influence_instincts(self):
# #         """
# #         [ru] Эмоции влияют на инстинкты. Например: страх может подавлять инстинкт исследования.
# #         [en] Emotions influence instincts. For example, fear can suppress the instinct to explore.
# #         """
# #         if not self.instinct_system:
# #             return
# #
# #         state = self.get_emotional_state()
# #
# #         # [ru] Пример: страх подавляет инстинкт исследования
# #         # [en] Example: fear suppresses the instinct of exploration
# #         if state['dominant_emotion'] == EmotionType.FEAR.value:
# #             self.instinct_system.suppress_instinct('explore', state['intensity'])
# #
# #         # [ru] Радость усиливает инстинкт исследования
# #         # [en] Joy strengthens the instinct of exploration
# #         if state['dominant_emotion'] == EmotionType.JOY.value:
# #             self.instinct_system.boost_instinct('explore', state['intensity'])