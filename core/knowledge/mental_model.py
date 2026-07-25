# core/knowledge/mental_model.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class MentalModel:
    """
    [ru] Ментальная модель - визуально-образное представление знания. В текущей реализации - это последовательность состояний (кадров).
    [en] Mental model - a visual-imagery representation of knowledge. In the current implementation, it is a sequence of states (frames).
    """

    def __init__(self, id: str, name: str, sequence: List[np.ndarray]):
        self.id = id
        self.name = name
        self.sequence = sequence  # [ru] Список тензоров (кадров)  [en] List of tensors (frames)
        self.embedding = self._encode_sequence(sequence)

        # [ru] Свойства, выведенные из опыта
        # [en] Properties inferred from experience
        self.properties: List[str] = []

    def _encode_sequence(self, sequence: List[np.ndarray]) -> np.ndarray:
        """
        [ru] Кодирует последовательность в вектор.
        [en] Encodes the sequence into a vector.
        """
        if not sequence:
            return np.zeros(128)
        # [ru] Усредняем кадры
        # [en] Average the frames
        avg = np.mean(sequence, axis=0)
        # [ru] Нормализуем
        # [en] Normalize
        return avg / (np.linalg.norm(avg) + 1e-8)



# # core/knowledge/mental_model.py
# from typing import List, Dict, Any, Tuple, Optional
# import numpy as np
#
# class MentalModel:
#     """
#     [ru] Ментальная модель - визуально-образное представление знания.
#     [en] Mental model - a visual-imagery representation of knowledge.
#     [ru] В текущей реализации - это последовательность состояний (кадров).
#     [en] In the current implementation, it is a sequence of states (frames).
#     """
#
#     def __init__(self, id: str, name: str, sequence: List[np.ndarray]):
#         self.id = id
#         self.name = name
#         self.sequence = sequence  # [ru] Список тензоров (кадров)  [en] List of tensors (frames)
#         self.embedding = self._encode_sequence(sequence)
#
#         # [ru] Свойства, выведенные из опыта
#         # [en] Properties inferred from experience
#         self.properties: List[str] = []
#
#     def _encode_sequence(self, sequence: List[np.ndarray]) -> np.ndarray:
#         """
#         [ru] Кодирует последовательность в вектор.
#         [en] Encodes the sequence into a vector.
#         """
#         if not sequence:
#             return np.zeros(128)
#         # [ru] Усредняем кадры
#         # [en] Average the frames
#         avg = np.mean(sequence, axis=0)
#         # [ru] Нормализуем
#         # [en] Normalize
#         return avg / (np.linalg.norm(avg) + 1e-8)
#
#
#
# # # core/knowledge/mental_model.py
# # from typing import List, Dict, Any, Tuple, Optional
# # import numpy as np
# #
# # class MentalModel:
# #     """
# #     Ментальная модель - визуально-образное представление знания.
# #     В текущей реализации - это последовательность состояний (кадров).
# #     """
# #
# #     def __init__(self, id: str, name: str, sequence: List[np.ndarray]):
# #         self.id = id
# #         self.name = name
# #         self.sequence = sequence  # Список тензоров (кадров)
# #         self.embedding = self._encode_sequence(sequence)
# #
# #         # Свойства, выведенные из опыта
# #         self.properties: List[str] = []
# #
# #     def _encode_sequence(self, sequence: List[np.ndarray]) -> np.ndarray:
# #         """
# #         Кодирует последовательность в вектор.
# #         """
# #         if not sequence:
# #             return np.zeros(128)
# #         # Усредняем кадры
# #         avg = np.mean(sequence, axis=0)
# #         # Нормализуем
# #         return avg / (np.linalg.norm(avg) + 1e-8)