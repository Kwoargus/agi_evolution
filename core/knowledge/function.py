# core/knowledge/function.py
"""
[ru] Модуль для работы с функциями моделей знаний.
[en] Module for working with knowledge model functions.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Function:
    """
    [ru] Функция модели знания - описывает, что модель может делать.
    [ru] Attributes:
        [ru] id: Уникальный идентификатор
        [ru] name: Название функции
        [ru] description: Описание функциональности
        [ru] params: Параметры функции
        [ru] return_type: Тип возвращаемого значения
        [ru] implementation: Ссылка на реализацию (если есть)
        [ru] metadata: Дополнительные метаданные

    [en] Knowledge model function - describes what the model can do.
    [en] Attributes:
        [en] id: Unique identifier
        [en] name: Function name
        [en] description: Description of functionality
        [en] params: Function parameters
        [en] return_type: Return value type
        [en] implementation: Reference to implementation (if any)
        [en] metadata: Additional metadata
    """

    id: str
    name: str
    description: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "void"
    implementation: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, *args, **kwargs) -> Any:
        """
        [ru] Выполняет функцию, если есть реализация.
        [en] Executes the function if an implementation is provided.
        """
        if self.implementation is not None:
            return self.implementation(*args, **kwargs)
        raise NotImplementedError(f"Function {self.name} has no implementation")

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь.
        [en] Converts to a dictionary.
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'params': self.params,
            'return_type': self.return_type,
            'metadata': self.metadata
        }



# # core/knowledge/function.py
# """
# Модуль для работы с функциями моделей знаний.
# """
#
# from typing import Any, Callable, Dict, List, Optional
# from dataclasses import dataclass, field
#
#
# @dataclass
# class Function:
#     """
#     Функция модели знания - описывает, что модель может делать.
#
#     Attributes:
#         id: Уникальный идентификатор
#         name: Название функции
#         description: Описание функциональности
#         params: Параметры функции
#         return_type: Тип возвращаемого значения
#         implementation: Ссылка на реализацию (если есть)
#         metadata: Дополнительные метаданные
#     """
#
#     id: str
#     name: str
#     description: str = ""
#     params: List[Dict[str, Any]] = field(default_factory=list)
#     return_type: str = "void"
#     implementation: Optional[Callable] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)
#
#     def execute(self, *args, **kwargs) -> Any:
#         """
#         Выполняет функцию, если есть реализация.
#         """
#         if self.implementation is not None:
#             return self.implementation(*args, **kwargs)
#         raise NotImplementedError(f"Function {self.name} has no implementation")
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Преобразует в словарь."""
#         return {
#             'id': self.id,
#             'name': self.name,
#             'description': self.description,
#             'params': self.params,
#             'return_type': self.return_type,
#             'metadata': self.metadata
#         }