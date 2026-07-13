# core/knowledge/function.py
"""
Модуль для работы с функциями моделей знаний.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Function:
    """
    Функция модели знания - описывает, что модель может делать.

    Attributes:
        id: Уникальный идентификатор
        name: Название функции
        description: Описание функциональности
        params: Параметры функции
        return_type: Тип возвращаемого значения
        implementation: Ссылка на реализацию (если есть)
        metadata: Дополнительные метаданные
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
        Выполняет функцию, если есть реализация.
        """
        if self.implementation is not None:
            return self.implementation(*args, **kwargs)
        raise NotImplementedError(f"Function {self.name} has no implementation")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'params': self.params,
            'return_type': self.return_type,
            'metadata': self.metadata
        }