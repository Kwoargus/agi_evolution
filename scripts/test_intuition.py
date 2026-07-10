# scripts/test_intuition.py
"""
Тестирование интуиции бота.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emotions.emotion_system import EmotionSystem
from core.emotions.intuition_engine import IntuitionEngine


def test_intuition():
    """Тестирует работу интуиции."""
    print("🧠 Тестирование интуиции...")

    # Создаем систему
    emotion_system = EmotionSystem()
    intuition = emotion_system.intuition

    # Обучаем интуицию на примерах
    print("\n📚 Обучение интуиции...")

    # Пример 1: Яблоко рядом с волком
    intuition.learn_from_experience(
        situation="вижу яблоко рядом с волком",
        action="пойти за яблоком",
        consequence="волк напал",
        emotion="страх",
        success=False,
        intensity=0.9
    )

    # Пример 2: Яблоко рядом с огнем
    intuition.learn_from_experience(
        situation="вижу яблоко рядом с огнем",
        action="пойти за яблоком",
        consequence="обжёгся",
        emotion="страх",
        success=False,
        intensity=0.8
    )

    # Пример 3: Успешный сбор яблока
    intuition.learn_from_experience(
        situation="вижу яблоко в безопасном месте",
        action="пойти за яблоком",
        consequence="собрал яблоко",
        emotion="радость",
        success=True,
        intensity=0.9
    )

    # Тестируем предсказания
    print("\n🔮 Предсказания интуиции:")

    test_cases = [
        ("вижу яблоко рядом с волком", "пойти за яблоком"),
        ("вижу яблоко рядом с огнем", "пойти за яблоком"),
        ("вижу яблоко в безопасном месте", "пойти за яблоком"),
        ("вижу хищника", "подойти близко"),
    ]

    for situation, action in test_cases:
        prediction = intuition.predict_consequence(situation, action)
        if prediction:
            print(f"\n  Ситуация: {situation}")
            print(f"  Действие: {action}")
            print(f"  → {prediction['consequence']} (эмоция: {prediction['emotion']}, вероятность: {prediction['probability']:.2f})")
        else:
            print(f"\n  Ситуация: {situation} → Нет опыта")

    print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
    test_intuition()