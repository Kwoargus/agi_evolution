# scripts/visualize_hypothesis.py
"""
Визуализация графа гипотезы.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional

from db.knowledge_db import KnowledgeDB
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.knowledge_node import KnowledgeNode


class HypothesisVisualizer:
    """
    Визуализатор гипотез.
    """

    # Цвета для разных статусов функций
    COLOR_COVERED = '#4CAF50'   # зелёный — покрыто
    COLOR_MISSING = '#F44336'   # красный — недостаёт
    COLOR_MAIN = '#2196F3'      # синий — главный узел
    COLOR_PART = '#FF9800'      # оранжевый — часть
    COLOR_ADDED = '#9C27B0'     # фиолетовый — добавленный узел

    def __init__(self):
        self.db = KnowledgeDB()
        self.gg = GlobalKnowledgeGraph()
        self.gg.load_from_db()

    def visualize_hypothesis(self, hypothesis_id: Optional[str] = None,
                             save_path: str = "hypothesis_graph.png"):
        """
        Визуализирует гипотезу.

        Args:
            hypothesis_id: ID гипотезы (если None — берём последнюю)
            save_path: Путь для сохранения изображения
        """
        # 1. Загружаем гипотезу
        if hypothesis_id is None:
            hypothesis = self._get_latest_hypothesis()
        else:
            hypothesis = self._get_hypothesis_by_id(hypothesis_id)

        if not hypothesis:
            print("❌ Гипотеза не найдена")
            return

        print(f"📌 Визуализация гипотезы: {hypothesis['id']}")
        print(f"   Задача: {hypothesis.get('task_description', 'не указана')}")
        print(f"   Статус: {hypothesis['status']}")
        print(f"   Оценка: {hypothesis['actual_score']:.2f}")

        # 2. Строим граф
        G = nx.DiGraph()

        # Получаем информацию об узлах
        metadata = hypothesis.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        node_names = metadata.get('node_names', [])
        covered = metadata.get('covered_functions', [])
        missing = metadata.get('missing_functions', [])
        added = metadata.get('added_nodes', [])
        modifications = hypothesis.get('modifications', [])

        # Если нет node_names, пытаемся восстановить из source_combination
        if not node_names:
            source_id = hypothesis.get('source_combination_id')
            if source_id:
                # Пытаемся загрузить комбинацию (упрощённо)
                # В реальности нужно загружать комбинацию из БД
                pass
            node_names = ["Главный узел", "Часть 1", "Часть 2"]  # fallback

        print(f"\n📦 Узлы гипотезы: {len(node_names)}")
        for name in node_names:
            is_added = any(a in name for a in added) or any(mod in name for mod in modifications)
            is_main = name == node_names[0] if node_names else False
            print(f"   - {name} {'(добавлен)' if is_added else ''}")

        print(f"\n✅ Покрытые функции: {len(covered)}")
        for f in covered:
            print(f"   - {f}")

        print(f"\n❌ Недостающие функции: {len(missing)}")
        for f in missing:
            print(f"   - {f}")

        # 3. Добавляем узлы в граф
        for i, name in enumerate(node_names):
            is_added = any(a in name for a in added) or any(mod in name for mod in modifications)
            is_main = i == 0

            if is_main:
                color = self.COLOR_MAIN
                node_type = 'main'
            elif is_added:
                color = self.COLOR_ADDED
                node_type = 'added'
            else:
                color = self.COLOR_PART
                node_type = 'part'

            G.add_node(name, color=color, node_type=node_type, is_main=is_main, is_added=is_added)

        # 4. Добавляем связи (если есть данные о связях в metadata)
        # Если нет — создаём простые связи "часть → главный"
        if len(node_names) > 1:
            main_node = node_names[0]
            for name in node_names[1:]:
                # Проверяем, есть ли связи в metadata
                connections = metadata.get('connections', [])
                if connections:
                    for conn in connections:
                        if conn.get('source') == main_node and conn.get('target') == name:
                            G.add_edge(main_node, name, relation=conn.get('description', 'has_part'))
                        elif conn.get('source') == name and conn.get('target') == main_node:
                            G.add_edge(name, main_node, relation=conn.get('description', 'part_of'))
                else:
                    # Автоматическая связь "содержит"
                    G.add_edge(main_node, name, relation='содержит')

        # 5. Добавляем функции как узлы (для наглядности)
        # Создаём узлы для функций и связываем с узлами, которые их выполняют
        # Это упрощённо: пока просто добавляем информацию в атрибуты

        # 6. Отрисовка
        plt.figure(figsize=(14, 10))

        # Позиционирование (радиальное)
        pos = nx.spring_layout(G, k=1.5, seed=42)

        # Цвета узлов
        colors = [G.nodes[n]['color'] for n in G.nodes]

        # Размеры узлов
        sizes = []
        for n in G.nodes:
            if G.nodes[n].get('is_main', False):
                sizes.append(3000)
            elif G.nodes[n].get('is_added', False):
                sizes.append(1500)
            else:
                sizes.append(1200)

        # Отрисовка узлов
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)

        # Отрисовка рёбер
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.7, arrows=True, arrowsize=20)

        # Отрисовка подписей
        labels = {}
        for n in G.nodes:
            label = n
            if G.nodes[n].get('is_added', False):
                label = f"{n} ✦"
            elif G.nodes[n].get('is_main', False):
                label = f"★ {n}"
            labels[n] = label

        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

        # Подписи рёбер
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        # Легенда
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLOR_MAIN,
                      markersize=15, label='Главный узел'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLOR_PART,
                      markersize=15, label='Часть'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLOR_ADDED,
                      markersize=15, label='Добавленный узел'),
        ]

        # Информация о покрытии
        status_text = f"Статус: {hypothesis['status']}\n"
        status_text += f"Оценка: {hypothesis['actual_score']:.2f}\n"
        status_text += f"Покрыто функций: {len(covered)}/{len(covered)+len(missing)}\n"
        if covered:
            status_text += f"✅ {', '.join(covered[:3])}"
            if len(covered) > 3:
                status_text += f" и ещё {len(covered)-3}"
        if missing:
            status_text += f"\n❌ {', '.join(missing[:3])}"
            if len(missing) > 3:
                status_text += f" и ещё {len(missing)-3}"

        plt.text(0.02, 0.98, status_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(handles=legend_elements, loc='lower left', fontsize=10)

        plt.title(f"Граф гипотезы: {hypothesis['id'][:12]}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Граф сохранён: {save_path}")
        plt.close()

    def _get_latest_hypothesis(self) -> Optional[Dict]:
        """Возвращает последнюю гипотезу из БД."""
        hypotheses = self.db.load_all_hypotheses()
        if not hypotheses:
            return None
        return hypotheses[0]  # уже отсортированы по created_at DESC

    def _get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[Dict]:
        """Возвращает гипотезу по ID."""
        hypotheses = self.db.load_all_hypotheses()
        for h in hypotheses:
            if h['id'] == hypothesis_id:
                return h
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Визуализация графа гипотезы')
    parser.add_argument('--id', type=str, help='ID гипотезы (если не указан — последняя)')
    parser.add_argument('--output', type=str, default='hypothesis_graph.png',
                       help='Путь для сохранения изображения')
    args = parser.parse_args()

    visualizer = HypothesisVisualizer()
    visualizer.visualize_hypothesis(
        hypothesis_id=args.id,
        save_path=args.output
    )


if __name__ == "__main__":
    main()