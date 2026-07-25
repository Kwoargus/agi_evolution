# scripts/inspect_graphrag_output.py
"""
Просмотр результатов работы GraphRAG.
"""

import pandas as pd
import os

def inspect_output(output_dir="output"):
    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ GRAPHRAG")
    print("=" * 70)

    # 1. Entities (узлы)
    entities_path = os.path.join(output_dir, "entities.parquet")
    if os.path.exists(entities_path):
        entities = pd.read_parquet(entities_path)
        print(f"\n📌 СУЩНОСТИ (УЗЛЫ): {len(entities)}")
        print("=" * 50)
        print(entities[['title', 'type', 'description', 'frequency']].head(10).to_string(index=False))
    else:
        print("❌ entities.parquet не найден")

    # 2. Relationships (связи)
    rel_path = os.path.join(output_dir, "relationships.parquet")
    if os.path.exists(rel_path):
        relationships = pd.read_parquet(rel_path)
        print(f"\n🔗 СВЯЗИ (РЁБРА): {len(relationships)}")
        print("=" * 50)
        print(relationships[['source', 'target', 'description', 'weight']].head(10).to_string(index=False))
    else:
        print("❌ relationships.parquet не найден")

    # 3. Communities (сообщества)
    comm_path = os.path.join(output_dir, "communities.parquet")
    if os.path.exists(comm_path):
        communities = pd.read_parquet(comm_path)
        print(f"\n👥 СООБЩЕСТВА: {len(communities)}")
        print("=" * 50)
        print(communities[['community', 'level', 'title', 'size']].head(10).to_string(index=False))
    else:
        print("❌ communities.parquet не найден")

    # 4. Community Reports (отчёты)
    reports_path = os.path.join(output_dir, "community_reports.parquet")
    if os.path.exists(reports_path):
        reports = pd.read_parquet(reports_path)
        print(f"\n📄 ОТЧЁТЫ ПО СООБЩЕСТВАМ: {len(reports)}")
        print("=" * 50)
        for idx, row in reports.head(3).iterrows():
            print(f"\n--- Отчёт {row['community']} ---")
            print(f"Название: {row['title']}")
            print(f"Краткое содержание: {row.get('summary', '')[:300]}...")
    else:
        print("❌ community_reports.parquet не найден")

    print("\n" + "=" * 70)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")

if __name__ == "__main__":
    inspect_output()