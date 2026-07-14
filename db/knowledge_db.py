# db/knowledge_db.py
"""
Модуль для работы с БД графа знаний.
"""
import time

import psycopg2
import psycopg2.extras
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.combination import Combination
from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
from core.knowledge.function import Function


class KnowledgeDB:
    """
    Класс для работы с БД графа знаний.
    """

    def __init__(self, host='localhost', port=5432,
                 dbname='postgres', user='postgres', password='postgres'):
        self.conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.schema = 'agi_evolution'

    def _get_connection(self):
        """Возвращает соединение с БД."""
        return psycopg2.connect(**self.conn_params)

    def _to_json(self, data):
        """Преобразует данные в JSON."""
        return json.dumps(data) if data is not None else None

    def _from_json(self, data):
        """Преобразует JSON в объект Python."""
        return json.loads(data) if data else {}

    # ============================================================
    # УЗЛЫ
    # ============================================================

    def save_node(self, node: KnowledgeNode) -> bool:
        """Сохраняет узел в БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.knowledge_nodes 
                (id, name, node_type, properties, description, embedding, parameters, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    node_type = EXCLUDED.node_type,
                    properties = EXCLUDED.properties,
                    description = EXCLUDED.description,
                    embedding = EXCLUDED.embedding,
                    parameters = EXCLUDED.parameters,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                node.id,
                node.name,
                node.node_type,
                node.properties,
                node.description,
                node.embedding.tolist() if hasattr(node.embedding, 'tolist') else node.embedding,
                self._to_json(node.parameters),
                self._to_json(node.metadata)
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения узла {node.id}: {e}")
            return False

    def load_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Загружает узел из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, name, node_type, properties, description, 
                       embedding, parameters, metadata
                FROM {self.schema}.knowledge_nodes
                WHERE id = %s
            """, (node_id,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            node = KnowledgeNode(
                id=row['id'],
                name=row['name'],
                node_type=row['node_type'],
                properties=row['properties'] if row['properties'] else [],
                description=row['description'] or '',
                embedding=np.array(row['embedding']) if row['embedding'] else None
            )

            node.parameters = self._from_json(row['parameters'])
            node.metadata = self._from_json(row['metadata'])

            # Загружаем связи
            node.connections = self._get_node_connections(node_id)

            return node
        except Exception as e:
            print(f"❌ Ошибка загрузки узла {node_id}: {e}")
            return None

    def load_nodes_by_properties(self, properties: List[str], limit: int = 100) -> List[KnowledgeNode]:
        """Загружает узлы по свойствам."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Используем оператор @> для проверки наличия всех свойств
            cur.execute(f"""
                SELECT id, name, node_type, properties, description, 
                       embedding, parameters, metadata
                FROM {self.schema}.knowledge_nodes
                WHERE properties @> %s::text[]
                LIMIT %s
            """, (properties, limit))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            nodes = []
            for row in rows:
                node = KnowledgeNode(
                    id=row['id'],
                    name=row['name'],
                    node_type=row['node_type'],
                    properties=row['properties'] if row['properties'] else [],
                    description=row['description'] or '',
                    embedding=np.array(row['embedding']) if row['embedding'] else None
                )
                node.parameters = self._from_json(row['parameters'])
                node.metadata = self._from_json(row['metadata'])
                nodes.append(node)

            return nodes
        except Exception as e:
            print(f"❌ Ошибка загрузки узлов по свойствам: {e}")
            return []

    def _get_node_connections(self, node_id: str) -> List[str]:
        """Загружает связи узла."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                SELECT target_id FROM {self.schema}.knowledge_edges
                WHERE source_id = %s
                UNION
                SELECT source_id FROM {self.schema}.knowledge_edges
                WHERE target_id = %s
            """, (node_id, node_id))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            return [row[0] for row in rows]
        except Exception:
            return []

    # ============================================================
    # РЕБРА
    # ============================================================

    def save_edge(self, edge: KnowledgeEdge) -> bool:
        """Сохраняет ребро в БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.knowledge_edges 
                (id, source_id, target_id, edge_type, weight, description, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    edge_type = EXCLUDED.edge_type,
                    weight = EXCLUDED.weight,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.weight,
                edge.description,
                self._to_json(edge.metadata)
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения ребра {edge.id}: {e}")
            return False

    def load_edges_from_node(self, node_id: str) -> List[KnowledgeEdge]:
        """Загружает все ребра, связанные с узлом."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, edge_type, weight, description, metadata
                FROM {self.schema}.knowledge_edges
                WHERE source_id = %s OR target_id = %s
            """, (node_id, node_id))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            edges = []
            for row in rows:
                edge = KnowledgeEdge(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    edge_type=EdgeType(row['edge_type']),
                    weight=row['weight'],
                    description=row['description'] or '',
                    metadata=self._from_json(row['metadata'])
                )
                edges.append(edge)

            return edges
        except Exception as e:
            print(f"❌ Ошибка загрузки ребер для узла {node_id}: {e}")
            return []

    # ============================================================
    # ИНДИВИДУАЛЬНЫЙ ГРАФ
    # ============================================================

    def init_individual_graph(self, bot_id: str, name: str, description: str = "") -> bool:
        """Инициализирует индивидуальный граф для бота."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.individual_knowledge_graphs 
                (bot_id, name, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (bot_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    updated_at = CURRENT_TIMESTAMP
            """, (bot_id, name, description))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка инициализации ИГЗ для {bot_id}: {e}")
            return False

    def sync_node_to_individual(self, bot_id: str, node_id: str, confidence: float = 0.5) -> bool:
        """Синхронизирует узел с индивидуальным графом бота."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.individual_node_links 
                (bot_id, node_id, confidence, last_used)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (bot_id, node_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    last_used = CURRENT_TIMESTAMP
            """, (bot_id, node_id, confidence))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка синхронизации узла {node_id} для {bot_id}: {e}")
            return False

    def load_individual_nodes(self, bot_id: str) -> List[KnowledgeNode]:
        """Загружает все узлы, синхронизированные с ботом."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT n.id, n.name, n.node_type, n.properties, n.description, 
                       n.embedding, n.parameters, n.metadata
                FROM {self.schema}.knowledge_nodes n
                JOIN {self.schema}.individual_node_links l ON n.id = l.node_id
                WHERE l.bot_id = %s
                ORDER BY l.confidence DESC
            """, (bot_id,))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            nodes = []
            for row in rows:
                node = KnowledgeNode(
                    id=row['id'],
                    name=row['name'],
                    node_type=row['node_type'],
                    properties=row['properties'] if row['properties'] else [],
                    description=row['description'] or '',
                    embedding=np.array(row['embedding']) if row['embedding'] else None
                )
                node.parameters = self._from_json(row['parameters'])
                node.metadata = self._from_json(row['metadata'])
                nodes.append(node)

            return nodes
        except Exception as e:
            print(f"❌ Ошибка загрузки индивидуальных узлов для {bot_id}: {e}")
            return []

    def load_all_nodes(self) -> List:
        """Загружает все узлы из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, name, node_type, properties, description, parameters, metadata
                FROM {self.schema}.knowledge_nodes
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            from core.knowledge.knowledge_node import KnowledgeNode

            nodes = []
            for row in rows:
                node = KnowledgeNode(
                    id=row['id'],
                    name=row['name'],
                    node_type=row['node_type'],
                    properties=row['properties'] if row['properties'] else [],
                    description=row['description'] or ''
                )

                # Восстанавливаем параметры и метаданные
                # Они уже могут быть словарями (если из JSONB), или строками
                if row.get('parameters'):
                    if isinstance(row['parameters'], dict):
                        node.parameters = row['parameters']
                    else:
                        node.parameters = self._from_json(row['parameters'])

                if row.get('metadata'):
                    if isinstance(row['metadata'], dict):
                        node.metadata = row['metadata']
                    else:
                        node.metadata = self._from_json(row['metadata'])

                nodes.append(node)

            return nodes
        except Exception as e:
            print(f"❌ Ошибка загрузки узлов: {e}")
            return []

    def save_mental_model(self, model) -> bool:
        """
        Сохраняет ментальную модель в БД.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Генерируем UUID, если id не является валидным UUID
            import uuid
            try:
                uuid.UUID(model.id)
                model_id = model.id
            except (ValueError, AttributeError, TypeError):
                model_id = str(uuid.uuid4())
                print(f"   ⚠️ ID {model.id} не является UUID, заменён на {model_id}")

            # Подготовка данных
            model_name = model.name if hasattr(model, 'name') else "Ментальная модель"
            model_type = getattr(model, 'model_type', 'mental_model')

            # Свойства - JSONB
            if hasattr(model, 'properties'):
                if isinstance(model.properties, dict):
                    properties_json = json.dumps(model.properties)
                else:
                    properties_json = json.dumps({'props': model.properties})
            else:
                properties_json = json.dumps({})

            # Последовательность - ТЕКСТОВЫЙ МАССИВ (text[])
            if hasattr(model, 'sequence') and model.sequence:
                if isinstance(model.sequence, list):
                    sequence_array = '{' + ','.join(f'"{s}"' for s in model.sequence) + '}'
                else:
                    sequence_array = '{}'
            else:
                sequence_array = '{}'

            # Эмбеддинг - МАССИВ DOUBLE PRECISION (double precision[])
            if hasattr(model, 'embedding') and model.embedding is not None:
                if hasattr(model.embedding, 'tolist'):
                    embedding_list = model.embedding.tolist()
                elif isinstance(model.embedding, list):
                    embedding_list = model.embedding
                else:
                    embedding_list = list(model.embedding)

                # Формат для double precision[]: {1.0, 0.5, 0.2}
                embedding_array = '{' + ','.join(str(float(x)) for x in embedding_list) + '}'
            else:
                # Пустой массив
                embedding_array = '{}'

            # Метаданные - JSONB
            if hasattr(model, 'metadata'):
                metadata_json = json.dumps(model.metadata)
            else:
                metadata_json = json.dumps({})

            # Описание
            description = getattr(model, 'description', model_name)

            # Выполняем запрос
            cur.execute("""
                INSERT INTO agi_evolution.mental_models 
                (id, name, model_type, properties, sequence, embedding, description, metadata, created_at)
                VALUES (%s, %s, %s, %s::jsonb, %s::text[], %s::double precision[], %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    model_type = EXCLUDED.model_type,
                    properties = EXCLUDED.properties,
                    sequence = EXCLUDED.sequence,
                    embedding = EXCLUDED.embedding,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                model_id,
                model_name,
                model_type,
                properties_json,  # ::jsonb
                sequence_array,  # ::text[]
                embedding_array,  # ::double precision[]  ← ИСПРАВЛЕНО!
                description,
                metadata_json,  # ::jsonb
                datetime.now()
            ))

            conn.commit()
            cur.close()
            conn.close()

            print(f"✅ Ментальная модель сохранена: {model_id} ({model_name})")
            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения ментальной модели: {e}")
            import traceback
            traceback.print_exc()
            return False

    # def save_mental_model(self, model) -> bool:
    #     """
    #     Сохраняет ментальную модель в БД.
    #     """
    #     try:
    #         conn = self._get_connection()
    #         cur = conn.cursor()
    #
    #         # Генерируем UUID, если id не является валидным UUID
    #         import uuid
    #         try:
    #             uuid.UUID(model.id)
    #             model_id = model.id
    #         except (ValueError, AttributeError, TypeError):
    #             model_id = str(uuid.uuid4())
    #             print(f"   ⚠️ ID {model.id} не является UUID, заменён на {model_id}")
    #
    #         # Подготовка данных
    #         model_name = model.name if hasattr(model, 'name') else "Ментальная модель"
    #         model_type = getattr(model, 'model_type', 'mental_model')
    #
    #         # Свойства - ДЛЯ JSONB ИСПОЛЬЗУЕМ json.dumps()
    #         if hasattr(model, 'properties'):
    #             if isinstance(model.properties, dict):
    #                 properties_json = json.dumps(model.properties)
    #             else:
    #                 properties_json = json.dumps({'props': model.properties})
    #         else:
    #             properties_json = json.dumps({})
    #
    #         # Последовательность - ДЛЯ JSONB ИСПОЛЬЗУЕМ json.dumps()
    #         if hasattr(model, 'sequence'):
    #             sequence_json = json.dumps(model.sequence)
    #         else:
    #             sequence_json = json.dumps([])
    #
    #         # Эмбеддинг
    #         if hasattr(model, 'embedding') and model.embedding is not None:
    #             if hasattr(model.embedding, 'tolist'):
    #                 embedding_json = json.dumps(model.embedding.tolist())
    #             else:
    #                 embedding_json = json.dumps(model.embedding)
    #         else:
    #             embedding_json = None
    #
    #         # Метаданные - ДЛЯ JSONB ИСПОЛЬЗУЕМ json.dumps()
    #         if hasattr(model, 'metadata'):
    #             metadata_json = json.dumps(model.metadata)
    #         else:
    #             metadata_json = json.dumps({})
    #
    #         # Описание
    #         description = getattr(model, 'description', model_name)
    #
    #         # Выполняем запрос - используем jsonb_set или просто передаём JSON
    #         cur.execute("""
    #             INSERT INTO agi_evolution.mental_models
    #             (id, name, model_type, properties, sequence, embedding, description, metadata, created_at)
    #             VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s)
    #             ON CONFLICT (id) DO UPDATE SET
    #                 name = EXCLUDED.name,
    #                 model_type = EXCLUDED.model_type,
    #                 properties = EXCLUDED.properties,
    #                 sequence = EXCLUDED.sequence,
    #                 embedding = EXCLUDED.embedding,
    #                 description = EXCLUDED.description,
    #                 metadata = EXCLUDED.metadata,
    #                 updated_at = CURRENT_TIMESTAMP
    #         """, (
    #             model_id,
    #             model_name,
    #             model_type,
    #             properties_json,  # ← json.dumps()
    #             sequence_json,  # ← json.dumps()
    #             embedding_json,  # ← json.dumps()
    #             description,
    #             metadata_json,  # ← json.dumps()
    #             datetime.now()
    #         ))
    #
    #         conn.commit()
    #         cur.close()
    #         conn.close()
    #
    #         print(f"✅ Ментальная модель сохранена: {model_id} ({model_name})")
    #         return True
    #
    #     except Exception as e:
    #         print(f"❌ Ошибка сохранения ментальной модели: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False


    # def save_mental_model(self, model) -> bool:
    #     """
    #     Сохраняет ментальную модель в БД.
    #
    #     Args:
    #         model: Объект MentalModel из core.thinking.models
    #
    #     Returns:
    #         True если сохранено успешно
    #     """
    #     try:
    #         conn = self._get_connection()
    #         cur = conn.cursor()
    #
    #         # Подготовка данных
    #         model_id = model.id
    #         model_name = model.name if hasattr(model, 'name') else "Ментальная модель"
    #         model_type = getattr(model, 'model_type', 'mental_model')
    #
    #         # Свойства
    #         if hasattr(model, 'properties'):
    #             if isinstance(model.properties, dict):
    #                 properties_json = json.dumps(model.properties)
    #             else:
    #                 properties_json = json.dumps({'props': model.properties})
    #         else:
    #             properties_json = '{}'
    #
    #         # Последовательность
    #         if hasattr(model, 'sequence'):
    #             sequence_json = json.dumps(model.sequence)
    #         else:
    #             sequence_json = '[]'
    #
    #         # Эмбеддинг
    #         if hasattr(model, 'embedding') and model.embedding is not None:
    #             if hasattr(model.embedding, 'tolist'):
    #                 embedding_json = json.dumps(model.embedding.tolist())
    #             else:
    #                 embedding_json = json.dumps(model.embedding)
    #         else:
    #             embedding_json = None
    #
    #         # Метаданные
    #         if hasattr(model, 'metadata'):
    #             metadata_json = json.dumps(model.metadata)
    #         else:
    #             metadata_json = '{}'
    #
    #         # Описание
    #         description = getattr(model, 'description', model_name)
    #
    #         # Выполняем запрос
    #         cur.execute("""
    #             INSERT INTO agi_evolution.mental_models
    #             (id, name, model_type, properties, sequence, embedding, description, metadata, created_at)
    #             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    #             ON CONFLICT (id) DO UPDATE SET
    #                 name = EXCLUDED.name,
    #                 model_type = EXCLUDED.model_type,
    #                 properties = EXCLUDED.properties,
    #                 sequence = EXCLUDED.sequence,
    #                 embedding = EXCLUDED.embedding,
    #                 description = EXCLUDED.description,
    #                 metadata = EXCLUDED.metadata,
    #                 updated_at = CURRENT_TIMESTAMP
    #         """, (
    #             model_id,
    #             model_name,
    #             model_type,
    #             properties_json,
    #             sequence_json,
    #             embedding_json,
    #             description,
    #             metadata_json,
    #             datetime.now()
    #         ))
    #
    #         conn.commit()
    #         cur.close()
    #         conn.close()
    #
    #         print(f"✅ Ментальная модель сохранена: {model_id} ({model_name})")
    #         return True
    #
    #     except Exception as e:
    #         print(f"❌ Ошибка сохранения ментальной модели {model.id if hasattr(model, 'id') else 'unknown'}: {e}")
    #         return False

    # def save_mental_model(self, model) -> bool:
    #     """Сохраняет ментальную модель в БД."""
    #     try:
    #         conn = self._get_connection()
    #         cur = conn.cursor()
    #
    #         # Проверяем существование колонок
    #         cur.execute("""
    #             SELECT column_name
    #             FROM information_schema.columns
    #             WHERE table_schema = 'agi_evolution'
    #               AND table_name = 'mental_models'
    #         """)
    #         existing_columns = [row[0] for row in cur.fetchall()]
    #
    #         # Формируем запрос в зависимости от существующих колонок
    #         columns = ['id']
    #         placeholders = ['%s']
    #         values = [model.id]
    #
    #         # name
    #         if 'name' in existing_columns:
    #             columns.append('name')
    #             placeholders.append('%s')
    #             values.append(model.name)
    #
    #         # model_type
    #         if 'model_type' in existing_columns:
    #             columns.append('model_type')
    #             placeholders.append('%s')
    #             values.append('mental_model')
    #
    #         # properties (бывшие attributes)
    #         if 'properties' in existing_columns:
    #             columns.append('properties')
    #             placeholders.append('%s')
    #             values.append(self._to_json(model.properties if hasattr(model, 'properties') else {}))
    #
    #         # sequence
    #         if 'sequence' in existing_columns:
    #             columns.append('sequence')
    #             placeholders.append('%s')
    #             values.append(model.sequence if hasattr(model, 'sequence') else [])
    #
    #         # embedding
    #         if 'embedding' in existing_columns:
    #             columns.append('embedding')
    #             placeholders.append('%s')
    #             values.append(model.embedding if hasattr(model, 'embedding') else None)
    #
    #         # description
    #         if 'description' in existing_columns:
    #             columns.append('description')
    #             placeholders.append('%s')
    #             values.append(model.name)
    #
    #         # metadata
    #         if 'metadata' in existing_columns:
    #             columns.append('metadata')
    #             placeholders.append('%s')
    #             values.append(self._to_json(model.metadata if hasattr(model, 'metadata') else {}))
    #
    #         # created_at
    #         if 'created_at' in existing_columns:
    #             columns.append('created_at')
    #             placeholders.append('%s')
    #             values.append(datetime.now())
    #
    #         # updated_at
    #         if 'updated_at' in existing_columns:
    #             columns.append('updated_at')
    #             placeholders.append('%s')
    #             values.append(datetime.now())
    #
    #         # Строим и выполняем запрос
    #         columns_str = ', '.join(columns)
    #         placeholders_str = ', '.join(placeholders)
    #         update_parts = []
    #
    #         for col in columns:
    #             if col != 'id' and col != 'created_at':
    #                 update_parts.append(f"{col} = EXCLUDED.{col}")
    #
    #         update_str = ', '.join(update_parts) if update_parts else 'updated_at = EXCLUDED.updated_at'
    #
    #         query = f"""
    #             INSERT INTO agi_evolution.mental_models
    #             ({columns_str})
    #             VALUES ({placeholders_str})
    #             ON CONFLICT (id) DO UPDATE SET
    #                 {update_str}
    #         """
    #
    #         cur.execute(query, values)
    #         conn.commit()
    #         cur.close()
    #         conn.close()
    #         return True
    #
    #     except Exception as e:
    #         print(f"❌ Ошибка сохранения ментальной модели {model.id}: {e}")
    #         return False


    def load_all_mental_models(self) -> List:
        """Загружает все ментальные модели из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT id, name, model_type, properties, sequence, embedding, 
                       description, metadata, created_at
                FROM agi_evolution.mental_models
                ORDER BY created_at DESC
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            from core.thinking.models import MentalModel

            models = []
            for row in rows:
                # Восстанавливаем свойства
                if row.get('properties'):
                    if isinstance(row['properties'], dict):
                        properties = row['properties']
                    else:
                        try:
                            properties = json.loads(row['properties'])
                        except:
                            properties = {}
                else:
                    properties = {}

                # Восстанавливаем последовательность
                if row.get('sequence'):
                    if isinstance(row['sequence'], list):
                        sequence = row['sequence']
                    else:
                        try:
                            sequence = json.loads(row['sequence'])
                        except:
                            sequence = []
                else:
                    sequence = []

                # Восстанавливаем эмбеддинг
                if row.get('embedding'):
                    if isinstance(row['embedding'], list):
                        embedding = row['embedding']
                    else:
                        try:
                            embedding = json.loads(row['embedding'])
                        except:
                            embedding = None
                else:
                    embedding = None

                # Восстанавливаем метаданные
                if row.get('metadata'):
                    if isinstance(row['metadata'], dict):
                        metadata = row['metadata']
                    else:
                        try:
                            metadata = json.loads(row['metadata'])
                        except:
                            metadata = {}
                else:
                    metadata = {}

                model = MentalModel(
                    id=row['id'],
                    name=row.get('name', ''),
                    sequence=sequence,
                    embedding=embedding,
                    properties=properties,
                    metadata=metadata,
                    created_at=row['created_at'].timestamp() if row.get('created_at') else time.time()
                )
                model.model_type = row.get('model_type', 'mental_model')
                models.append(model)

            return models
        except Exception as e:
            print(f"❌ Ошибка загрузки ментальных моделей: {e}")
            return []

    # def load_all_mental_models(self) -> List:
    #     """Загружает все ментальные модели из БД."""
    #     try:
    #         conn = self._get_connection()
    #         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    #
    #         cur.execute(f"""
    #             SELECT * FROM agi_evolution.mental_models
    #             ORDER BY created_at DESC
    #         """)
    #
    #         rows = cur.fetchall()
    #         cur.close()
    #         conn.close()
    #
    #         from core.thinking.models import MentalModel
    #
    #         models = []
    #         for row in rows:
    #             model = MentalModel(
    #                 id=row['id'],
    #                 name=row.get('name', ''),
    #                 sequence=row.get('sequence', []),
    #                 embedding=row.get('embedding'),
    #                 properties=row.get('properties', {}) if isinstance(row.get('properties'), dict) else row.get('properties', []),
    #                 metadata=row.get('metadata', {}),
    #                 created_at=row.get('created_at').timestamp() if row.get('created_at') else time.time()
    #             )
    #             models.append(model)
    #
    #         return models
    #     except Exception as e:
    #         print(f"❌ Ошибка загрузки ментальных моделей: {e}")
    #         return []

    def get_mental_model(self, model_id: str):
        """Загружает ментальную модель по ID."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT * FROM agi_evolution.mental_models
                WHERE id = %s
            """, (model_id,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            from core.thinking.models import MentalModel

            return MentalModel(
                id=row['id'],
                name=row.get('name', ''),
                sequence=row.get('sequence', []),
                embedding=row.get('embedding'),
                properties=row.get('properties', {}) if isinstance(row.get('properties'), dict) else row.get('properties', []),
                metadata=row.get('metadata', {}),
                created_at=row.get('created_at').timestamp() if row.get('created_at') else time.time()
            )
        except Exception as e:
            print(f"❌ Ошибка загрузки ментальной модели {model_id}: {e}")
            return None

    def delete_mental_model(self, model_id: str) -> bool:
        """Удаляет ментальную модель."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                DELETE FROM agi_evolution.mental_models
                WHERE id = %s
            """, (model_id,))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка удаления ментальной модели {model_id}: {e}")
            return False


    def load_all_edges(self) -> List:
        """Загружает все ребра из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, edge_type, weight, description, metadata
                FROM {self.schema}.knowledge_edges
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType

            edges = []
            for row in rows:
                edge = KnowledgeEdge(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    edge_type=EdgeType(row['edge_type']),
                    weight=row['weight'] or 0.5,
                    description=row['description'] or '',
                    metadata=self._from_json(row['metadata']) if row.get('metadata') else {}
                )
                edges.append(edge)

            return edges
        except Exception as e:
            print(f"❌ Ошибка загрузки ребер: {e}")
            return []