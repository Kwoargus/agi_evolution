# db/knowledge_db.py
"""
[ru] Модуль для работы с БД графа знаний.
[en] Module for working with the knowledge graph database.
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
    [ru] Класс для работы с БД графа знаний.
    [en] Class for working with the knowledge graph database.
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


    def save_combination(self, combination) -> bool:
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            node_ids = [n.id for n in combination.nodes]
            edge_ids = []
            properties = combination.properties
            score = combination.metadata.get('score', 0.0)
            metadata = json.dumps(combination.metadata)

            cur.execute(f"""
                INSERT INTO {self.schema}.combinations 
                (id, node_ids, edge_ids, properties, score, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    node_ids = EXCLUDED.node_ids,
                    edge_ids = EXCLUDED.edge_ids,
                    properties = EXCLUDED.properties,
                    score = EXCLUDED.score,
                    metadata = EXCLUDED.metadata
            """, (
                combination.id,
                node_ids,
                edge_ids,
                properties,
                score,
                metadata,
                datetime.now()
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"[ru] ❌ Ошибка сохранения комбинации {combination.id}: {e}")
            print(f"[en] ❌ Error saving combination {combination.id}: {e}")
            return False

    def load_all_hypotheses(self) -> List[Dict]:
        """
        [ru] Загружает все гипотезы из БД (возвращает список словарей).
        [en] Loads all hypotheses from the database (returns a list of dictionaries).
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT * FROM {self.schema}.hypotheses
                ORDER BY created_at DESC
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            # [ru] Преобразуем JSONB в dict
            # [en] Convert JSONB to dict
            for row in rows:
                if row.get('test_results'):
                    if isinstance(row['test_results'], str):
                        row['test_results'] = json.loads(row['test_results'])
                if row.get('metadata'):
                    if isinstance(row['metadata'], str):
                        row['metadata'] = json.loads(row['metadata'])
                # [ru] modifications уже массив
                # [en] modifications is already an array
            return rows

        except Exception as e:
            print(f"[ru] ❌ Ошибка загрузки гипотез: {e}")
            print(f"[en] ❌ Error loading hypotheses: {e}")
            return []


    def save_hypothesis(self, hypothesis) -> bool:
        """
        [ru] Сохраняет гипотезу в БД.
        [en] Saves a hypothesis to the database.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # [ru] modifications — массив строк
            # [en] modifications — array of strings
            modifications = hypothesis.modifications  # [ru] список строк  [en] list of strings

            # [ru] created_at — конвертируем в datetime
            # [en] created_at — convert to datetime
            from datetime import datetime
            created_at = datetime.fromtimestamp(hypothesis.created_at) if hasattr(hypothesis, 'created_at') else datetime.now()
            updated_at = datetime.now()

            # [ru] test_results и metadata — JSONB
            # [en] test_results and metadata — JSONB
            test_results_json = json.dumps(hypothesis.test_results)
            metadata_json = json.dumps(hypothesis.metadata)

            cur.execute(f"""
                INSERT INTO {self.schema}.hypotheses 
                (id, task_description, source_combination_id, modifications, 
                 description, predicted_score, actual_score, status, 
                 test_results, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    task_description = EXCLUDED.task_description,
                    source_combination_id = EXCLUDED.source_combination_id,
                    modifications = EXCLUDED.modifications,
                    description = EXCLUDED.description,
                    predicted_score = EXCLUDED.predicted_score,
                    actual_score = EXCLUDED.actual_score,
                    status = EXCLUDED.status,
                    test_results = EXCLUDED.test_results,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, (
                hypothesis.id,
                hypothesis.task_description,
                hypothesis.source_combination.id,
                modifications,          # [ru] ← список строк (массив)  [en] ← list of strings (array)
                hypothesis.description,
                hypothesis.predicted_score,
                hypothesis.actual_score,
                hypothesis.status.value,
                test_results_json,
                metadata_json,
                created_at,
                updated_at
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True

        except Exception as e:
            print(f"[ru] ❌ Ошибка сохранения гипотезы {hypothesis.id}: {e}")
            print(f"[en] ❌ Error saving hypothesis {hypothesis.id}: {e}")
            return False


    def _get_connection(self):
        """
        [ru] Возвращает соединение с БД.
        [en] Returns a database connection.
        """
        return psycopg2.connect(**self.conn_params)

    def _to_json(self, data):
        """
        [ru] Преобразует данные в JSON.
        [en] Converts data to JSON.
        """
        return json.dumps(data) if data is not None else None

    def _from_json(self, data):
        """
        [ru] Преобразует JSON в объект Python.
        [en] Converts JSON to a Python object.
        """
        return json.loads(data) if data else {}

    # ============================================================
    # [ru] УЗЛЫ
    # [en] NODES
    # ============================================================

    def save_node(self, node: KnowledgeNode) -> bool:
        """
        [ru] Сохраняет узел в БД.
        [en] Saves a node to the database.
        """
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
            print(f"[ru] ❌ Ошибка сохранения узла {node.id}: {e}")
            print(f"[en] ❌ Error saving node {node.id}: {e}")
            return False

    def load_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        [ru] Загружает узел из БД.
        [en] Loads a node from the database.
        """
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

            # [ru] Загружаем связи
            # [en] Load connections
            node.connections = self._get_node_connections(node_id)

            return node
        except Exception as e:
            print(f"[ru] ❌ Ошибка загрузки узла {node_id}: {e}")
            print(f"[en] ❌ Error loading node {node_id}: {e}")
            return None

    def load_nodes_by_properties(self, properties: List[str], limit: int = 100) -> List[KnowledgeNode]:
        """
        [ru] Загружает узлы по свойствам.
        [en] Loads nodes by properties.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # [ru] Используем оператор @> для проверки наличия всех свойств
            # [en] Use the @> operator to check for the presence of all properties
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
            print(f"[ru] ❌ Ошибка загрузки узлов по свойствам: {e}")
            print(f"[en] ❌ Error loading nodes by properties: {e}")
            return []

    def _get_node_connections(self, node_id: str) -> List[str]:
        """
        [ru] Загружает связи узла.
        [en] Loads the node's connections.
        """
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
    # [ru] РЕБРА
    # [en] EDGES
    # ============================================================

    def save_edge(self, edge: KnowledgeEdge) -> bool:
        """
        [ru] Сохраняет ребро в БД.
        [en] Saves an edge to the database.
        """
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
            print(f"[ru] ❌ Ошибка сохранения ребра {edge.id}: {e}")
            print(f"[en] ❌ Error saving edge {edge.id}: {e}")
            return False

    def load_edges_from_node(self, node_id: str) -> List[KnowledgeEdge]:
        """
        [ru] Загружает все ребра, связанные с узлом.
        [en] Loads all edges connected to the node.
        """
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
            print(f"[ru] ❌ Ошибка загрузки ребер для узла {node_id}: {e}")
            print(f"[en] ❌ Error loading edges for node {node_id}: {e}")
            return []

    # ============================================================
    # [ru] ИНДИВИДУАЛЬНЫЙ ГРАФ
    # [en] INDIVIDUAL GRAPH
    # ============================================================

    def init_individual_graph(self, bot_id: str, name: str, description: str = "") -> bool:
        """
        [ru] Инициализирует индивидуальный граф для бота.
        [en] Initializes the individual graph for the bot.
        """
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
            print(f"[ru] ❌ Ошибка инициализации ИГЗ для {bot_id}: {e}")
            print(f"[en] ❌ Error initializing IKG for {bot_id}: {e}")
            return False

    def sync_node_to_individual(self, bot_id: str, node_id: str, confidence: float = 0.5) -> bool:
        """
        [ru] Синхронизирует узел с индивидуальным графом бота.
        [en] Synchronizes a node with the bot's individual graph.
        """
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
            print(f"[ru] ❌ Ошибка синхронизации узла {node_id} для {bot_id}: {e}")
            print(f"[en] ❌ Error synchronizing node {node_id} for {bot_id}: {e}")
            return False

    def load_individual_nodes(self, bot_id: str) -> List[KnowledgeNode]:
        """
        [ru] Загружает все узлы, синхронизированные с ботом.
        [en] Loads all nodes synchronized with the bot.
        """
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
            print(f"[ru] ❌ Ошибка загрузки индивидуальных узлов для {bot_id}: {e}")
            print(f"[en] ❌ Error loading individual nodes for {bot_id}: {e}")
            return []

    def load_all_nodes(self) -> List:
        """
        [ru] Загружает все узлы из БД.
        [en] Loads all nodes from the database.
        """
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

                # [ru] Восстанавливаем параметры и метаданные
                # [en] Restore parameters and metadata
                # [ru] Они уже могут быть словарями (если из JSONB), или строками
                # [en] They may already be dictionaries (if from JSONB), or strings
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
            print(f"[ru] ❌ Ошибка загрузки узлов: {e}")
            print(f"[en] ❌ Error loading nodes: {e}")
            return []

    def save_mental_model(self, model) -> bool:
        """
        [ru] Сохраняет ментальную модель в БД.
        [en] Saves a mental model to the database.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # [ru] Генерируем UUID, если id не является валидным UUID
            # [en] Generate UUID if id is not a valid UUID
            import uuid
            try:
                uuid.UUID(model.id)
                model_id = model.id
            except (ValueError, AttributeError, TypeError):
                model_id = str(uuid.uuid4())
                print(f"[ru]    ⚠️ ID {model.id} не является UUID, заменён на {model_id}")
                print(f"[en]    ⚠️ ID {model.id} is not a UUID, replaced with {model_id}")

            # [ru] Подготовка данных
            # [en] Prepare data
            model_name = model.name if hasattr(model, 'name') else "Ментальная модель"
            model_type = getattr(model, 'model_type', 'mental_model')

            # [ru] Свойства - JSONB
            # [en] Properties - JSONB
            if hasattr(model, 'properties'):
                if isinstance(model.properties, dict):
                    properties_json = json.dumps(model.properties)
                else:
                    properties_json = json.dumps({'props': model.properties})
            else:
                properties_json = json.dumps({})

            # [ru] Последовательность - ТЕКСТОВЫЙ МАССИВ (text[])
            # [en] Sequence - TEXT ARRAY (text[])
            if hasattr(model, 'sequence') and model.sequence:
                if isinstance(model.sequence, list):
                    sequence_array = '{' + ','.join(f'"{s}"' for s in model.sequence) + '}'
                else:
                    sequence_array = '{}'
            else:
                sequence_array = '{}'

            # [ru] Эмбеддинг - МАССИВ DOUBLE PRECISION (double precision[])
            # [en] Embedding - DOUBLE PRECISION ARRAY (double precision[])
            if hasattr(model, 'embedding') and model.embedding is not None:
                if hasattr(model.embedding, 'tolist'):
                    embedding_list = model.embedding.tolist()
                elif isinstance(model.embedding, list):
                    embedding_list = model.embedding
                else:
                    embedding_list = list(model.embedding)

                # [ru] Формат для double precision[]: {1.0, 0.5, 0.2}
                # [en] Format for double precision[]: {1.0, 0.5, 0.2}
                embedding_array = '{' + ','.join(str(float(x)) for x in embedding_list) + '}'
            else:
                # [ru] Пустой массив
                # [en] Empty array
                embedding_array = '{}'

            # [ru] Метаданные - JSONB
            # [en] Metadata - JSONB
            if hasattr(model, 'metadata'):
                metadata_json = json.dumps(model.metadata)
            else:
                metadata_json = json.dumps({})

            # [ru] Описание
            # [en] Description
            description = getattr(model, 'description', model_name)

            # [ru] Выполняем запрос
            # [en] Execute the query
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
                embedding_array,  # ::double precision[]  [ru] ← ИСПРАВЛЕНО!  [en] ← FIXED!
                description,
                metadata_json,  # ::jsonb
                datetime.now()
            ))

            conn.commit()
            cur.close()
            conn.close()

            print(f"[ru] ✅ Ментальная модель сохранена: {model_id} ({model_name})")
            print(f"[en] ✅ Mental model saved: {model_id} ({model_name})")
            return True

        except Exception as e:
            print(f"[ru] ❌ Ошибка сохранения ментальной модели: {e}")
            print(f"[en] ❌ Error saving mental model: {e}")
            import traceback
            traceback.print_exc()
            return False


    def load_all_mental_models(self) -> List:
        """
        [ru] Загружает все ментальные модели из БД.
        [en] Loads all mental models from the database.
        """
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
                # [ru] Восстанавливаем свойства
                # [en] Restore properties
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

                # [ru] Восстанавливаем последовательность
                # [en] Restore sequence
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

                # [ru] Восстанавливаем эмбеддинг
                # [en] Restore embedding
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

                # [ru] Восстанавливаем метаданные
                # [en] Restore metadata
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
            print(f"[ru] ❌ Ошибка загрузки ментальных моделей: {e}")
            print(f"[en] ❌ Error loading mental models: {e}")
            return []



    def get_mental_model(self, model_id: str):
        """
        [ru] Загружает ментальную модель по ID.
        [en] Loads a mental model by ID.
        """
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
            print(f"[ru] ❌ Ошибка загрузки ментальной модели {model_id}: {e}")
            print(f"[en] ❌ Error loading mental model {model_id}: {e}")
            return None

    def delete_mental_model(self, model_id: str) -> bool:
        """
        [ru] Удаляет ментальную модель.
        [en] Deletes a mental model.
        """
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
            print(f"[ru] ❌ Ошибка удаления ментальной модели {model_id}: {e}")
            print(f"[en] ❌ Error deleting mental model {model_id}: {e}")
            return False


    def load_all_edges(self) -> List:
        """
        [ru] Загружает все ребра из БД.
        [en] Loads all edges from the database.
        """
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
            print(f"[ru] ❌ Ошибка загрузки ребер: {e}")
            print(f"[en] ❌ Error loading edges: {e}")
            return []





# # db/knowledge_db.py
# """
# Модуль для работы с БД графа знаний.
# """
# import time
#
# import psycopg2
# import psycopg2.extras
# import json
# import numpy as np
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime
# import uuid
#
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.combination import Combination
# from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
# from core.knowledge.function import Function
#
#
# class KnowledgeDB:
#     """
#     Класс для работы с БД графа знаний.
#     """
#
#     def __init__(self, host='localhost', port=5432,
#                  dbname='postgres', user='postgres', password='postgres'):
#         self.conn_params = {
#             'host': host,
#             'port': port,
#             'dbname': dbname,
#             'user': user,
#             'password': password
#         }
#         self.schema = 'agi_evolution'
#
#
#     def save_combination(self, combination) -> bool:
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             node_ids = [n.id for n in combination.nodes]
#             edge_ids = []
#             properties = combination.properties
#             score = combination.metadata.get('score', 0.0)
#             metadata = json.dumps(combination.metadata)
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.combinations
#                 (id, node_ids, edge_ids, properties, score, metadata, created_at)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (id) DO UPDATE SET
#                     node_ids = EXCLUDED.node_ids,
#                     edge_ids = EXCLUDED.edge_ids,
#                     properties = EXCLUDED.properties,
#                     score = EXCLUDED.score,
#                     metadata = EXCLUDED.metadata
#             """, (
#                 combination.id,
#                 node_ids,
#                 edge_ids,
#                 properties,
#                 score,
#                 metadata,
#                 datetime.now()
#             ))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка сохранения комбинации {combination.id}: {e}")
#             return False
#
#     def load_all_hypotheses(self) -> List[Dict]:
#         """
#         Загружает все гипотезы из БД (возвращает список словарей).
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT * FROM {self.schema}.hypotheses
#                 ORDER BY created_at DESC
#             """)
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             # Преобразуем JSONB в dict
#             for row in rows:
#                 if row.get('test_results'):
#                     if isinstance(row['test_results'], str):
#                         row['test_results'] = json.loads(row['test_results'])
#                 if row.get('metadata'):
#                     if isinstance(row['metadata'], str):
#                         row['metadata'] = json.loads(row['metadata'])
#                 # modifications уже массив
#             return rows
#
#         except Exception as e:
#             print(f"❌ Ошибка загрузки гипотез: {e}")
#             return []
#
#
#     def save_hypothesis(self, hypothesis) -> bool:
#         """
#         Сохраняет гипотезу в БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             # modifications — массив строк
#             modifications = hypothesis.modifications  # список строк
#
#             # created_at — конвертируем в datetime
#             from datetime import datetime
#             created_at = datetime.fromtimestamp(hypothesis.created_at) if hasattr(hypothesis, 'created_at') else datetime.now()
#             updated_at = datetime.now()
#
#             # test_results и metadata — JSONB
#             test_results_json = json.dumps(hypothesis.test_results)
#             metadata_json = json.dumps(hypothesis.metadata)
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.hypotheses
#                 (id, task_description, source_combination_id, modifications,
#                  description, predicted_score, actual_score, status,
#                  test_results, metadata, created_at, updated_at)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
#                 ON CONFLICT (id) DO UPDATE SET
#                     task_description = EXCLUDED.task_description,
#                     source_combination_id = EXCLUDED.source_combination_id,
#                     modifications = EXCLUDED.modifications,
#                     description = EXCLUDED.description,
#                     predicted_score = EXCLUDED.predicted_score,
#                     actual_score = EXCLUDED.actual_score,
#                     status = EXCLUDED.status,
#                     test_results = EXCLUDED.test_results,
#                     metadata = EXCLUDED.metadata,
#                     updated_at = EXCLUDED.updated_at
#             """, (
#                 hypothesis.id,
#                 hypothesis.task_description,
#                 hypothesis.source_combination.id,
#                 modifications,          # ← список строк (массив)
#                 hypothesis.description,
#                 hypothesis.predicted_score,
#                 hypothesis.actual_score,
#                 hypothesis.status.value,
#                 test_results_json,
#                 metadata_json,
#                 created_at,
#                 updated_at
#             ))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#
#         except Exception as e:
#             print(f"❌ Ошибка сохранения гипотезы {hypothesis.id}: {e}")
#             return False
#
#
#     def _get_connection(self):
#         """
#         Возвращает соединение с БД.
#         """
#         return psycopg2.connect(**self.conn_params)
#
#     def _to_json(self, data):
#         """
#         Преобразует данные в JSON.
#         """
#         return json.dumps(data) if data is not None else None
#
#     def _from_json(self, data):
#         """
#         Преобразует JSON в объект Python.
#         """
#         return json.loads(data) if data else {}
#
#     # ============================================================
#     # УЗЛЫ
#     # ============================================================
#
#     def save_node(self, node: KnowledgeNode) -> bool:
#         """
#         Сохраняет узел в БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.knowledge_nodes
#                 (id, name, node_type, properties, description, embedding, parameters, metadata)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (id) DO UPDATE SET
#                     name = EXCLUDED.name,
#                     node_type = EXCLUDED.node_type,
#                     properties = EXCLUDED.properties,
#                     description = EXCLUDED.description,
#                     embedding = EXCLUDED.embedding,
#                     parameters = EXCLUDED.parameters,
#                     metadata = EXCLUDED.metadata,
#                     updated_at = CURRENT_TIMESTAMP
#             """, (
#                 node.id,
#                 node.name,
#                 node.node_type,
#                 node.properties,
#                 node.description,
#                 node.embedding.tolist() if hasattr(node.embedding, 'tolist') else node.embedding,
#                 self._to_json(node.parameters),
#                 self._to_json(node.metadata)
#
#             ))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка сохранения узла {node.id}: {e}")
#             return False
#
#     def load_node(self, node_id: str) -> Optional[KnowledgeNode]:
#         """
#         Загружает узел из БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT id, name, node_type, properties, description,
#                        embedding, parameters, metadata
#                 FROM {self.schema}.knowledge_nodes
#                 WHERE id = %s
#             """, (node_id,))
#
#             row = cur.fetchone()
#             cur.close()
#             conn.close()
#
#             if not row:
#                 return None
#
#             node = KnowledgeNode(
#                 id=row['id'],
#                 name=row['name'],
#                 node_type=row['node_type'],
#                 properties=row['properties'] if row['properties'] else [],
#                 description=row['description'] or '',
#                 embedding=np.array(row['embedding']) if row['embedding'] else None
#             )
#
#             node.parameters = self._from_json(row['parameters'])
#             node.metadata = self._from_json(row['metadata'])
#
#             # Загружаем связи
#             node.connections = self._get_node_connections(node_id)
#
#             return node
#         except Exception as e:
#             print(f"❌ Ошибка загрузки узла {node_id}: {e}")
#             return None
#
#     def load_nodes_by_properties(self, properties: List[str], limit: int = 100) -> List[KnowledgeNode]:
#         """
#         Загружает узлы по свойствам.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             # Используем оператор @> для проверки наличия всех свойств
#             cur.execute(f"""
#                 SELECT id, name, node_type, properties, description,
#                        embedding, parameters, metadata
#                 FROM {self.schema}.knowledge_nodes
#                 WHERE properties @> %s::text[]
#                 LIMIT %s
#             """, (properties, limit))
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             nodes = []
#             for row in rows:
#                 node = KnowledgeNode(
#                     id=row['id'],
#                     name=row['name'],
#                     node_type=row['node_type'],
#                     properties=row['properties'] if row['properties'] else [],
#                     description=row['description'] or '',
#                     embedding=np.array(row['embedding']) if row['embedding'] else None
#                 )
#                 node.parameters = self._from_json(row['parameters'])
#                 node.metadata = self._from_json(row['metadata'])
#                 nodes.append(node)
#
#             return nodes
#         except Exception as e:
#             print(f"❌ Ошибка загрузки узлов по свойствам: {e}")
#             return []
#
#     def _get_node_connections(self, node_id: str) -> List[str]:
#         """
#         Загружает связи узла.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 SELECT target_id FROM {self.schema}.knowledge_edges
#                 WHERE source_id = %s
#                 UNION
#                 SELECT source_id FROM {self.schema}.knowledge_edges
#                 WHERE target_id = %s
#             """, (node_id, node_id))
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             return [row[0] for row in rows]
#         except Exception:
#             return []
#
#     # ============================================================
#     # РЕБРА
#     # ============================================================
#
#     def save_edge(self, edge: KnowledgeEdge) -> bool:
#         """
#         Сохраняет ребро в БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.knowledge_edges
#                 (id, source_id, target_id, edge_type, weight, description, metadata)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (id) DO UPDATE SET
#                     source_id = EXCLUDED.source_id,
#                     target_id = EXCLUDED.target_id,
#                     edge_type = EXCLUDED.edge_type,
#                     weight = EXCLUDED.weight,
#                     description = EXCLUDED.description,
#                     metadata = EXCLUDED.metadata,
#                     updated_at = CURRENT_TIMESTAMP
#             """, (
#                 edge.id,
#                 edge.source_id,
#                 edge.target_id,
#                 edge.edge_type.value,
#                 edge.weight,
#                 edge.description,
#                 self._to_json(edge.metadata)
#             ))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка сохранения ребра {edge.id}: {e}")
#             return False
#
#     def load_edges_from_node(self, node_id: str) -> List[KnowledgeEdge]:
#         """
#         Загружает все ребра, связанные с узлом.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT id, source_id, target_id, edge_type, weight, description, metadata
#                 FROM {self.schema}.knowledge_edges
#                 WHERE source_id = %s OR target_id = %s
#             """, (node_id, node_id))
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             edges = []
#             for row in rows:
#                 edge = KnowledgeEdge(
#                     id=row['id'],
#                     source_id=row['source_id'],
#                     target_id=row['target_id'],
#                     edge_type=EdgeType(row['edge_type']),
#                     weight=row['weight'],
#                     description=row['description'] or '',
#                     metadata=self._from_json(row['metadata'])
#                 )
#                 edges.append(edge)
#
#             return edges
#         except Exception as e:
#             print(f"❌ Ошибка загрузки ребер для узла {node_id}: {e}")
#             return []
#
#     # ============================================================
#     # ИНДИВИДУАЛЬНЫЙ ГРАФ
#     # ============================================================
#
#     def init_individual_graph(self, bot_id: str, name: str, description: str = "") -> bool:
#         """
#         Инициализирует индивидуальный граф для бота.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.individual_knowledge_graphs
#                 (bot_id, name, description)
#                 VALUES (%s, %s, %s)
#                 ON CONFLICT (bot_id) DO UPDATE SET
#                     name = EXCLUDED.name,
#                     description = EXCLUDED.description,
#                     updated_at = CURRENT_TIMESTAMP
#             """, (bot_id, name, description))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка инициализации ИГЗ для {bot_id}: {e}")
#             return False
#
#     def sync_node_to_individual(self, bot_id: str, node_id: str, confidence: float = 0.5) -> bool:
#         """
#         Синхронизирует узел с индивидуальным графом бота.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 INSERT INTO {self.schema}.individual_node_links
#                 (bot_id, node_id, confidence, last_used)
#                 VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
#                 ON CONFLICT (bot_id, node_id) DO UPDATE SET
#                     confidence = EXCLUDED.confidence,
#                     last_used = CURRENT_TIMESTAMP
#             """, (bot_id, node_id, confidence))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка синхронизации узла {node_id} для {bot_id}: {e}")
#             return False
#
#     def load_individual_nodes(self, bot_id: str) -> List[KnowledgeNode]:
#         """
#         Загружает все узлы, синхронизированные с ботом.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT n.id, n.name, n.node_type, n.properties, n.description,
#                        n.embedding, n.parameters, n.metadata
#                 FROM {self.schema}.knowledge_nodes n
#                 JOIN {self.schema}.individual_node_links l ON n.id = l.node_id
#                 WHERE l.bot_id = %s
#                 ORDER BY l.confidence DESC
#             """, (bot_id,))
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             nodes = []
#             for row in rows:
#                 node = KnowledgeNode(
#                     id=row['id'],
#                     name=row['name'],
#                     node_type=row['node_type'],
#                     properties=row['properties'] if row['properties'] else [],
#                     description=row['description'] or '',
#                     embedding=np.array(row['embedding']) if row['embedding'] else None
#                 )
#                 node.parameters = self._from_json(row['parameters'])
#                 node.metadata = self._from_json(row['metadata'])
#                 nodes.append(node)
#
#             return nodes
#         except Exception as e:
#             print(f"❌ Ошибка загрузки индивидуальных узлов для {bot_id}: {e}")
#             return []
#
#     def load_all_nodes(self) -> List:
#         """
#         Загружает все узлы из БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT id, name, node_type, properties, description, parameters, metadata
#                 FROM {self.schema}.knowledge_nodes
#             """)
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             from core.knowledge.knowledge_node import KnowledgeNode
#
#             nodes = []
#             for row in rows:
#                 node = KnowledgeNode(
#                     id=row['id'],
#                     name=row['name'],
#                     node_type=row['node_type'],
#                     properties=row['properties'] if row['properties'] else [],
#                     description=row['description'] or ''
#                 )
#
#                 # Восстанавливаем параметры и метаданные
#                 # Они уже могут быть словарями (если из JSONB), или строками
#                 if row.get('parameters'):
#                     if isinstance(row['parameters'], dict):
#                         node.parameters = row['parameters']
#                     else:
#                         node.parameters = self._from_json(row['parameters'])
#
#                 if row.get('metadata'):
#                     if isinstance(row['metadata'], dict):
#                         node.metadata = row['metadata']
#                     else:
#                         node.metadata = self._from_json(row['metadata'])
#
#                 nodes.append(node)
#
#             return nodes
#         except Exception as e:
#             print(f"❌ Ошибка загрузки узлов: {e}")
#             return []
#
#     def save_mental_model(self, model) -> bool:
#         """
#         Сохраняет ментальную модель в БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             # Генерируем UUID, если id не является валидным UUID
#             import uuid
#             try:
#                 uuid.UUID(model.id)
#                 model_id = model.id
#             except (ValueError, AttributeError, TypeError):
#                 model_id = str(uuid.uuid4())
#                 print(f"   ⚠️ ID {model.id} не является UUID, заменён на {model_id}")
#
#             # Подготовка данных
#             model_name = model.name if hasattr(model, 'name') else "Ментальная модель"
#             model_type = getattr(model, 'model_type', 'mental_model')
#
#             # Свойства - JSONB
#             if hasattr(model, 'properties'):
#                 if isinstance(model.properties, dict):
#                     properties_json = json.dumps(model.properties)
#                 else:
#                     properties_json = json.dumps({'props': model.properties})
#             else:
#                 properties_json = json.dumps({})
#
#             # Последовательность - ТЕКСТОВЫЙ МАССИВ (text[])
#             if hasattr(model, 'sequence') and model.sequence:
#                 if isinstance(model.sequence, list):
#                     sequence_array = '{' + ','.join(f'"{s}"' for s in model.sequence) + '}'
#                 else:
#                     sequence_array = '{}'
#             else:
#                 sequence_array = '{}'
#
#             # Эмбеддинг - МАССИВ DOUBLE PRECISION (double precision[])
#             if hasattr(model, 'embedding') and model.embedding is not None:
#                 if hasattr(model.embedding, 'tolist'):
#                     embedding_list = model.embedding.tolist()
#                 elif isinstance(model.embedding, list):
#                     embedding_list = model.embedding
#                 else:
#                     embedding_list = list(model.embedding)
#
#                 # Формат для double precision[]: {1.0, 0.5, 0.2}
#                 embedding_array = '{' + ','.join(str(float(x)) for x in embedding_list) + '}'
#             else:
#                 # Пустой массив
#                 embedding_array = '{}'
#
#             # Метаданные - JSONB
#             if hasattr(model, 'metadata'):
#                 metadata_json = json.dumps(model.metadata)
#             else:
#                 metadata_json = json.dumps({})
#
#             # Описание
#             description = getattr(model, 'description', model_name)
#
#             # Выполняем запрос
#             cur.execute("""
#                 INSERT INTO agi_evolution.mental_models
#                 (id, name, model_type, properties, sequence, embedding, description, metadata, created_at)
#                 VALUES (%s, %s, %s, %s::jsonb, %s::text[], %s::double precision[], %s, %s::jsonb, %s)
#                 ON CONFLICT (id) DO UPDATE SET
#                     name = EXCLUDED.name,
#                     model_type = EXCLUDED.model_type,
#                     properties = EXCLUDED.properties,
#                     sequence = EXCLUDED.sequence,
#                     embedding = EXCLUDED.embedding,
#                     description = EXCLUDED.description,
#                     metadata = EXCLUDED.metadata,
#                     updated_at = CURRENT_TIMESTAMP
#             """, (
#                 model_id,
#                 model_name,
#                 model_type,
#                 properties_json,  # ::jsonb
#                 sequence_array,  # ::text[]
#                 embedding_array,  # ::double precision[]  ← ИСПРАВЛЕНО!
#                 description,
#                 metadata_json,  # ::jsonb
#                 datetime.now()
#             ))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#
#             print(f"✅ Ментальная модель сохранена: {model_id} ({model_name})")
#             return True
#
#         except Exception as e:
#             print(f"❌ Ошибка сохранения ментальной модели: {e}")
#             import traceback
#             traceback.print_exc()
#             return False
#
#
#     def load_all_mental_models(self) -> List:
#         """Загружает все ментальные модели из БД."""
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute("""
#                 SELECT id, name, model_type, properties, sequence, embedding,
#                        description, metadata, created_at
#                 FROM agi_evolution.mental_models
#                 ORDER BY created_at DESC
#             """)
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             from core.thinking.models import MentalModel
#
#             models = []
#             for row in rows:
#                 # Восстанавливаем свойства
#                 if row.get('properties'):
#                     if isinstance(row['properties'], dict):
#                         properties = row['properties']
#                     else:
#                         try:
#                             properties = json.loads(row['properties'])
#                         except:
#                             properties = {}
#                 else:
#                     properties = {}
#
#                 # Восстанавливаем последовательность
#                 if row.get('sequence'):
#                     if isinstance(row['sequence'], list):
#                         sequence = row['sequence']
#                     else:
#                         try:
#                             sequence = json.loads(row['sequence'])
#                         except:
#                             sequence = []
#                 else:
#                     sequence = []
#
#                 # Восстанавливаем эмбеддинг
#                 if row.get('embedding'):
#                     if isinstance(row['embedding'], list):
#                         embedding = row['embedding']
#                     else:
#                         try:
#                             embedding = json.loads(row['embedding'])
#                         except:
#                             embedding = None
#                 else:
#                     embedding = None
#
#                 # Восстанавливаем метаданные
#                 if row.get('metadata'):
#                     if isinstance(row['metadata'], dict):
#                         metadata = row['metadata']
#                     else:
#                         try:
#                             metadata = json.loads(row['metadata'])
#                         except:
#                             metadata = {}
#                 else:
#                     metadata = {}
#
#                 model = MentalModel(
#                     id=row['id'],
#                     name=row.get('name', ''),
#                     sequence=sequence,
#                     embedding=embedding,
#                     properties=properties,
#                     metadata=metadata,
#                     created_at=row['created_at'].timestamp() if row.get('created_at') else time.time()
#                 )
#                 model.model_type = row.get('model_type', 'mental_model')
#                 models.append(model)
#
#             return models
#         except Exception as e:
#             print(f"❌ Ошибка загрузки ментальных моделей: {e}")
#             return []
#
#
#
#     def get_mental_model(self, model_id: str):
#         """Загружает ментальную модель по ID."""
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT * FROM agi_evolution.mental_models
#                 WHERE id = %s
#             """, (model_id,))
#
#             row = cur.fetchone()
#             cur.close()
#             conn.close()
#
#             if not row:
#                 return None
#
#             from core.thinking.models import MentalModel
#
#             return MentalModel(
#                 id=row['id'],
#                 name=row.get('name', ''),
#                 sequence=row.get('sequence', []),
#                 embedding=row.get('embedding'),
#                 properties=row.get('properties', {}) if isinstance(row.get('properties'), dict) else row.get('properties', []),
#                 metadata=row.get('metadata', {}),
#                 created_at=row.get('created_at').timestamp() if row.get('created_at') else time.time()
#             )
#         except Exception as e:
#             print(f"❌ Ошибка загрузки ментальной модели {model_id}: {e}")
#             return None
#
#     def delete_mental_model(self, model_id: str) -> bool:
#         """Удаляет ментальную модель."""
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor()
#
#             cur.execute(f"""
#                 DELETE FROM agi_evolution.mental_models
#                 WHERE id = %s
#             """, (model_id,))
#
#             conn.commit()
#             cur.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print(f"❌ Ошибка удаления ментальной модели {model_id}: {e}")
#             return False
#
#
#     def load_all_edges(self) -> List:
#         """
#         Загружает все ребра из БД.
#         """
#         try:
#             conn = self._get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#
#             cur.execute(f"""
#                 SELECT id, source_id, target_id, edge_type, weight, description, metadata
#                 FROM {self.schema}.knowledge_edges
#             """)
#
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#
#             from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
#
#             edges = []
#             for row in rows:
#                 edge = KnowledgeEdge(
#                     id=row['id'],
#                     source_id=row['source_id'],
#                     target_id=row['target_id'],
#                     edge_type=EdgeType(row['edge_type']),
#                     weight=row['weight'] or 0.5,
#                     description=row['description'] or '',
#                     metadata=self._from_json(row['metadata']) if row.get('metadata') else {}
#                 )
#                 edges.append(edge)
#
#             return edges
#         except Exception as e:
#             print(f"❌ Ошибка загрузки ребер: {e}")
#             return []