# core/knowledge/adapters/conceptnet_adapter.py
from core.knowledge import KnowledgeNode, KnowledgeEdge


class ConceptNetAdapter:
    """
    Адаптер для преобразования ConceptNet в ваш ГЗ.
    """

    def __init__(self):
        self.type_map = {
            'PhysicalObject': 'object',
            'Device': 'system',
            'Process': 'process',
            'Material': 'material',
            'Concept': 'concept'
        }

        self.edge_map = {
            'IsA': 'instance_of',
            'PartOf': 'part_of',
            'HasProperty': 'has_property',
            'UsedFor': 'used_for',
            'Causes': 'causes',
            'CapableOf': 'has_function'
        }

    def convert_node(self, conceptnet_node):
        """
        Преобразует узел ConceptNet в ваш KnowledgeNode.
        """
        node = KnowledgeNode(
            id=f"ext_{conceptnet_node.id}",
            name=conceptnet_node.label,
            node_type=self.type_map.get(conceptnet_node.type, 'concept'),
            properties=self._extract_properties(conceptnet_node),
            description=conceptnet_node.description or ""
        )
        return node

    def convert_edge(self, conceptnet_edge):
        """
        Преобразует ребро ConceptNet в ваше KnowledgeEdge.
        """
        edge = KnowledgeEdge(
            id=f"ext_{conceptnet_edge.id}",
            source_id=f"ext_{conceptnet_edge.source}",
            target_id=f"ext_{conceptnet_edge.target}",
            edge_type=self.edge_map.get(conceptnet_edge.relation, 'related_to'),
            weight=conceptnet_edge.weight or 0.5,
            description=conceptnet_edge.description or ""
        )
        return edge