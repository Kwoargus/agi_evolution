-- core/knowledge/sql/knowledge_tables.sql

-- Узлы графа знаний
CREATE TABLE IF NOT EXISTS agi_evolution.knowledge_nodes (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    node_type VARCHAR(50) NOT NULL,
    properties TEXT[] DEFAULT '{}',
    description TEXT,
    embedding FLOAT[],
    parameters JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ребра графа знаний
CREATE TABLE IF NOT EXISTS agi_evolution.knowledge_edges (
    id VARCHAR(50) PRIMARY KEY,
    source_id VARCHAR(50) REFERENCES agi_evolution.knowledge_nodes(id),
    target_id VARCHAR(50) REFERENCES agi_evolution.knowledge_nodes(id),
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 0.5,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индивидуальные графы знаний (привязка к боту)
CREATE TABLE IF NOT EXISTS agi_evolution.individual_knowledge_graphs (
    bot_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    sync_status JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Связи бота с узлами ГГЗ (синхронизация)
CREATE TABLE IF NOT EXISTS agi_evolution.individual_node_links (
    bot_id VARCHAR(50) REFERENCES agi_evolution.individual_knowledge_graphs(bot_id),
    node_id VARCHAR(50) REFERENCES agi_evolution.knowledge_nodes(id),
    confidence FLOAT DEFAULT 0.5,
    usage_count INT DEFAULT 0,
    last_used TIMESTAMP,
    PRIMARY KEY (bot_id, node_id)
);

-- Комбинации (кандидаты)
CREATE TABLE IF NOT EXISTS agi_evolution.combinations (
    id VARCHAR(50) PRIMARY KEY,
    node_ids TEXT[] DEFAULT '{}',
    edge_ids TEXT[] DEFAULT '{}',
    properties TEXT[] DEFAULT '{}',
    score FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Гипотезы
CREATE TABLE IF NOT EXISTS agi_evolution.hypotheses (
    id VARCHAR(50) PRIMARY KEY,
    task_description TEXT,
    source_combination_id VARCHAR(50) REFERENCES agi_evolution.combinations(id),
    modifications TEXT[] DEFAULT '{}',
    description TEXT,
    predicted_score FLOAT DEFAULT 0.0,
    actual_score FLOAT DEFAULT 0.0,
    status VARCHAR(30) DEFAULT 'proposed',
    test_results JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы
CREATE INDEX idx_knowledge_nodes_properties ON agi_evolution.knowledge_nodes USING GIN(properties);
CREATE INDEX idx_knowledge_nodes_type ON agi_evolution.knowledge_nodes(node_type);
CREATE INDEX idx_knowledge_edges_source ON agi_evolution.knowledge_edges(source_id);
CREATE INDEX idx_knowledge_edges_target ON agi_evolution.knowledge_edges(target_id);
CREATE INDEX idx_knowledge_edges_type ON agi_evolution.knowledge_edges(edge_type);
CREATE INDEX idx_individual_node_links_bot ON agi_evolution.individual_node_links(bot_id);
CREATE INDEX idx_combinations_properties ON agi_evolution.combinations USING GIN(properties);
CREATE INDEX idx_hypotheses_status ON agi_evolution.hypotheses(status);