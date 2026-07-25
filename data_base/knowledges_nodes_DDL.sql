-- agi_evolution.knowledge_nodes definition

-- Drop table

-- DROP TABLE agi_evolution.knowledge_nodes;

CREATE TABLE agi_evolution.knowledge_nodes (
	id varchar(50) NOT NULL,
	"name" varchar(200) NOT NULL,
	node_type varchar(50) NOT NULL,
	properties _text DEFAULT '{}'::text[] NULL,
	description text NULL,
	embedding _float8 NULL,
	parameters jsonb DEFAULT '{}'::jsonb NULL,
	metadata jsonb DEFAULT '{}'::jsonb NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT knowledge_nodes_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_knowledge_nodes_properties ON agi_evolution.knowledge_nodes USING gin (properties);
CREATE INDEX idx_knowledge_nodes_type ON agi_evolution.knowledge_nodes USING btree (node_type);

