-- agi_evolution.knowledge_edges definition

-- Drop table

-- DROP TABLE agi_evolution.knowledge_edges;

CREATE TABLE agi_evolution.knowledge_edges (
	id varchar(50) NOT NULL,
	source_id varchar(50) NULL,
	target_id varchar(50) NULL,
	edge_type varchar(50) NOT NULL,
	weight float8 DEFAULT 0.5 NULL,
	description text NULL,
	metadata jsonb DEFAULT '{}'::jsonb NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT knowledge_edges_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_knowledge_edges_source ON agi_evolution.knowledge_edges USING btree (source_id);
CREATE INDEX idx_knowledge_edges_target ON agi_evolution.knowledge_edges USING btree (target_id);
CREATE INDEX idx_knowledge_edges_type ON agi_evolution.knowledge_edges USING btree (edge_type);


-- agi_evolution.knowledge_edges foreign keys

ALTER TABLE agi_evolution.knowledge_edges ADD CONSTRAINT knowledge_edges_source_id_fkey FOREIGN KEY (source_id) REFERENCES agi_evolution.knowledge_nodes(id);
ALTER TABLE agi_evolution.knowledge_edges ADD CONSTRAINT knowledge_edges_target_id_fkey FOREIGN KEY (target_id) REFERENCES agi_evolution.knowledge_nodes(id);