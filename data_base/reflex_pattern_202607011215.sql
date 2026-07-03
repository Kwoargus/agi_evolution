INSERT INTO agi_prototype.reflex_pattern (id,signal_type,signal_threshold,"action",success_rate,usage_count,last_used,sense_type) VALUES
	 (0,'temperature_sense',50.0,'move_on',NULL,NULL,NULL,'touch'),
	 (1,'cutting_edge_sence',0.9,'move_on',NULL,NULL,NULL,'touch'),
	 (3,'predator_roar',0.1,'move_on',NULL,NULL,NULL,'sound'),
	 (4,'food_smell',0.2,'grab',NULL,NULL,NULL,'smell'),
	 (2,'predator_smell',1.0,'move_on',NULL,NULL,NULL,'smell'),
	 (5,'loud crash',0.99,'look',NULL,NULL,NULL,'sound'),
	 (6,'bright_flash',0.99,'look',NULL,NULL,NULL,'vision');
