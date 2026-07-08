INSERT INTO agi_evolution.reflex_pattern (sense_type,signal_type,signal_threshold,"action",success_rate,usage_count,last_used) VALUES
	 ('smell','food_smell',0.0,'grab',NULL,NULL,NULL),
	 ('smell','predator_smel',0.0,'move_on',NULL,NULL,NULL),
	 ('visual','bright_flash',1.0,'move_on',NULL,NULL,NULL),
	 ('sound','loud_crash',1.0,'move_on',NULL,NULL,NULL),
	 ('touch','temperature_sense',50.0,'move_on',NULL,NULL,NULL),
	 ('sound','predator_roar',1.0,'move_on',NULL,NULL,NULL);
