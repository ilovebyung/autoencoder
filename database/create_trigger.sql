CREATE TABLE setting_log (
	id int,
	old_name text,
	new_name text,
	old_value numeric,
	new_value numeric,
	date_created DATETIME
);

CREATE TRIGGER log_setting_after_update 
   AFTER UPDATE ON setting
   WHEN old.name <> new.name
        OR old.value <> new.value
BEGIN
	INSERT INTO setting_log (
		old_id,
		new_id,
		old_name,
		new_name,
		old_value,
		new_value,
		date_created
	)
VALUES
	(
		old.id,
		new.id,
		old.name,
		new.name,
		old.value,
		new.value,
		DATETIME('NOW')
	) ;
END;

DROP TRIGGER log_contact_after_update;