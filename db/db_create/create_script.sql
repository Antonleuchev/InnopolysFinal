BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "result_history" (
	"id"	INTEGER,
	"class"	INTEGER,
	"usage_id"	INTEGER,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "usage_history" (
	"id"	INTEGER UNIQUE,
	"date"	TEXT,
	"time"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);
COMMIT;
