import sqlite3
from models import *

class DbManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        
    def get_all_history(self):
        for row in self.cur.execute("SELECT * FROM usage_history ORDER BY id"):
            print(row)
            
    def insert_history(self, date, time):
        self.cur.execute(f"INSERT INTO usage_history(date, time) VALUES (?, ?)", (date, time))
        self.con.commit()
        
    def insert_result(self, original_file_path, processed_file_path, usage_id):
        self.cur.execute(f"INSERT INTO result_history(original_path, processed_path, usage_id) VALUES (?, ?, ?)", (original_file_path, processed_file_path, usage_id))
        self.con.commit()
        
    def get_all_result_with_usage(self):
        rows = self.cur.execute(f"SELECT result_history.original_path, result_history.processed_path, usage_history.date, usage_history.time FROM result_history INNER JOIN usage_history ON usage_history.id == result_history.usage_id")
        result = []
        for row in rows:
            result.append(Result(row[0], row[1], row[2], row[3]))
        return result
        
    def get_last_inserted_row_id(self):
        return self.cur.lastrowid