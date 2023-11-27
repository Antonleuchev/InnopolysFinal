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
        
   