from db.db import *
import datetime

db_manager = DbManager('db\\db.db')

x = datetime.datetime.now()
print(x)
date, time = str(x).split(' ')
print(date, time)
db_manager.insert_history(date, time)

print()
db_manager.get_all_history()