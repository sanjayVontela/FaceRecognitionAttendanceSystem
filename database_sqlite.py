import sqlite3
import datetime

class Database:

    def __init__(self):
        self.conn=sqlite3.connect('students.db')
        self.cur=self.conn.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS students1 (rollno text PRIMARY KEY, name text)')
        try:
            self.cur.execute("insert into students1 values ('18311A1230','anirudh')")
            self.cur.execute("insert into students1 values ('18311A1253','amar')")
            self.cur.execute("insert into students1 values ('18311A1256','manish')")
            self.cur.execute("insert into students1 values ('18311A1258','sanjay')")
            
        except Exception:
            pass
        self.conn.commit()

    def update(self):
        date = datetime.datetime.now().date()
        date = str(date)
        try:
            self.cur.execute('alter table students1 add "%s" text'%(date))
        except Exception:
            pass
        self.conn.commit()
        
    def attendance(self,time,name):
        time = str(time)
        date = str(datetime.datetime.now().date())
        self.cur.execute('update students1 set "%s" = \'%s\' where name = \'%s\''%(date,time,name))
        self.conn.commit()
    def insert(self,name,rollno):
        try:
            self.cur.execute("INSERT INTO students1 VALUES (?,?)",(rollno,name))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def view(self):
        self.cur.execute("SELECT * FROM students1")
        rows=self.cur.fetchall()
        return rows

    def delete(self,name):
        self.cur.execute("DELETE FROM students1 WHERE name=?",(name,))
        self.conn.commit()

    
    def __del__(self):
        self.conn.close()

database = Database()

