import sqlite3
import time
import pandas as pd
import os


class BasicDB:
    def __init__(self, path, connect=True):
        self.path = path
        self.conn = None
        self.cursor = None
        if connect:
            self.connect()

    def connect(self):
        while True:
            try:
                self.conn = sqlite3.connect(self.path)
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
                break  # Database is no longer locked
            except sqlite3.OperationalError:
                print("Database is locked. Waiting...")
                time.sleep(10 * 60)  # Wait for 10 minutes before trying again

    def close(self):
        self.conn.close()

    def sql_to_df(self, sql):
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return pd.DataFrame(results, columns=[d[0] for d in self.cursor.description])

    def exists_table(self, table):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return len(self.cursor.fetchall()) > 0

    def exists_column(self, table, column, add=None):
        self.cursor.execute(f"PRAGMA table_info({table})")
        columns = [column[1] for column in self.cursor.fetchall()]
        if column not in columns:
            if add is not None:
                self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {add}")
            return False
        return True
