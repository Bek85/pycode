import sqlite3
from langchain.tools import Tool


def get_db_connection():
    conn = sqlite3.connect("db/db.sqlite")
    return conn


def execute_query(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=execute_query,
)
