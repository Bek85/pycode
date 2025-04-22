import sqlite3
from langchain.tools import Tool


def get_db_connection():
    conn = sqlite3.connect("db/db.sqlite")
    return conn


def list_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = cursor.fetchall()
    return "\n".join([row[0] for row in rows if row[0] is not None])


def execute_query(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=execute_query,
)
