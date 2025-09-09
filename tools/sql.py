import sqlite3
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List


def get_db_connection():
    conn = sqlite3.connect("../db/db.sqlitedb")
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


def describe_tables(table_names):
    conn = get_db_connection()
    cursor = conn.cursor()
    tables = ", ".join("'" + table + "'" for table in table_names)
    rows = cursor.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});"
    )
    return "\n".join([row[0] for row in rows if row[0] is not None])


class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, return the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema,
)


class RunQueryArgsSchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=execute_query,
    args_schema=RunQueryArgsSchema,
)
