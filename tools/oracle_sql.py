# Oracle connection details are loaded from environment variables (.env)
import os
import oracledb
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List
from dotenv import load_dotenv


load_dotenv()

ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE")
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
DSN = f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"


def get_db_connection():
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=DSN)
    return conn


def list_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM user_tables")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return "\n".join([row[0] for row in rows if row[0] is not None])


def execute_query(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        if cursor.description:
            result = cursor.fetchall()
        else:
            result = "Query executed."
        cursor.close()
        conn.close()
        return result
    except Exception as err:
        cursor.close()
        conn.close()
        return f"The following error occurred: {str(err)}"


def describe_tables(table_names):
    conn = get_db_connection()
    cursor = conn.cursor()
    output = ""
    for table in table_names:
        cursor.execute(
            f"SELECT column_name, data_type FROM user_tab_columns WHERE table_name = '{table.upper()}'"
        )
        rows = cursor.fetchall()
        output += f"Table: {table}\n"
        if rows:
            for row in rows:
                output += f"  {row[0]}: {row[1]}\n"
        else:
            output += "  No columns found.\n"
    cursor.close()
    conn.close()
    return output


class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, return the schema of those tables (Oracle)",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema,
)


class RunQueryArgsSchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_oracle_query",
    description="Run an Oracle SQL query",
    func=execute_query,
    args_schema=RunQueryArgsSchema,
)
