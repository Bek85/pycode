import os
import oracledb
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool

load_dotenv()

ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE")
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
DSN = f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"


def get_db_connection():
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=DSN)


def list_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM user_tables")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return "\n".join([row[0] for row in rows if row[0] is not None])


# --- TOOL 1: Describe Tables ---
class DescribeTablesArgs(BaseModel):
    table_names: List[str] = Field(description="List of Oracle table names")


@tool(args_schema=DescribeTablesArgs)
def describe_tables(table_names: List[str]) -> str:
    """Given a list of Oracle table names, return their column names and data types."""
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


# --- TOOL 2: Run Oracle Query ---
class RunQueryArgs(BaseModel):
    query: str = Field(description="The Oracle SQL query to execute")


@tool(args_schema=RunQueryArgs)
def run_query(query: str) -> str:
    """Executes an Oracle SQL query and returns the result."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        if cursor.description:
            result = cursor.fetchall()
            return str(result)
        else:
            return "Query executed successfully."
    except Exception as err:
        return f"Error occurred: {str(err)}"
    finally:
        cursor.close()
        conn.close()


__all__ = ["run_query", "describe_tables", "list_tables"]
