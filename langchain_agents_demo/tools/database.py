"""
Database tools with comprehensive error handling and logging.

This module provides enhanced database tools with proper connection management,
error handling, and performance monitoring.
"""

import sqlite3
import time
from typing import List, Optional, Union, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from langchain.tools import Tool
from pydantic.v1 import BaseModel

from ..config import AppConfig
from ..utils import AgentLogger, log_function_call, log_error, log_performance


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class DatabaseConnectionManager:
    """
    Database connection manager with connection pooling and error handling.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = AgentLogger.get_logger(__name__)
        self.db_path = Path(config.database.get_absolute_path())
        
        # Ensure database exists
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure the database file exists."""
        if not self.db_path.exists():
            self.logger.warning(f"Database file not found at {self.db_path}")
            # Create parent directories if they don't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # Note: We don't create an empty database here as it should be provided
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection with proper error handling.
        
        Yields:
            sqlite3.Connection: Database connection
            
        Raises:
            DatabaseError: If connection fails
        """
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.config.database.connection_timeout
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except sqlite3.Error as e:
            log_error(e, "Database connection failed")
            raise DatabaseError(f"Database connection failed: {str(e)}") from e
        finally:
            if conn:
                conn.close()


class DatabaseService:
    """
    High-level database service with business logic and error handling.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.connection_manager = DatabaseConnectionManager(config)
        self.logger = AgentLogger.get_logger(__name__)
    
    def list_tables(self) -> str:
        """
        List all tables in the database.
        
        Returns:
            String with table names separated by newlines
            
        Raises:
            DatabaseError: If operation fails
        """
        start_time = time.time()
        
        try:
            log_function_call("list_tables")
            
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                rows = cursor.fetchall()
                
                result = "\n".join([row["name"] for row in rows if row["name"] is not None])
                
                log_performance("list_tables", time.time() - start_time, {"tables_count": len(rows)})
                log_function_call("list_tables", result=f"Found {len(rows)} tables")
                
                return result
                
        except Exception as e:
            log_error(e, "Failed to list tables")
            raise DatabaseError(f"Failed to list tables: {str(e)}") from e
    
    def describe_tables(self, table_names: List[str]) -> str:
        """
        Get schema information for specified tables.
        
        Args:
            table_names: List of table names to describe
            
        Returns:
            String containing table schemas
            
        Raises:
            DatabaseError: If operation fails
        """
        start_time = time.time()
        
        try:
            log_function_call("describe_tables", {"table_names": table_names})
            
            if not table_names:
                return "No tables specified"
            
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Escape table names to prevent SQL injection
                tables_param = ", ".join("?" for _ in table_names)
                query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables_param});"
                
                cursor.execute(query, table_names)
                rows = cursor.fetchall()
                
                result = "\n\n".join([row["sql"] for row in rows if row["sql"] is not None])
                
                log_performance("describe_tables", time.time() - start_time, {
                    "requested_tables": len(table_names),
                    "found_tables": len(rows)
                })
                log_function_call("describe_tables", result=f"Described {len(rows)} tables")
                
                return result
                
        except Exception as e:
            log_error(e, f"Failed to describe tables: {table_names}")
            raise DatabaseError(f"Failed to describe tables: {str(e)}") from e
    
    def _optimize_query_for_large_results(self, query: str) -> str:
        """
        Automatically optimize queries that might return large result sets.

        Args:
            query: Original SQL query

        Returns:
            Optimized query with appropriate limits
        """
        query_upper = query.upper().strip()

        # If query already has LIMIT, leave it alone
        if 'LIMIT' in query_upper:
            return query

        # For SELECT queries without LIMIT, add safety limits
        if query_upper.startswith('SELECT'):
            # Check if this looks like an aggregation query
            has_aggregation = any(keyword in query_upper for keyword in
                                ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'GROUP BY'])

            # If it's not an aggregation and doesn't have LIMIT, add one
            if not has_aggregation:
                # Add LIMIT 1000 as safety
                query = query.rstrip(';') + ' LIMIT 1000'
                self.logger.info(f"Added safety LIMIT to query: {query[:50]}...")

        return query

    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], str]:
        """
        Execute a SQL query with proper error handling.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as list of dictionaries or error message
        """
        start_time = time.time()
        
        try:
            # Optimize query to prevent large result sets
            original_query = query
            query = self._optimize_query_for_large_results(query)

            log_function_call("execute_query", {"query_preview": query[:100]})

            # Basic query validation
            if not query.strip():
                return "Empty query provided"
            
            # Prevent dangerous operations (basic protection)
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT']
            query_upper = query.upper().strip()
            
            for keyword in dangerous_keywords:
                if query_upper.startswith(keyword):
                    error_msg = f"Query type '{keyword}' is not allowed for security reasons"
                    self.logger.warning(f"Blocked dangerous query: {query[:50]}...")
                    return error_msg
            
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)

                if query_upper.startswith('SELECT'):
                    rows = cursor.fetchall()
                    # Convert sqlite3.Row objects to dictionaries
                    result = [dict(row) for row in rows]

                    # Truncate results if too large to prevent context overflow
                    MAX_ROWS = 100  # Reasonable limit for context window
                    original_count = len(result)

                    if len(result) > MAX_ROWS:
                        result = result[:MAX_ROWS]
                        # Add a message indicating truncation
                        truncation_info = {
                            "_truncated": True,
                            "_total_rows": original_count,
                            "_showing_rows": len(result),
                            "_message": f"Results truncated to {MAX_ROWS} rows out of {original_count} total rows"
                        }
                        result.append(truncation_info)

                    log_performance("execute_query", time.time() - start_time, {
                        "query_type": "SELECT",
                        "rows_returned": len(result),
                        "original_count": original_count,
                        "truncated": original_count > MAX_ROWS
                    })
                    log_function_call("execute_query", result=f"Returned {len(result)} rows (original: {original_count})")

                    return result
                else:
                    # For non-SELECT queries (though we block most above)
                    conn.commit()
                    result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
                    
                    log_performance("execute_query", time.time() - start_time, {
                        "query_type": "OTHER",
                        "rows_affected": cursor.rowcount
                    })
                    log_function_call("execute_query", result=result)
                    
                    return result
                    
        except sqlite3.Error as e:
            error_msg = f"SQL Error: {str(e)}"
            log_error(e, f"SQL query execution failed: {query[:50]}...")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            log_error(e, f"Query execution failed: {query[:50]}...")
            return error_msg


# Pydantic schemas for tool validation
class DescribeTablesArgsSchema(BaseModel):
    """Schema for describe_tables tool arguments."""
    table_names: List[str]


class RunQueryArgsSchema(BaseModel):
    """Schema for run_query tool arguments."""
    query: str


def create_database_tools(config: AppConfig) -> List[Tool]:
    """
    Create database tools with proper configuration and error handling.
    
    Args:
        config: Application configuration
        
    Returns:
        List of configured database tools
    """
    db_service = DatabaseService(config)
    
    # Describe tables tool
    describe_tables_tool = Tool.from_function(
        name="describe_tables",
        description="Given a list of table names, return the schema of those tables",
        func=db_service.describe_tables,
        args_schema=DescribeTablesArgsSchema,
    )
    
    # Run query tool
    run_query_tool = Tool.from_function(
        name="run_sqlite_query", 
        description="Run a SELECT sqlite query to retrieve data. Only SELECT queries are allowed for security.",
        func=db_service.execute_query,
        args_schema=RunQueryArgsSchema,
    )
    
    return [describe_tables_tool, run_query_tool]


def get_tables_info(config: AppConfig) -> str:
    """
    Get information about available tables.
    
    Args:
        config: Application configuration
        
    Returns:
        String containing table information
    """
    try:
        db_service = DatabaseService(config)
        return db_service.list_tables()
    except Exception as e:
        log_error(e, "Failed to get tables info")
        return "Error: Unable to retrieve table information"