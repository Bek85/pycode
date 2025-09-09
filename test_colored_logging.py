#!/usr/bin/env python3
"""
Test script to demonstrate colored logging functionality.
"""

import os
import sys
import logging

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_demos.utils.logging import AgentLogger, log_agent_action, log_performance, log_function_call
from langchain_demos.config import LoggingConfig

def test_colored_logging():
    """Test various log levels and categories with colors."""
    
    # Force colored output for demonstration
    config = LoggingConfig(colored=True, level="DEBUG")
    logger = AgentLogger.get_logger(__name__, config)
    
    print("🎨 Testing Colored Logging Output:")
    print("=" * 50)
    
    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message") 
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print()
    print("Agent Action Examples:")
    print("-" * 30)
    
    # Test agent actions
    log_agent_action("main", "started", {"mode": "demo", "version": "1.0"})
    log_agent_action("agent_api", "initialized", {"fastapi_available": True})
    log_agent_action("tool_calling", "agent_created", {"tools_count": 5, "agent_id": "demo_agent"})
    log_agent_action("tool_calling", "execute_query", {"query_preview": "SELECT * FROM users LIMIT 10"})
    
    print()
    print("Function Call Examples:")
    print("-" * 30)
    
    # Test function calls
    log_function_call("list_tables")
    log_function_call("describe_tables", {"table_names": ["users", "orders"]})
    log_function_call("execute_query", {"query_preview": "COUNT(*) FROM orders"})
    
    print()
    print("Performance Examples:")
    print("-" * 30)
    
    # Test performance logs
    log_performance("database_connection", 0.25, {"host": "localhost", "status": "success"})
    log_performance("query_execution", 1.45, {"query_type": "SELECT", "rows_returned": 150})
    log_performance("agent_initialization", 3.22, {"tables_found": 8, "tools_loaded": 5})
    
    print()
    print("Mixed Content Examples:")
    print("-" * 30)
    
    logger.info("Found 6 tables in database")
    logger.info("Performance - data_processing: 2.34s - Details: {'records_processed': 1500}")
    logger.info("Agent [deepseek] - initialized_successfully - Details: {'model': 'deepseek-chat'}")
    logger.info("Calling execute_query with args: {'query_preview': 'How many orders are there?'}")
    logger.info("execute_query result: Returned 25 rows")
    
    print()
    print("Error Logging Examples (RED):")
    print("-" * 30)
    
    # Test error and critical logs
    logger.error("Database connection failed: Connection timeout after 30 seconds")
    logger.critical("System failure: Unable to initialize agent service")
    logger.warning("Warning: Query took longer than expected (15.4s)")
    
    try:
        # Simulate an error to test exception logging
        raise ValueError("This is a simulated error for testing")
    except Exception as e:
        logger.error(f"Exception occurred: {e}", exc_info=False)  # Don't show full traceback for demo
    
    print()
    print("✅ Colored logging test completed!")

if __name__ == "__main__":
    test_colored_logging()