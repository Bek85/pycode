"""
Usage examples for the production agent system.

This module demonstrates various ways to use the refactored agent system
in different scenarios and environments.
"""

import asyncio
import json
from typing import Optional

from ..config import AppConfig, AgentType, ModelProvider
from ..services import QueryRequest, create_agent_service_sync
from ..api import create_agent_api_sync


def basic_usage_example():
    """
    Basic usage example using the service layer directly.
    """
    print("=== Basic Usage Example ===")
    
    try:
        # Create service (automatically initializes with default config)
        service = create_agent_service_sync()
        
        # Create a query request
        request = QueryRequest(
            query="How many orders are there? Generate a report with the results.",
            session_id="example_session"
        )
        
        # Execute the query
        response = service.execute_query_sync(request)
        
        if response.success:
            print(f"Query executed successfully!")
            print(f"Result: {response.output}")
            print(f"Execution time: {response.execution_time:.2f}s")
            print(f"Agent type: {response.metadata.get('agent_type', 'unknown')}")
        else:
            print(f"Query failed: {response.error_message}")
            
    except Exception as e:
        print(f"Error in basic usage: {e}")


def api_usage_example():
    """
    Example using the API layer for more control.
    """
    print("\n=== API Usage Example ===")
    
    try:
        # Create API instance
        api = create_agent_api_sync()
        
        # Execute query with OpenAI model
        response = api.execute_query_sync(
            query="List all tables in the database",
            session_id="api_session",
            model_provider="openai"  # Use OpenAI instead of default
        )
        
        if response.success:
            print(f"API query executed successfully!")
            print(f"Result: {response.output}")
        else:
            print(f"API query failed: {response.error_message}")
            
        # Get service information
        info = asyncio.run(api.get_service_info())
        print(f"\nService info: Available tools: {info.get('tools_available', [])}")
        
    except Exception as e:
        print(f"Error in API usage: {e}")


def custom_configuration_example():
    """
    Example with custom configuration.
    """
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Create custom configuration
        config = AppConfig()
        config.agent.verbose = True  # Enable verbose mode
        config.agent.agent_type = AgentType.OPENAI_FUNCTIONS  # Use OpenAI functions agent
        config.model.provider = ModelProvider.DEEPSEEK  # Use DeepSeek model
        config.model.temperature = 0.1  # Lower temperature for more deterministic responses
        
        # Create service with custom config
        service = create_agent_service_sync(config)
        
        # Execute query
        request = QueryRequest(
            query="Describe the structure of all tables",
            session_id="custom_config_session"
        )
        
        response = service.execute_query_sync(request)
        
        if response.success:
            print(f"Custom config query executed successfully!")
            print(f"Result: {response.output}")
            print(f"Configuration used: {config.to_dict()}")
        else:
            print(f"Custom config query failed: {response.error_message}")
            
    except Exception as e:
        print(f"Error in custom configuration: {e}")


def session_management_example():
    """
    Example demonstrating session management.
    """
    print("\n=== Session Management Example ===")
    
    try:
        api = create_agent_api_sync()
        
        # Execute queries in different sessions
        sessions = ["session_1", "session_2", "session_3"]
        
        for session in sessions:
            response = api.execute_query_sync(
                query=f"This is a query in {session}. Remember this context.",
                session_id=session
            )
            
            if response.success:
                print(f"Session {session}: Query executed")
            else:
                print(f"Session {session}: Failed - {response.error_message}")
        
        # Clear a specific session
        asyncio.run(api.clear_session("session_2"))
        print("Cleared session_2")
        
        # Execute follow-up queries to test memory
        for session in sessions:
            response = api.execute_query_sync(
                query="What was the context from the previous query?",
                session_id=session
            )
            
            print(f"Session {session} follow-up: {'Success' if response.success else 'Failed'}")
            
    except Exception as e:
        print(f"Error in session management: {e}")


async def async_usage_example():
    """
    Example using async operations.
    """
    print("\n=== Async Usage Example ===")
    
    try:
        from ..api import create_agent_api
        
        # Create API instance asynchronously
        api = await create_agent_api()
        
        # Execute multiple queries concurrently
        queries = [
            "How many orders are there?",
            "What tables are available?", 
            "Generate a summary report"
        ]
        
        tasks = [
            api.execute_query(query, f"async_session_{i}")
            for i, query in enumerate(queries)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            if response.success:
                print(f"Async query {i+1}: Success - {response.output[:100]}...")
            else:
                print(f"Async query {i+1}: Failed - {response.error_message}")
                
    except Exception as e:
        print(f"Error in async usage: {e}")


def error_handling_example():
    """
    Example demonstrating error handling.
    """
    print("\n=== Error Handling Example ===")
    
    try:
        api = create_agent_api_sync()
        
        # Test various error scenarios
        error_queries = [
            "",  # Empty query
            "DROP TABLE orders;",  # Dangerous query (should be blocked)
            "SELECT * FROM nonexistent_table;",  # Invalid SQL
        ]
        
        for i, query in enumerate(error_queries):
            print(f"\nTesting error scenario {i+1}: '{query}'")
            
            response = api.execute_query_sync(
                query=query,
                session_id=f"error_test_{i}"
            )
            
            if response.success:
                print(f"Unexpected success: {response.output}")
            else:
                print(f"Expected error handled: {response.error_message}")
                
    except Exception as e:
        print(f"Error in error handling example: {e}")


def performance_comparison_example():
    """
    Example comparing different agent types and models.
    """
    print("\n=== Performance Comparison Example ===")
    
    try:
        api = create_agent_api_sync()
        
        query = "How many orders are there?"
        configurations = [
            ("openai_functions", "openai"),
            ("tool_calling", "deepseek"),
            ("tool_calling", "openai"),
        ]
        
        results = []
        
        for agent_type, model_provider in configurations:
            print(f"\nTesting: {agent_type} with {model_provider}")
            
            response = api.execute_query_sync(
                query=query,
                session_id=f"perf_test_{agent_type}_{model_provider}",
                agent_type=agent_type,
                model_provider=model_provider
            )
            
            results.append({
                "config": f"{agent_type}/{model_provider}",
                "success": response.success,
                "execution_time": response.execution_time,
                "error": response.error_message if not response.success else None
            })
            
            if response.success:
                print(f"Success in {response.execution_time:.2f}s")
            else:
                print(f"Failed: {response.error_message}")
        
        # Print summary
        print("\n--- Performance Summary ---")
        for result in results:
            status = "SUCCESS" if result["success"] else "FAILED"
            time_str = f"{result['execution_time']:.2f}s" if result["success"] else "N/A"
            print(f"{result['config']}: {status} ({time_str})")
            
    except Exception as e:
        print(f"Error in performance comparison: {e}")


def main():
    """
    Run all examples.
    """
    print("Running Production Agent System Examples")
    print("=" * 50)
    
    # Run synchronous examples
    basic_usage_example()
    api_usage_example()
    custom_configuration_example()
    session_management_example()
    error_handling_example()
    performance_comparison_example()
    
    # Run async example
    asyncio.run(async_usage_example())
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()