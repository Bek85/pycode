"""
Production-ready LangChain Agent System.

This package provides a complete, production-ready implementation of LangChain agents
with proper configuration management, error handling, logging, and API interfaces.

Key Features:
- Modular architecture with separation of concerns
- Comprehensive configuration management
- Production-ready error handling and logging  
- Agent factory pattern for flexible agent creation
- REST API and programmatic interfaces
- Session management and history
- Tool integration with database and reporting
- Type safety and documentation

Usage Examples:

Programmatic Usage:
    from langchain_demos import create_agent_service_sync, QueryRequest
    
    service = create_agent_service_sync()
    request = QueryRequest(query="How many orders are there?")
    response = service.execute_query_sync(request)
    print(response.output)

API Usage:
    from langchain_demos import create_agent_api_sync
    
    api = create_agent_api_sync()
    response = api.execute_query_sync("List all tables")
    print(response.output)

Server Mode:
    python -m langchain_demos.main --server --port 8000

Interactive Mode:
    python -m langchain_demos.main --interactive

Single Query:
    python -m langchain_demos.main --query "How many customers are there?"
"""

# Main exports for easy access
from .config import (
    AppConfig,
    AgentType,
    ModelProvider,
    get_config
)

from .services import (
    AgentService,
    QueryRequest,
    QueryResponse,
    create_agent_service,
    create_agent_service_sync
)

from .api import (
    AgentAPI,
    create_agent_api,
    create_agent_api_sync,
    run_fastapi_server
)

from .agents import (
    AgentFactory,
    BaseAgent,
    AgentExecutionError,
    AgentCreationError
)

from .tools import (
    create_database_tools,
    create_reporting_tools,
    get_tables_info
)

from .utils import (
    AgentLogger,
    log_function_call,
    log_agent_action,
    log_error,
    log_performance
)

# Version information
__version__ = "1.0.0"
__author__ = "Agent System Team"
__description__ = "Production-ready LangChain Agent System"

# Export main components
__all__ = [
    # Configuration
    "AppConfig",
    "AgentType",
    "ModelProvider", 
    "get_config",
    
    # Services
    "AgentService",
    "QueryRequest",
    "QueryResponse",
    "create_agent_service",
    "create_agent_service_sync",
    
    # API
    "AgentAPI",
    "create_agent_api",
    "create_agent_api_sync",
    "run_fastapi_server",
    
    # Agents
    "AgentFactory",
    "BaseAgent",
    "AgentExecutionError",
    "AgentCreationError",
    
    # Tools
    "create_database_tools",
    "create_reporting_tools",
    "get_tables_info",
    
    # Utils
    "AgentLogger",
    "log_function_call",
    "log_agent_action",
    "log_error",
    "log_performance",
    
    # Package info
    "__version__"
]