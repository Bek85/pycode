# Production-Ready LangChain Agent System

A complete, production-ready implementation of LangChain agents with proper architecture, error handling, logging, and API interfaces.

## 🚀 Key Features

- **Modular Architecture**: Clean separation of concerns with distinct layers for configuration, agents, tools, services, and APIs
- **Agent Factory Pattern**: Flexible agent creation with support for multiple agent types and models
- **Comprehensive Configuration**: Environment-based configuration with validation and type safety
- **Production-Ready Error Handling**: Comprehensive error handling with proper exception hierarchies
- **Structured Logging**: Centralized logging with configurable levels and outputs
- **Multiple Interfaces**: REST API, programmatic API, CLI, and interactive modes
- **Session Management**: Persistent conversation history with session isolation
- **Tool Integration**: Database operations, report generation, and extensible tool framework
- **Type Safety**: Full type hints and Pydantic validation throughout
- **Dependency Injection**: Testable design with proper dependency management

## 📁 Architecture

```
langchain_demos/
├── config/              # Configuration management
│   ├── __init__.py
│   └── agent_config.py  # Centralized configuration classes
├── agents/              # Agent factory and implementations  
│   ├── __init__.py
│   └── factory.py       # Agent factory pattern
├── tools/               # Tool implementations
│   ├── __init__.py
│   ├── database.py      # Database tools with connection management
│   └── reporting.py     # Report generation tools
├── services/            # Business logic layer
│   ├── __init__.py
│   └── agent_service.py # High-level service orchestration
├── api/                 # API interfaces
│   ├── __init__.py
│   └── agent_api.py     # REST API and programmatic interface
├── utils/               # Utilities and logging
│   ├── __init__.py
│   └── logging.py       # Centralized logging system
├── examples/            # Usage examples
│   ├── __init__.py
│   └── usage_examples.py
├── main.py              # Main entry point with CLI
└── README.md            # This file
```

## 🛠 Installation

1. **Install Dependencies**:
```bash
pip install langchain langchain-openai langchain-community
pip install sqlite3 python-dotenv pydantic
pip install fastapi uvicorn  # Optional: for REST API
```

2. **Configure Environment**:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
OPENAI_API_KEY=your_openai_key_here
DEFAULT_MODEL_PROVIDER=deepseek
LOG_LEVEL=INFO
```

3. **Verify Database**:
Ensure your SQLite database exists at the configured path (default: `../db/db.sqlitedb`).

## 🚀 Usage

### Command Line Interface

```bash
# Interactive mode
python -m langchain_demos.main --interactive

# Single query
python -m langchain_demos.main --query "How many orders are there?"

# REST API server
python -m langchain_demos.main --server --port 8000

# Custom configuration
python -m langchain_demos.main --config-file custom.env --query "List tables"
```

### Programmatic Usage

```python
from langchain_demos import create_agent_service_sync, QueryRequest

# Create service
service = create_agent_service_sync()

# Execute query
request = QueryRequest(
    query="How many orders are there?",
    session_id="my_session"
)
response = service.execute_query_sync(request)

if response.success:
    print(response.output)
else:
    print(f"Error: {response.error_message}")
```

### API Usage

```python
from langchain_demos import create_agent_api_sync

# Create API instance
api = create_agent_api_sync()

# Execute query with specific model
response = api.execute_query_sync(
    query="Generate a sales report",
    session_id="api_session",
    model_provider="openai"
)

print(response.output)
```

### REST API

```bash
# Start server
python -m langchain_demos.main --server --port 8000

# Execute query via HTTP
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many customers are there?", "session_id": "rest_session"}'

# Get service info
curl "http://localhost:8000/info"

# Clear session
curl -X DELETE "http://localhost:8000/sessions/rest_session"
```

## ⚙️ Configuration

The system uses environment variables for configuration:

### Core Settings
- `DEFAULT_MODEL_PROVIDER`: Model provider (openai/deepseek)
- `DEFAULT_AGENT_TYPE`: Agent type (openai_functions/tool_calling)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `LOG_FILE`: Optional log file path

### Database Settings
- `DB_PATH`: Database file path (default: ../db/db.sqlitedb)
- `DB_TIMEOUT`: Connection timeout in seconds

### Model Settings
- `MODEL_TEMPERATURE`: Model temperature (0.0-2.0)
- `MODEL_MAX_TOKENS`: Maximum tokens per request
- `MODEL_TIMEOUT`: Request timeout in seconds

### Tool Settings
- `ENABLED_TOOLS`: Comma-separated list of enabled tools
- `REPORT_DIR`: Directory for generated reports
- `SQL_TIMEOUT`: SQL query timeout in seconds

### Agent Settings
- `AGENT_VERBOSE`: Enable verbose agent output (true/false)
- `AGENT_MAX_ITERATIONS`: Maximum agent iterations
- `SESSION_TIMEOUT`: Session timeout in seconds

## 🔧 Available Agent Types

### OpenAI Functions Agent
- Uses OpenAI's function calling capabilities
- Best for complex reasoning tasks
- Requires OpenAI API key

### Tool Calling Agent
- Universal tool calling agent
- Works with any model that supports tool calling
- Compatible with local models

## 🛠 Available Tools

### Database Tools
- **run_sqlite_query**: Execute SELECT queries safely
- **describe_tables**: Get table schema information

### Reporting Tools
- **generate_report**: Create HTML reports with automatic formatting

## 📊 Session Management

The system supports persistent conversation history:

```python
# Queries in the same session maintain context
api = create_agent_api_sync()

api.execute_query_sync("Remember that I like blue color", session_id="user123")
response = api.execute_query_sync("What color do I like?", session_id="user123")
# Response will remember the blue color preference

# Different sessions are isolated
api.execute_query_sync("I like red color", session_id="user456")  # Won't affect user123
```

## 🚨 Error Handling

The system provides comprehensive error handling:

```python
try:
    response = api.execute_query_sync("Some query")
    if response.success:
        print(response.output)
    else:
        print(f"Query failed: {response.error_message}")
except AgentAPIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📝 Logging

Structured logging throughout the system:

```python
from langchain_demos import AgentLogger

logger = AgentLogger.get_logger(__name__)
logger.info("Custom log message")
```

## 🧪 Testing

Run the examples to verify functionality:

```python
from langchain_demos.examples import main

# Run all usage examples
main()
```

## 🏗 Extending the System

### Adding New Tools

1. Create a new tool class:
```python
from langchain.tools import Tool
from pydantic.v1 import BaseModel

class MyToolArgsSchema(BaseModel):
    param: str

def my_tool_function(param: str) -> str:
    return f"Processed: {param}"

my_tool = Tool.from_function(
    name="my_tool",
    description="Description of what my tool does",
    func=my_tool_function,
    args_schema=MyToolArgsSchema,
)
```

2. Register in the configuration:
```python
# Add to enabled_tools in config
config.tools.enabled_tools.append("my_tool")
```

### Adding New Agent Types

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods
3. Register in the `AgentFactory`

### Custom Configuration

```python
from langchain_demos import AppConfig, AgentType, ModelProvider

config = AppConfig()
config.agent.agent_type = AgentType.TOOL_CALLING
config.model.provider = ModelProvider.OPENAI
config.model.temperature = 0.2
config.logging.level = "DEBUG"

service = create_agent_service_sync(config)
```

## 📋 Production Considerations

### Security
- SQL injection protection via parameterized queries
- Dangerous SQL operation blocking
- Input validation and sanitization
- File path security for reports

### Performance
- Connection pooling for database operations
- Agent instance caching
- Configurable timeouts
- Background task support

### Monitoring
- Comprehensive logging with structured format
- Performance metrics tracking
- Error tracking and reporting
- Session management monitoring

### Scalability
- Stateless service design
- Session-based context management
- Configurable resource limits
- Async operation support

## 🤝 Contributing

1. Follow the existing architecture patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Follow type hint conventions
5. Add proper error handling and logging

## 📄 License

This project is part of a learning/demonstration codebase. Please review the main project license.

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify database path in configuration
   - Check file permissions
   - Ensure database file exists

2. **Model Provider Errors**
   - Check API keys in environment variables
   - Verify model availability
   - Check network connectivity

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify package structure

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python -m langchain_demos.main --log-level DEBUG --interactive
```

For more assistance, check the logs and error messages for specific guidance.