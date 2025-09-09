"""
Production-ready main entry point for the agent system.

This module provides multiple interfaces for interacting with the agent system:
- Programmatic API
- REST API server  
- Command-line interface
- Interactive mode (for development/testing)
"""

import asyncio
import argparse
import sys
import json
import time
from typing import Optional

from .config import AppConfig, AgentType, ModelProvider, get_config
from .api import AgentAPI, create_agent_api, create_agent_api_sync, run_fastapi_server
from .services import QueryRequest, create_agent_service_sync
from .utils import AgentLogger, log_agent_action, colorize_result_output, colorize_execution_time
from .utils.logging import ColorCodes


def setup_logging():
    """Setup logging for the application."""
    config = get_config()
    from .utils.logging import AgentLogger
    AgentLogger.setup_logging(config.logging)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Production Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m langchain_demos.main --query "How many orders are there?"
  python -m langchain_demos.main --server --port 8000
  python -m langchain_demos.main --interactive
  python -m langchain_demos.main --config-file config.env --query "List tables"
        """
    )
    
    # Operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--server", "-s",
        action="store_true",
        help="Run as REST API server"
    )
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true", 
        help="Run in interactive mode"
    )
    mode_group.add_argument(
        "--query", "-q",
        type=str,
        help="Execute a single query and exit"
    )
    
    # Configuration options
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to environment configuration file"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (overrides config)"
    )
    
    # Server options
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Query options
    parser.add_argument(
        "--session-id",
        default="default",
        help="Session ID for query execution (default: default)"
    )
    parser.add_argument(
        "--agent-type",
        choices=["openai_functions", "tool_calling"],
        help="Agent type to use"
    )
    parser.add_argument(
        "--model-provider", 
        choices=["openai", "deepseek"],
        help="Model provider to use"
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    return parser


def load_config(config_file: Optional[str] = None) -> AppConfig:
    """Load configuration with optional config file."""
    if config_file:
        import os
        from dotenv import load_dotenv
        
        if os.path.exists(config_file):
            load_dotenv(config_file)
        else:
            print(f"Warning: Config file {config_file} not found")
    
    return get_config()


async def execute_single_query(args, config: AppConfig):
    """Execute a single query and exit."""
    logger = AgentLogger.get_logger(__name__)
    
    try:
        # Create API instance
        api = await create_agent_api(config)
        
        # Convert string arguments to enums if provided
        agent_type = AgentType(args.agent_type) if args.agent_type else None
        model_provider = ModelProvider(args.model_provider) if args.model_provider else None
        
        # Execute query
        log_agent_action("main", "executing_single_query", {
            "query_preview": args.query[:100],
            "session_id": args.session_id,
            "agent_type": args.agent_type,
            "model_provider": args.model_provider
        })
        
        response = await api.execute_query(
            query=args.query,
            session_id=args.session_id,
            agent_type=args.agent_type,
            model_provider=args.model_provider
        )
        
        # Output result
        if args.output_format == "json":
            result = {
                "success": response.success,
                "output": response.output,
                "execution_time": response.execution_time,
                "metadata": response.metadata
            }
            if not response.success:
                result["error"] = response.error_message
            print(json.dumps(result, indent=2))
        else:
            if response.success:
                # Colorize the result output
                colored_result = colorize_result_output(response.output)
                colored_execution_time = colorize_execution_time(response.execution_time)
                
                print(f"\n{ColorCodes.BRIGHT_GREEN}[Result]{ColorCodes.RESET}")
                print(colored_result)
                print(f"\n{colored_execution_time}")
            else:
                print(f"\nError: {response.error_message}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        if args.output_format == "json":
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"Error: {e}")
        sys.exit(1)


def run_server_mode(args, config: AppConfig):
    """Run in server mode."""
    logger = AgentLogger.get_logger(__name__)
    
    try:
        log_agent_action("main", "starting_server", {
            "host": args.host,
            "port": args.port,
            "reload": args.reload
        })
        
        print(f"Starting agent API server on {args.host}:{args.port}")
        print(f"Swagger UI available at: http://{args.host}:{args.port}/docs")
        
        run_fastapi_server(
            config=config,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error starting server: {e}")
        sys.exit(1)


async def run_interactive_mode(config: AppConfig):
    """Run in interactive mode for development and testing."""
    logger = AgentLogger.get_logger(__name__)
    
    try:
        # Create API instance
        api = await create_agent_api(config)
        
        print(f"{ColorCodes.BRIGHT_CYAN}=== Agent Interactive Mode ==={ColorCodes.RESET}")
        print(f"{ColorCodes.GREEN}Type {ColorCodes.BRIGHT_YELLOW}'help'{ColorCodes.GREEN} for available commands, {ColorCodes.BRIGHT_YELLOW}'quit'{ColorCodes.GREEN} to exit{ColorCodes.RESET}")
        print(f"{ColorCodes.BLUE}Use {ColorCodes.BRIGHT_YELLOW}'openai:'{ColorCodes.BLUE} prefix to use OpenAI model instead of default{ColorCodes.RESET}")
        print(f"{ColorCodes.BLUE}Use {ColorCodes.BRIGHT_YELLOW}'info'{ColorCodes.BLUE} to get service information{ColorCodes.RESET}")
        print(f"{ColorCodes.BLUE}Use {ColorCodes.BRIGHT_YELLOW}'clear'{ColorCodes.BLUE} to clear current session{ColorCodes.RESET}")
        print(f"{ColorCodes.BLUE}Use {ColorCodes.BRIGHT_YELLOW}'sessions'{ColorCodes.BLUE} to manage sessions{ColorCodes.RESET}")
        
        session_id = "interactive"
        
        while True:
            try:
                user_input = input(f"\n{ColorCodes.BRIGHT_GREEN}[{session_id}]{ColorCodes.RESET} {ColorCodes.CYAN}>{ColorCodes.RESET} ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print(f"""
{ColorCodes.BRIGHT_YELLOW}Available commands:{ColorCodes.RESET}
  {ColorCodes.BRIGHT_CYAN}help{ColorCodes.RESET}              - Show this help message
  {ColorCodes.BRIGHT_CYAN}info{ColorCodes.RESET}              - Show service information  
  {ColorCodes.BRIGHT_CYAN}clear{ColorCodes.RESET}             - Clear current session history
  {ColorCodes.BRIGHT_CYAN}sessions{ColorCodes.RESET}          - Show session management options
  {ColorCodes.BRIGHT_CYAN}quit/exit/q{ColorCodes.RESET}       - Exit interactive mode
  
{ColorCodes.BRIGHT_YELLOW}Query execution:{ColorCodes.RESET}
  {ColorCodes.GREEN}<query>{ColorCodes.RESET}           - Execute query with default model
  {ColorCodes.GREEN}openai:<query>{ColorCodes.RESET}    - Execute query with OpenAI model
  
{ColorCodes.BRIGHT_YELLOW}Examples:{ColorCodes.RESET}
  {ColorCodes.DIM}How many orders are there?{ColorCodes.RESET}
  {ColorCodes.DIM}openai:List all tables{ColorCodes.RESET}
  {ColorCodes.DIM}Generate a report with order statistics{ColorCodes.RESET}
                    """)
                    continue
                
                elif user_input.lower() == 'info':
                    info = await api.get_service_info()
                    print(json.dumps(info, indent=2))
                    continue
                
                elif user_input.lower() == 'clear':
                    await api.clear_session(session_id)
                    print(f"Session '{session_id}' cleared")
                    continue
                
                elif user_input.lower() == 'sessions':
                    print("""
Session management:
  clear             - Clear current session
  session <id>      - Switch to session <id>
                    """)
                    continue
                
                elif user_input.lower().startswith('session '):
                    new_session = user_input[8:].strip()
                    if new_session:
                        session_id = new_session
                        print(f"Switched to session: {session_id}")
                    continue
                
                # Handle query execution
                use_openai = user_input.startswith("openai:")
                if use_openai:
                    query = user_input[7:].strip()
                    model_provider = "openai"
                else:
                    query = user_input
                    model_provider = None
                
                if not query:
                    print("Please provide a query")
                    continue
                
                print(f"{ColorCodes.BRIGHT_YELLOW}Executing query{' with OpenAI' if use_openai else ''}...{ColorCodes.RESET}")
                start_time = time.time()
                
                response = await api.execute_query(
                    query=query,
                    session_id=session_id,
                    model_provider=model_provider
                )
                
                if response.success:
                    # Colorize the result output
                    colored_result = colorize_result_output(response.output)
                    colored_execution_time = colorize_execution_time(response.execution_time)
                    
                    print(f"\n{ColorCodes.BRIGHT_GREEN}[Result]{ColorCodes.RESET}")
                    print(colored_result)
                    print(f"\n{colored_execution_time}")
                else:
                    print(f"\nError: {response.error_message}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
                
    except Exception as e:
        logger.error(f"Failed to start interactive mode: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Override log level if specified
    if args.log_level:
        config.logging.level = args.log_level
    
    # Setup logging
    setup_logging()
    logger = AgentLogger.get_logger(__name__)
    
    log_agent_action("main", "started", {
        "mode": "server" if args.server else "interactive" if args.interactive else "query" if args.query else "help",
        "config_file": args.config_file
    })
    
    try:
        # Determine mode and execute
        if args.server:
            run_server_mode(args, config)
        elif args.query:
            asyncio.run(execute_single_query(args, config))
        elif args.interactive:
            asyncio.run(run_interactive_mode(config))
        else:
            # Default to showing help
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()