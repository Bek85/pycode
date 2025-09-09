"""
Agent service layer providing high-level business logic and orchestration.

This module provides a service layer that orchestrates agent operations,
manages dependencies, and provides a clean interface for the application.
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

from ..config import AppConfig, AgentType, ModelProvider
from ..agents import AgentFactory, BaseAgent, AgentExecutionError, AgentCreationError
from ..tools import create_database_tools, create_reporting_tools, get_tables_info
from ..utils import AgentLogger, log_function_call, log_error, log_performance, log_agent_action

# Import the centralized LLM configuration
try:
    from ...config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import get_llm


@dataclass 
class QueryRequest:
    """Request object for agent queries."""
    query: str
    session_id: str = "default"
    agent_type: Optional[AgentType] = None
    model_provider: Optional[ModelProvider] = None


@dataclass
class QueryResponse:
    """Response object for agent queries."""
    output: str
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class AgentServiceError(Exception):
    """Exception raised by agent service operations."""
    pass


class AgentService:
    """
    High-level service for managing agent operations.
    
    This service provides a clean interface for creating and executing
    agent queries with proper error handling, logging, and dependency management.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the agent service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = AgentLogger.get_logger(__name__)
        
        # Initialize components
        self._agent_factory: Optional[AgentFactory] = None
        self._tools_cache: Dict[str, List] = {}
        self._tables_info_cache: Optional[str] = None
        self._llm_cache: Dict[str, Any] = {}
        
        # Service state
        self._initialized = False
        
        log_agent_action("agent_service", "initialized", {"config_valid": True})
    
    async def initialize(self) -> None:
        """
        Initialize the service asynchronously.
        
        This method sets up all necessary components and validates the configuration.
        """
        if self._initialized:
            self.logger.info("Service already initialized")
            return
        
        try:
            start_time = time.time()
            
            # Initialize agent factory
            self._agent_factory = AgentFactory(self.config)
            
            # Cache database tables information
            self._tables_info_cache = get_tables_info(self.config)
            
            # Initialize tools cache
            self._init_tools_cache()
            
            # Initialize LLM cache
            self._init_llm_cache()
            
            self._initialized = True
            
            log_performance("service_initialization", time.time() - start_time, {
                "tables_found": len(self._tables_info_cache.split('\n')) if self._tables_info_cache else 0
            })
            
            log_agent_action("agent_service", "initialized_successfully", {
                "initialization_time": time.time() - start_time
            })
            
        except Exception as e:
            log_error(e, "Service initialization failed")
            raise AgentServiceError(f"Failed to initialize service: {str(e)}") from e
    
    def _init_tools_cache(self):
        """Initialize the tools cache."""
        try:
            # Create database tools if enabled
            if "sql" in self.config.tools.enabled_tools or "describe_tables" in self.config.tools.enabled_tools:
                db_tools = create_database_tools(self.config)
                self._tools_cache["database"] = db_tools
            
            # Create reporting tools if enabled  
            if "report" in self.config.tools.enabled_tools:
                report_tools = create_reporting_tools(self.config)
                self._tools_cache["reporting"] = report_tools
            
            self.logger.info(f"Initialized tools cache with {len(self._tools_cache)} tool categories")
            
        except Exception as e:
            log_error(e, "Failed to initialize tools cache")
            raise
    
    def _init_llm_cache(self):
        """Initialize the LLM cache."""
        try:
            # Cache the configured model provider
            provider_name = self.config.model.provider.value
            self._llm_cache[provider_name] = get_llm(provider_name)
            
            self.logger.info(f"Initialized LLM cache for provider: {provider_name}")
            
        except Exception as e:
            log_error(e, f"Failed to initialize LLM for provider: {self.config.model.provider.value}")
            raise
    
    def _ensure_initialized(self):
        """Ensure the service is initialized."""
        if not self._initialized:
            raise AgentServiceError("Service not initialized. Call initialize() first.")
    
    def _get_all_tools(self) -> List:
        """Get all enabled tools."""
        all_tools = []
        for tool_list in self._tools_cache.values():
            all_tools.extend(tool_list)
        return all_tools
    
    def _get_llm(self, provider: Optional[ModelProvider] = None):
        """Get LLM instance for the specified provider."""
        provider = provider or self.config.model.provider
        provider_name = provider.value
        
        if provider_name not in self._llm_cache:
            try:
                self._llm_cache[provider_name] = get_llm(provider_name)
            except Exception as e:
                log_error(e, f"Failed to get LLM for provider: {provider_name}")
                raise AgentServiceError(f"Failed to get LLM: {str(e)}") from e
        
        return self._llm_cache[provider_name]
    
    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a query using the configured agent.
        
        Args:
            request: Query request object
            
        Returns:
            Query response object
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            log_function_call("execute_query", {
                "query_preview": request.query[:100],
                "session_id": request.session_id,
                "agent_type": request.agent_type.value if request.agent_type else "default",
                "model_provider": request.model_provider.value if request.model_provider else "default"
            })
            
            # Determine agent type and model provider
            agent_type = request.agent_type or self.config.agent.agent_type
            model_provider = request.model_provider or self.config.model.provider
            
            # Get or create agent
            agent = await self._get_or_create_agent(agent_type, model_provider, request.session_id)
            
            # Execute query
            result = agent.execute(request.query, request.session_id)
            
            execution_time = time.time() - start_time
            
            log_performance("query_execution", execution_time, {
                "agent_type": agent_type.value,
                "model_provider": model_provider.value,
                "session_id": request.session_id
            })
            
            response = QueryResponse(
                output=result["output"],
                metadata=result["metadata"],
                execution_time=execution_time,
                success=True
            )
            
            log_function_call("execute_query", result=f"Query executed successfully in {execution_time:.2f}s")
            
            return response
            
        except AgentExecutionError as e:
            execution_time = time.time() - start_time
            log_error(e, f"Agent execution failed for query: {request.query[:50]}...")
            
            return QueryResponse(
                output="",
                metadata={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_error(e, f"Unexpected error during query execution: {request.query[:50]}...")
            
            return QueryResponse(
                output="",
                metadata={},
                execution_time=execution_time,
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    async def _get_or_create_agent(
        self,
        agent_type: AgentType,
        model_provider: ModelProvider,
        session_id: str
    ) -> BaseAgent:
        """
        Get or create an agent for the specified configuration.
        
        Args:
            agent_type: Type of agent to create
            model_provider: Model provider to use
            session_id: Session identifier
            
        Returns:
            Configured agent instance
        """
        agent_id = f"{agent_type.value}_{model_provider.value}_{session_id}"
        
        # Try to get existing agent
        agent = self._agent_factory.get_agent(agent_id)
        
        if agent is None:
            # Create new agent
            try:
                tools = self._get_all_tools()
                llm = self._get_llm(model_provider)
                
                agent = self._agent_factory.create_agent(
                    agent_type=agent_type,
                    tools=tools,
                    llm=llm,
                    tables_info=self._tables_info_cache or "",
                    agent_id=agent_id
                )
                
                log_agent_action(
                    agent_type.value,
                    "agent_created",
                    {
                        "model_provider": model_provider.value,
                        "session_id": session_id,
                        "tools_count": len(tools)
                    }
                )
                
            except AgentCreationError as e:
                log_error(e, f"Failed to create agent: {agent_id}")
                raise AgentServiceError(f"Failed to create agent: {str(e)}") from e
        
        return agent
    
    def execute_query_sync(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a query synchronously (for backward compatibility).
        
        Args:
            request: Query request object
            
        Returns:
            Query response object
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.execute_query(request))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.execute_query(request))
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the service and its configuration.
        
        Returns:
            Dictionary containing service information
        """
        self._ensure_initialized()
        
        return {
            "initialized": self._initialized,
            "config": self.config.to_dict(),
            "tools_available": list(self._tools_cache.keys()),
            "agents_created": self._agent_factory.list_agents() if self._agent_factory else [],
            "tables_info": self._tables_info_cache,
            "llm_providers": list(self._llm_cache.keys())
        }
    
    def clear_all_agents(self):
        """Clear all created agents and their history."""
        if self._agent_factory:
            self._agent_factory.clear_all_agents()
            log_agent_action("agent_service", "cleared_all_agents", {})
    
    def clear_session(self, session_id: str):
        """
        Clear a specific session's history.
        
        Args:
            session_id: Session identifier to clear
        """
        if self._agent_factory:
            self._agent_factory.history_manager.clear_session(session_id)
            log_agent_action("agent_service", "cleared_session", {"session_id": session_id})
    
    @asynccontextmanager
    async def managed_session(self, session_id: str):
        """
        Context manager for managing agent sessions.
        
        Args:
            session_id: Session identifier
        """
        try:
            yield session_id
        finally:
            # Optionally clear session on exit
            if hasattr(self, '_auto_clear_sessions') and self._auto_clear_sessions:
                self.clear_session(session_id)


# Convenience functions for simple use cases
async def create_agent_service(config: Optional[AppConfig] = None) -> AgentService:
    """
    Create and initialize an agent service.
    
    Args:
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Initialized agent service
    """
    if config is None:
        from ..config import get_config
        config = get_config()
    
    service = AgentService(config)
    await service.initialize()
    return service


def create_agent_service_sync(config: Optional[AppConfig] = None) -> AgentService:
    """
    Create and initialize an agent service synchronously.
    
    Args:
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Initialized agent service
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(create_agent_service(config))
    except RuntimeError:
        # No event loop running, create one
        return asyncio.run(create_agent_service(config))