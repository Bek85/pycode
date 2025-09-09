"""
Production-ready API interface for agent operations.

This module provides both REST API and programmatic interfaces for interacting
with the agent system in production environments.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from contextlib import asynccontextmanager

# FastAPI imports (optional dependency)
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydanticBaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..services import AgentService, QueryRequest, QueryResponse, AgentServiceError, create_agent_service
from ..config import AppConfig, AgentType, ModelProvider, get_config
from ..utils import AgentLogger, log_function_call, log_error, log_agent_action


# Pydantic models for API validation (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    class QueryRequestModel(PydanticBaseModel):
        """API model for query requests."""
        query: str = Field(..., min_length=1, max_length=10000, description="The query to execute")
        session_id: str = Field(default="default", max_length=100, description="Session identifier")
        agent_type: Optional[str] = Field(default=None, description="Agent type (openai_functions or tool_calling)")
        model_provider: Optional[str] = Field(default=None, description="Model provider (openai or deepseek)")
    
    class QueryResponseModel(PydanticBaseModel):
        """API model for query responses."""
        output: str
        metadata: Dict[str, Any]
        execution_time: float
        success: bool
        error_message: Optional[str] = None
    
    class ServiceInfoModel(PydanticBaseModel):
        """API model for service information."""
        initialized: bool
        config: Dict[str, Any]
        tools_available: List[str]
        agents_created: List[Dict[str, Any]]
        tables_info: Optional[str]
        llm_providers: List[str]


class AgentAPIError(Exception):
    """Exception raised by API operations."""
    pass


class AgentAPI:
    """
    High-level API interface for agent operations.
    
    Provides both programmatic and REST API interfaces for production use.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the API interface.
        
        Args:
            config: Optional configuration (uses default if not provided)
        """
        self.config = config or get_config()
        self.logger = AgentLogger.get_logger(__name__)
        self._service: Optional[AgentService] = None
        self._initialized = False
        
        log_agent_action("agent_api", "initialized", {"fastapi_available": FASTAPI_AVAILABLE})
    
    async def initialize(self):
        """Initialize the API and underlying service."""
        if self._initialized:
            return
        
        try:
            self._service = await create_agent_service(self.config)
            self._initialized = True
            
            log_agent_action("agent_api", "initialized_successfully", {})
            
        except Exception as e:
            log_error(e, "API initialization failed")
            raise AgentAPIError(f"Failed to initialize API: {str(e)}") from e
    
    def _ensure_initialized(self):
        """Ensure the API is initialized."""
        if not self._initialized or not self._service:
            raise AgentAPIError("API not initialized. Call initialize() first.")
    
    async def execute_query(
        self,
        query: str,
        session_id: str = "default",
        agent_type: Optional[str] = None,
        model_provider: Optional[str] = None
    ) -> QueryResponse:
        """
        Execute a query through the API.
        
        Args:
            query: The query to execute
            session_id: Session identifier
            agent_type: Optional agent type override
            model_provider: Optional model provider override
            
        Returns:
            Query response object
        """
        self._ensure_initialized()
        
        try:
            # Validate and convert parameters
            agent_type_enum = None
            if agent_type:
                try:
                    agent_type_enum = AgentType(agent_type)
                except ValueError:
                    raise AgentAPIError(f"Invalid agent type: {agent_type}")
            
            model_provider_enum = None
            if model_provider:
                try:
                    model_provider_enum = ModelProvider(model_provider)
                except ValueError:
                    raise AgentAPIError(f"Invalid model provider: {model_provider}")
            
            # Create request object
            request = QueryRequest(
                query=query,
                session_id=session_id,
                agent_type=agent_type_enum,
                model_provider=model_provider_enum
            )
            
            # Execute query
            return await self._service.execute_query(request)
            
        except AgentServiceError as e:
            log_error(e, f"Service error during query execution: {query[:50]}...")
            raise AgentAPIError(f"Query execution failed: {str(e)}") from e
        except Exception as e:
            log_error(e, f"Unexpected error during API query execution: {query[:50]}...")
            raise AgentAPIError(f"Unexpected error: {str(e)}") from e
    
    def execute_query_sync(
        self,
        query: str,
        session_id: str = "default",
        agent_type: Optional[str] = None,
        model_provider: Optional[str] = None
    ) -> QueryResponse:
        """
        Execute a query synchronously.
        
        Args:
            query: The query to execute
            session_id: Session identifier
            agent_type: Optional agent type override
            model_provider: Optional model provider override
            
        Returns:
            Query response object
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.execute_query(query, session_id, agent_type, model_provider)
            )
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(
                self.execute_query(query, session_id, agent_type, model_provider)
            )
    
    async def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the service and its configuration.
        
        Returns:
            Dictionary containing service information
        """
        self._ensure_initialized()
        return self._service.get_service_info()
    
    async def clear_session(self, session_id: str):
        """
        Clear a specific session's history.
        
        Args:
            session_id: Session identifier to clear
        """
        self._ensure_initialized()
        self._service.clear_session(session_id)
    
    async def clear_all_sessions(self):
        """Clear all sessions and agents."""
        self._ensure_initialized()
        self._service.clear_all_agents()
    
    def create_fastapi_app(self) -> Optional[FastAPI]:
        """
        Create a FastAPI application instance.
        
        Returns:
            FastAPI app instance or None if FastAPI is not available
        """
        if not FASTAPI_AVAILABLE:
            self.logger.warning("FastAPI not available, cannot create REST API")
            return None
        
        app = FastAPI(
            title="Agent API",
            description="Production-ready API for LangChain agent operations",
            version="1.0.0"
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Lifespan management
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Initialize on startup
            await self.initialize()
            yield
            # Cleanup on shutdown (if needed)
        
        app.router.lifespan_context = lifespan
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}
        
        # Query execution endpoint
        @app.post("/query", response_model=QueryResponseModel)
        async def execute_query_endpoint(request: QueryRequestModel):
            """Execute a query through the API."""
            try:
                response = await self.execute_query(
                    query=request.query,
                    session_id=request.session_id,
                    agent_type=request.agent_type,
                    model_provider=request.model_provider
                )
                
                return QueryResponseModel(
                    output=response.output,
                    metadata=response.metadata,
                    execution_time=response.execution_time,
                    success=response.success,
                    error_message=response.error_message
                )
                
            except AgentAPIError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                log_error(e, "Unexpected error in query endpoint")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Service information endpoint
        @app.get("/info", response_model=ServiceInfoModel)
        async def get_info_endpoint():
            """Get service information."""
            try:
                info = await self.get_service_info()
                return ServiceInfoModel(**info)
            except Exception as e:
                log_error(e, "Error getting service info")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Session management endpoints
        @app.delete("/sessions/{session_id}")
        async def clear_session_endpoint(session_id: str):
            """Clear a specific session."""
            try:
                await self.clear_session(session_id)
                return {"message": f"Session {session_id} cleared successfully"}
            except Exception as e:
                log_error(e, f"Error clearing session {session_id}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @app.delete("/sessions")
        async def clear_all_sessions_endpoint():
            """Clear all sessions."""
            try:
                await self.clear_all_sessions()
                return {"message": "All sessions cleared successfully"}
            except Exception as e:
                log_error(e, "Error clearing all sessions")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        return app


# Convenience functions
async def create_agent_api(config: Optional[AppConfig] = None) -> AgentAPI:
    """
    Create and initialize an agent API.
    
    Args:
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Initialized agent API
    """
    api = AgentAPI(config)
    await api.initialize()
    return api


def create_agent_api_sync(config: Optional[AppConfig] = None) -> AgentAPI:
    """
    Create and initialize an agent API synchronously.
    
    Args:
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Initialized agent API
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(create_agent_api(config))
    except RuntimeError:
        # No event loop running, create one
        return asyncio.run(create_agent_api(config))


def run_fastapi_server(
    config: Optional[AppConfig] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Run the FastAPI server.
    
    Args:
        config: Optional configuration
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available. Install it with: pip install fastapi uvicorn")
    
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError("uvicorn not available. Install it with: pip install uvicorn")
    
    # Create API and FastAPI app
    api = AgentAPI(config)
    app = api.create_fastapi_app()
    
    if app is None:
        raise RuntimeError("Failed to create FastAPI app")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )