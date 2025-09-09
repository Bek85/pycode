"""
Agent factory for creating and configuring different types of agents.

This module provides a factory pattern implementation for creating
various agent types with proper configuration and dependency injection.
"""

from typing import Dict, List, Optional, Any, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.tools import BaseTool

from ..config import AppConfig, AgentType, ModelProvider
from ..utils import AgentLogger, log_agent_action, log_error, log_performance


class AgentInterface(Protocol):
    """Protocol defining the agent interface."""
    
    def execute(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Execute a query with the agent."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        ...


@dataclass
class AgentMetadata:
    """Metadata for agent instances."""
    agent_type: str
    model_provider: str
    created_at: float
    last_used: float
    usage_count: int = 0


class ChatMessageHistory(BaseChatMessageHistory):
    """
    Simple in-memory chat message history implementation.
    
    For production use, consider implementing persistent storage
    or using a distributed cache like Redis.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.logger = AgentLogger.get_logger(__name__)
    
    def add_message(self, message):
        """Add a message to the history."""
        self.messages.append(message)
        self.logger.debug(f"Added message to session {self.session_id}: {type(message).__name__}")
    
    def clear(self):
        """Clear all messages from the history."""
        self.messages = []
        self.logger.info(f"Cleared message history for session {self.session_id}")
    
    def get_messages(self):
        """Get all messages in the history."""
        return self.messages


class MessageHistoryManager:
    """Manages chat message histories for different sessions."""
    
    def __init__(self, max_sessions: int = 100):
        self.histories: Dict[str, ChatMessageHistory] = {}
        self.max_sessions = max_sessions
        self.logger = AgentLogger.get_logger(__name__)
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create a message history for a session."""
        if session_id not in self.histories:
            if len(self.histories) >= self.max_sessions:
                # Remove oldest session (simple LRU)
                oldest_session = min(self.histories.keys())
                del self.histories[oldest_session]
                self.logger.warning(f"Removed oldest session {oldest_session} due to max_sessions limit")
            
            self.histories[session_id] = ChatMessageHistory(session_id)
            self.logger.info(f"Created new message history for session {session_id}")
        
        return self.histories[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.histories:
            self.histories[session_id].clear()
            self.logger.info(f"Cleared session {session_id}")
    
    def clear_all_sessions(self):
        """Clear all sessions."""
        self.histories.clear()
        self.logger.info("Cleared all sessions")


class BaseAgent(ABC):
    """Base class for all agent implementations."""
    
    def __init__(self, config: AppConfig, tools: List[BaseTool], llm, prompt: ChatPromptTemplate):
        self.config = config
        self.tools = tools
        self.llm = llm
        self.prompt = prompt
        self.logger = AgentLogger.get_logger(__name__)
        self.metadata = AgentMetadata(
            agent_type=self.__class__.__name__,
            model_provider=config.model.provider.value,
            created_at=time.time(),
            last_used=time.time()
        )
    
    @abstractmethod
    def _create_agent(self):
        """Create the underlying agent instance."""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Get the agent type identifier."""
        pass
    
    def execute(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute a query with the agent.
        
        Args:
            query: The query to execute
            session_id: Session identifier for maintaining context
            
        Returns:
            Dictionary containing the result and metadata
        """
        start_time = time.time()
        
        try:
            log_agent_action(
                self.get_agent_type(),
                "execute_query",
                {"query_preview": query[:100], "session_id": session_id}
            )
            
            result = self._execute_query(query, session_id)
            
            # Update metadata
            self.metadata.last_used = time.time()
            self.metadata.usage_count += 1
            
            duration = time.time() - start_time
            log_performance("agent_execution", duration, {"agent_type": self.get_agent_type()})
            
            return {
                "output": result["output"],
                "metadata": {
                    "agent_type": self.get_agent_type(),
                    "model_provider": self.metadata.model_provider,
                    "execution_time": duration,
                    "session_id": session_id
                }
            }
            
        except Exception as e:
            log_error(e, f"Agent execution failed for {self.get_agent_type()}")
            raise AgentExecutionError(f"Failed to execute query: {str(e)}") from e
    
    @abstractmethod
    def _execute_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute the actual query (implemented by subclasses)."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "agent_type": self.get_agent_type(),
            "model_provider": self.metadata.model_provider,
            "tools": [tool.name for tool in self.tools],
            "created_at": self.metadata.created_at,
            "last_used": self.metadata.last_used,
            "usage_count": self.metadata.usage_count
        }


class OpenAIFunctionsAgent(BaseAgent):
    """OpenAI Functions agent implementation."""
    
    def __init__(self, config: AppConfig, tools: List[BaseTool], llm, prompt: ChatPromptTemplate, history_manager: MessageHistoryManager):
        super().__init__(config, tools, llm, prompt)
        self.history_manager = history_manager
        self.agent = self._create_agent()
        self.runnable_agent = self._create_runnable_agent()
    
    def _create_agent(self):
        """Create the OpenAI functions agent."""
        return create_openai_functions_agent(self.llm, self.tools, self.prompt)
    
    def _create_runnable_agent(self):
        """Create a runnable agent with message history."""
        agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.agent.verbose,
            max_iterations=self.config.agent.max_iterations,
            early_stopping_method=self.config.agent.early_stopping_method
        )
        
        return RunnableWithMessageHistory(
            agent_executor,
            self.history_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    def get_agent_type(self) -> str:
        return "openai_functions"
    
    def _execute_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute query using OpenAI functions agent."""
        return self.runnable_agent.invoke(
            {"input": query},
            {"configurable": {"session_id": session_id}}
        )


class ToolCallingAgent(BaseAgent):
    """Tool calling agent implementation."""
    
    def __init__(self, config: AppConfig, tools: List[BaseTool], llm, prompt: ChatPromptTemplate, history_manager: MessageHistoryManager):
        super().__init__(config, tools, llm, prompt)
        self.history_manager = history_manager
        self.agent = self._create_agent()
        self.runnable_agent = self._create_runnable_agent()
    
    def _create_agent(self):
        """Create the tool calling agent."""
        return create_tool_calling_agent(self.llm, self.tools, self.prompt)
    
    def _create_runnable_agent(self):
        """Create a runnable agent with message history."""
        agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.agent.verbose,
            max_iterations=self.config.agent.max_iterations,
            early_stopping_method=self.config.agent.early_stopping_method
        )
        
        return RunnableWithMessageHistory(
            agent_executor,
            self.history_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    def get_agent_type(self) -> str:
        return "tool_calling"
    
    def _execute_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute query using tool calling agent."""
        return self.runnable_agent.invoke(
            {"input": query},
            {"configurable": {"session_id": session_id}}
        )


class AgentExecutionError(Exception):
    """Exception raised when agent execution fails."""
    pass


class AgentFactory:
    """
    Factory class for creating and managing agent instances.
    
    Provides centralized agent creation with proper configuration,
    dependency injection, and lifecycle management.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = AgentLogger.get_logger(__name__)
        self.history_manager = MessageHistoryManager()
        self._agents: Dict[str, BaseAgent] = {}
    
    def create_agent(
        self,
        agent_type: AgentType,
        tools: List[BaseTool],
        llm,
        tables_info: str,
        agent_id: Optional[str] = None
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            tools: List of tools available to the agent
            llm: Language model instance
            tables_info: Information about available database tables
            agent_id: Optional unique identifier for the agent
            
        Returns:
            Configured agent instance
        """
        if agent_id and agent_id in self._agents:
            self.logger.info(f"Returning existing agent: {agent_id}")
            return self._agents[agent_id]
        
        try:
            # Create prompt template
            prompt = self._create_prompt(tables_info)
            
            # Create agent based on type
            if agent_type == AgentType.OPENAI_FUNCTIONS:
                agent = OpenAIFunctionsAgent(
                    config=self.config,
                    tools=tools,
                    llm=llm,
                    prompt=prompt,
                    history_manager=self.history_manager
                )
            elif agent_type == AgentType.TOOL_CALLING:
                agent = ToolCallingAgent(
                    config=self.config,
                    tools=tools,
                    llm=llm,
                    prompt=prompt,
                    history_manager=self.history_manager
                )
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
            
            # Store agent if ID provided
            if agent_id:
                self._agents[agent_id] = agent
            
            log_agent_action(
                agent.get_agent_type(),
                "agent_created",
                {"tools_count": len(tools), "agent_id": agent_id}
            )
            
            return agent
            
        except Exception as e:
            log_error(e, f"Failed to create agent of type {agent_type}")
            raise AgentCreationError(f"Failed to create agent: {str(e)}") from e
    
    def _create_prompt(self, tables_info: str) -> ChatPromptTemplate:
        """Create the prompt template for agents."""
        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful database assistant.\n"
                    f"The database has tables of: {tables_info}\n"
                    "Do not make any assumptions about what tables exist "
                    "or what columns exist. Instead, use the describe_tables function.\n"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
            input_variables=["input"],
        )
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an existing agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all created agents and their configurations."""
        return [
            {"agent_id": agent_id, **agent.get_config()}
            for agent_id, agent in self._agents.items()
        ]
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent by ID."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.logger.info(f"Removed agent: {agent_id}")
            return True
        return False
    
    def clear_all_agents(self):
        """Remove all agents."""
        self._agents.clear()
        self.history_manager.clear_all_sessions()
        self.logger.info("Cleared all agents")


class AgentCreationError(Exception):
    """Exception raised when agent creation fails."""
    pass