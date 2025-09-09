"""
Agent configuration management module.

This module provides centralized configuration management for agent operations,
including model settings, tool configurations, and runtime parameters.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path


class AgentType(Enum):
    """Available agent types."""
    OPENAI_FUNCTIONS = "openai_functions"
    TOOL_CALLING = "tool_calling"


class ModelProvider(Enum):
    """Available model providers."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "../db/db.sqlite"))
    connection_timeout: int = field(default_factory=lambda: int(os.getenv("DB_TIMEOUT", "30")))
    
    def get_absolute_path(self) -> str:
        """Get absolute database path."""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / self.db_path.lstrip("../"))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_path: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    colored: bool = field(default_factory=lambda: os.getenv("LOG_COLORED", "true").lower() == "true")
    
    def get_level(self) -> int:
        """Convert string level to logging level."""
        return getattr(logging, self.level.upper(), logging.INFO)


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: ModelProvider = field(default_factory=lambda: ModelProvider(os.getenv("DEFAULT_MODEL_PROVIDER", "deepseek")))
    temperature: float = field(default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.0")))
    max_tokens: Optional[int] = field(default_factory=lambda: int(os.getenv("MODEL_MAX_TOKENS", "4000")) if os.getenv("MODEL_MAX_TOKENS") else None)
    timeout: int = field(default_factory=lambda: int(os.getenv("MODEL_TIMEOUT", "120")))


@dataclass
class ToolConfig:
    """Tool configuration."""
    enabled_tools: List[str] = field(default_factory=lambda: os.getenv("ENABLED_TOOLS", "sql,report,describe_tables").split(","))
    sql_timeout: int = field(default_factory=lambda: int(os.getenv("SQL_TIMEOUT", "30")))
    report_dir: str = field(default_factory=lambda: os.getenv("REPORT_DIR", "reports"))
    
    def get_report_path(self) -> Path:
        """Get absolute report directory path."""
        current_dir = Path(__file__).parent.parent.parent
        return current_dir / self.report_dir


@dataclass
class AgentConfig:
    """Main agent configuration."""
    agent_type: AgentType = field(default_factory=lambda: AgentType(os.getenv("DEFAULT_AGENT_TYPE", "tool_calling")))
    verbose: bool = field(default_factory=lambda: os.getenv("AGENT_VERBOSE", "false").lower() == "true")
    session_timeout: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT", "3600")))
    max_iterations: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "15")))
    early_stopping_method: str = field(default_factory=lambda: os.getenv("AGENT_EARLY_STOPPING", "generate"))


@dataclass
class AppConfig:
    """Application configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate database path exists (create parent directories if needed)
        db_path = Path(self.database.get_absolute_path())
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate report directory exists (create if needed)
        report_path = self.tools.get_report_path()
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Validate model settings
        if self.model.temperature < 0 or self.model.temperature > 2:
            raise ValueError(f"Invalid temperature: {self.model.temperature}. Must be between 0 and 2.")
        
        if self.model.max_tokens and self.model.max_tokens < 1:
            raise ValueError(f"Invalid max_tokens: {self.model.max_tokens}. Must be positive.")
        
        # Validate enabled tools
        valid_tools = {"sql", "report", "describe_tables"}
        invalid_tools = set(self.tools.enabled_tools) - valid_tools
        if invalid_tools:
            raise ValueError(f"Invalid tools specified: {invalid_tools}. Valid tools: {valid_tools}")
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": {
                "db_path": self.database.db_path,
                "connection_timeout": self.database.connection_timeout,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
            },
            "model": {
                "provider": self.model.provider.value,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "timeout": self.model.timeout,
            },
            "tools": {
                "enabled_tools": self.tools.enabled_tools,
                "sql_timeout": self.tools.sql_timeout,
                "report_dir": self.tools.report_dir,
            },
            "agent": {
                "agent_type": self.agent.agent_type.value,
                "verbose": self.agent.verbose,
                "session_timeout": self.agent.session_timeout,
                "max_iterations": self.agent.max_iterations,
                "early_stopping_method": self.agent.early_stopping_method,
            }
        }


def get_config() -> AppConfig:
    """Get application configuration instance."""
    return AppConfig.from_env()


# Global configuration instance
config = get_config()