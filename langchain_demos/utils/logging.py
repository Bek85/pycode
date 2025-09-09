"""
Logging utilities for the agent system.

This module provides centralized logging configuration and utilities
for consistent logging across the application with colored output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from ..config import LoggingConfig


class ColorCodes:
    """ANSI color codes for terminal output."""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Bright colors
    BRIGHT_RED = '\033[1;91m'
    BRIGHT_GREEN = '\033[1;92m'
    BRIGHT_YELLOW = '\033[1;93m'
    BRIGHT_BLUE = '\033[1;94m'
    BRIGHT_MAGENTA = '\033[1;95m'
    BRIGHT_CYAN = '\033[1;96m'
    BRIGHT_WHITE = '\033[1;97m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels and different log types."""
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.DIM + ColorCodes.WHITE,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.BRIGHT_RED + ColorCodes.BOLD,
    }
    
    # Color mapping for specific log categories
    CATEGORY_COLORS = {
        'Agent': ColorCodes.BRIGHT_BLUE,
        'Performance': ColorCodes.BRIGHT_MAGENTA,
        'Calling': ColorCodes.CYAN,
        'result': ColorCodes.BRIGHT_GREEN,
        'Error': ColorCodes.BRIGHT_RED,
        'initialized': ColorCodes.BRIGHT_GREEN,
        'created': ColorCodes.BLUE,
        'execute_query': ColorCodes.BRIGHT_CYAN,
        'started': ColorCodes.GREEN,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def format(self, record):
        # Format the basic message
        formatted_message = super().format(record)
        
        # Don't add colors if output is being redirected (not a terminal)
        if not sys.stdout.isatty():
            return formatted_message
        
        # Get base color for log level
        level_color = self.LEVEL_COLORS.get(record.levelno, ColorCodes.WHITE)
        
        # Apply level color to timestamp and level name
        parts = formatted_message.split(' - ', 3)
        if len(parts) >= 4:
            timestamp, name, level, message = parts[0], parts[1], parts[2], ' - '.join(parts[3:])
            
            # Color the timestamp
            colored_timestamp = f"{ColorCodes.DIM}{timestamp}{ColorCodes.RESET}"
            
            # Color the logger name
            colored_name = f"{ColorCodes.DIM}{name}{ColorCodes.RESET}"
            
            # Color the level
            colored_level = f"{level_color}{level}{ColorCodes.RESET}"
            
            # Color specific parts of the message
            colored_message = self._colorize_message(message, record.levelno)
            
            return f"{colored_timestamp} - {colored_name} - {colored_level} - {colored_message}"
        else:
            # Fallback: just color the entire message with level color
            return f"{level_color}{formatted_message}{ColorCodes.RESET}"
    
    def _colorize_message(self, message, log_level=None):
        """Apply colors to specific parts of the message based on content."""
        colored_message = message
        
        # If this is an error or critical message, make the entire message red
        if log_level and log_level >= logging.ERROR:
            return f"{ColorCodes.BRIGHT_RED}{message}{ColorCodes.RESET}"
        
        # Apply category-specific colors
        for category, color in self.CATEGORY_COLORS.items():
            if category in message:
                if category == 'Agent':
                    # Special handling for Agent [...] patterns
                    import re
                    pattern = r'Agent \[([^\]]+)\]'
                    colored_message = re.sub(
                        pattern, 
                        f"{color}Agent [{ColorCodes.BOLD}\\1{ColorCodes.RESET}{color}]{ColorCodes.RESET}",
                        colored_message
                    )
                elif category == 'Performance':
                    # Highlight performance metrics
                    import re
                    pattern = r'Performance - ([^:]+): ([0-9.]+s)'
                    colored_message = re.sub(
                        pattern,
                        f"{color}Performance{ColorCodes.RESET} - {ColorCodes.BOLD}\\1{ColorCodes.RESET}: {ColorCodes.BRIGHT_YELLOW}\\2{ColorCodes.RESET}",
                        colored_message
                    )
                elif category == 'Calling':
                    # Highlight function calls
                    if message.startswith('Calling '):
                        func_name = message.split(' ', 2)[1]
                        colored_message = colored_message.replace(
                            f'Calling {func_name}',
                            f"{color}Calling {ColorCodes.BOLD}{func_name}{ColorCodes.RESET}"
                        )
                elif category == 'result':
                    # Highlight results
                    if ' result: ' in message:
                        parts = message.split(' result: ', 1)
                        if len(parts) == 2:
                            colored_message = f"{parts[0]} {color}result{ColorCodes.RESET}: {ColorCodes.BRIGHT_WHITE}{parts[1]}{ColorCodes.RESET}"
                else:
                    # General category highlighting
                    colored_message = colored_message.replace(
                        category,
                        f"{color}{category}{ColorCodes.RESET}"
                    )
        
        # Highlight numbers and times
        import re
        # Highlight execution times
        colored_message = re.sub(
            r'(\d+\.\d+s)',
            f"{ColorCodes.BRIGHT_YELLOW}\\1{ColorCodes.RESET}",
            colored_message
        )
        
        # Highlight counts and numbers in contexts
        colored_message = re.sub(
            r'(\d+) (tables?|rows?|tools?)',
            f"{ColorCodes.BRIGHT_CYAN}\\1{ColorCodes.RESET} \\2",
            colored_message
        )
        
        # Highlight query previews
        if 'query_preview' in colored_message:
            import re
            colored_message = re.sub(
                r"'query_preview': '([^']+)'",
                f"'query_preview': '{ColorCodes.BRIGHT_YELLOW}\\1{ColorCodes.RESET}'",
                colored_message
            )
        
        return colored_message


class AgentLogger:
    """
    Centralized logger for agent operations.
    
    Provides structured logging with proper formatting and
    configurable output destinations.
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically __name__ of calling module)
            config: Optional logging configuration
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        if config is None:
            from ..config import config as app_config
            config = app_config.logging
        
        logger = logging.getLogger(name)
        logger.setLevel(config.get_level())
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        colored_formatter = ColoredFormatter(config.format)
        plain_formatter = logging.Formatter(config.format)
        
        # Console handler with optional colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.get_level())
        
        # Use colored formatter only if enabled and stdout is a terminal
        if config.colored and sys.stdout.isatty():
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(plain_formatter)
            
        logger.addHandler(console_handler)
        
        # File handler (if configured)
        if config.file_path:
            file_path = Path(config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(config.get_level())
            file_handler.setFormatter(plain_formatter)  # Use plain formatter for files
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_logging(cls, config: LoggingConfig):
        """
        Setup global logging configuration.
        
        Args:
            config: Logging configuration
        """
        # Configure root logger
        logging.basicConfig(
            level=config.get_level(),
            format=config.format,
            handlers=[]
        )
        
        # Suppress overly verbose third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def log_function_call(func_name: str, args: dict = None, result: str = None):
    """
    Log function call with parameters and result.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        result: Function result or error message
    """
    logger = AgentLogger.get_logger(__name__)
    
    if args:
        logger.info(f"Calling {func_name} with args: {args}")
    else:
        logger.info(f"Calling {func_name}")
    
    if result:
        # Truncate long results for readability
        display_result = result[:200] + "..." if len(str(result)) > 200 else result
        logger.info(f"{func_name} result: {display_result}")


def log_agent_action(agent_type: str, action: str, details: dict = None):
    """
    Log agent actions for monitoring and debugging.
    
    Args:
        agent_type: Type of agent (e.g., "openai_functions", "tool_calling")
        action: Action being performed
        details: Additional details about the action
    """
    logger = AgentLogger.get_logger(__name__)
    
    log_msg = f"Agent [{agent_type}] - {action}"
    if details:
        log_msg += f" - Details: {details}"
    
    logger.info(log_msg)


def log_error(error: Exception, context: str = None):
    """
    Log error with context information.
    
    Args:
        error: Exception that occurred
        context: Additional context about where the error occurred
    """
    logger = AgentLogger.get_logger(__name__)
    
    if context:
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    else:
        logger.error(f"Error: {str(error)}", exc_info=True)


def log_performance(operation: str, duration: float, details: dict = None):
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        details: Additional performance details
    """
    logger = AgentLogger.get_logger(__name__)
    
    log_msg = f"Performance - {operation}: {duration:.2f}s"
    if details:
        log_msg += f" - Details: {details}"
    
    logger.info(log_msg)


def colorize_result_output(text: str, force_colors: bool = True) -> str:
    """
    Colorize the final result output for better visibility.
    
    Args:
        text: The result text to colorize
        force_colors: Force colors even if not in terminal (default True for better UX)
        
    Returns:
        Colorized text if colors are enabled, otherwise plain text
    """
    import sys
    import re
    import os
    
    # Check if colors should be applied
    if not force_colors and not sys.stdout.isatty():
        return text
    
    # Also check environment variable for color control
    if os.getenv('NO_COLOR') or os.getenv('LOG_COLORED', 'true').lower() == 'false':
        return text
    
    # Simple and effective approach - process specific patterns only
    colored_text = text
    
    # 1. Handle markdown-style bold text (**text**) - this includes numbers inside
    def replace_bold(match):
        content = match.group(1)
        return f"{ColorCodes.BRIGHT_YELLOW}{ColorCodes.BOLD}{content}{ColorCodes.RESET}"
    
    colored_text = re.sub(r'\*\*([^*]+)\*\*', replace_bold, colored_text)
    
    # 2. Handle currency values
    colored_text = re.sub(
        r'(\$[\d,]+(?:\.\d+)?)',
        f"{ColorCodes.BRIGHT_GREEN}\\1{ColorCodes.RESET}",
        colored_text
    )
    
    # 3. Handle dates
    colored_text = re.sub(
        r'(\d{4}-\d{2}-\d{2})',
        f"{ColorCodes.BRIGHT_MAGENTA}\\1{ColorCodes.RESET}",
        colored_text
    )
    
    # 4. Handle table borders
    colored_text = re.sub(
        r'(\|[^|\n]*\|)',
        f"{ColorCodes.DIM}\\1{ColorCodes.RESET}",
        colored_text
    )
    
    # 5. Handle specific phrases
    colored_text = re.sub(
        r'\b(There are)\b',
        f"{ColorCodes.GREEN}\\1{ColorCodes.RESET}",
        colored_text
    )
    
    colored_text = re.sub(
        r'\b(in the database)\b',
        f"{ColorCodes.BLUE}\\1{ColorCodes.RESET}",
        colored_text
    )
    
    return colored_text


def colorize_execution_time(execution_time: float, force_colors: bool = True) -> str:
    """
    Colorize execution time output.
    
    Args:
        execution_time: Execution time in seconds
        force_colors: Force colors even if not in terminal (default True for better UX)
        
    Returns:
        Colorized execution time string
    """
    import sys
    import os
    
    # Check if colors should be applied
    if not force_colors and not sys.stdout.isatty():
        return f"Execution time: {execution_time:.2f}s"
    
    # Also check environment variable for color control
    if os.getenv('NO_COLOR') or os.getenv('LOG_COLORED', 'true').lower() == 'false':
        return f"Execution time: {execution_time:.2f}s"
    
    # Color based on execution time
    if execution_time < 1.0:
        color = ColorCodes.BRIGHT_GREEN  # Fast
    elif execution_time < 5.0:
        color = ColorCodes.GREEN  # Good
    elif execution_time < 15.0:
        color = ColorCodes.YELLOW  # Moderate
    else:
        color = ColorCodes.RED  # Slow
    
    return f"{ColorCodes.DIM}Execution time: {color}{execution_time:.2f}s{ColorCodes.RESET}"