"""
Reporting tools with enhanced file management and error handling.

This module provides reporting capabilities with proper file management,
error handling, and security validation.
"""

import os
import time
from typing import Optional
from pathlib import Path
import html
import re

from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel

from ..config import AppConfig
from ..utils import AgentLogger, log_function_call, log_error, log_performance


class ReportingError(Exception):
    """Custom exception for reporting operations."""

    pass


class ReportService:
    """
    High-level reporting service with file management and validation.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = AgentLogger.get_logger(__name__)
        self.report_dir = config.tools.get_report_path()

        # Ensure reports directory exists
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, filename: str, html_content: str) -> str:
        """
        Generate an HTML report with proper validation and security.

        Args:
            filename: Name of the report file (without extension)
            html_content: HTML content to write

        Returns:
            Path to the generated report file

        Raises:
            ReportingError: If report generation fails
        """
        start_time = time.time()

        try:
            log_function_call(
                "generate_html_report",
                {"filename": filename, "content_length": len(html_content)},
            )

            # Validate and sanitize filename
            clean_filename = self._sanitize_filename(filename)
            if not clean_filename:
                raise ReportingError("Invalid filename provided")

            # Validate HTML content
            if not html_content.strip():
                raise ReportingError("Empty HTML content provided")

            # Create full file path
            file_path = self.report_dir / f"{clean_filename}.html"

            # Check if file exists and create backup if needed
            if file_path.exists():
                backup_path = (
                    self.report_dir / f"{clean_filename}_backup_{int(time.time())}.html"
                )
                file_path.rename(backup_path)
                self.logger.info(f"Created backup: {backup_path}")

            # Write the HTML file with proper error handling
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # Add basic HTML structure if not present
                    if not self._is_complete_html(html_content):
                        html_content = self._wrap_in_html_structure(
                            html_content, clean_filename
                        )

                    f.write(html_content)

                # Verify file was written successfully
                if not file_path.exists() or file_path.stat().st_size == 0:
                    raise ReportingError("File was not written successfully")

                log_performance(
                    "generate_html_report",
                    time.time() - start_time,
                    {"file_size": file_path.stat().st_size, "filename": clean_filename},
                )

                result = f"Report generated successfully: {file_path}"
                log_function_call("generate_html_report", result=result)

                return result

            except OSError as e:
                raise ReportingError(f"Failed to write file: {str(e)}") from e

        except Exception as e:
            log_error(e, f"Report generation failed for {filename}")
            if isinstance(e, ReportingError):
                raise
            else:
                raise ReportingError(
                    f"Unexpected error during report generation: {str(e)}"
                ) from e

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        if not filename:
            return ""

        # Remove or replace invalid characters
        filename = re.sub(r"[^\w\-_.]", "_", filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")

        # Limit length
        if len(filename) > 100:
            filename = filename[:100]

        # Ensure it's not empty after sanitization
        if not filename:
            filename = f"report_{int(time.time())}"

        return filename

    def _is_complete_html(self, content: str) -> bool:
        """
        Check if content is a complete HTML document.

        Args:
            content: HTML content to check

        Returns:
            True if content appears to be complete HTML
        """
        content_lower = content.lower()
        return (
            "<html>" in content_lower
            and "</html>" in content_lower
            and "<body>" in content_lower
            and "</body>" in content_lower
        )

    def _wrap_in_html_structure(self, content: str, title: str) -> str:
        """
        Wrap content in proper HTML structure.

        Args:
            content: HTML content to wrap
            title: Title for the HTML document

        Returns:
            Complete HTML document
        """
        # Escape the title to prevent XSS
        safe_title = html.escape(title)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .header {{
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="content">
        {content}
    </div>
</body>
</html>"""

    def list_reports(self) -> list:
        """
        List all generated reports.

        Returns:
            List of report file information
        """
        try:
            reports = []
            for file_path in self.report_dir.glob("*.html"):
                if file_path.is_file():
                    stat = file_path.stat()
                    reports.append(
                        {
                            "filename": file_path.name,
                            "size": stat.st_size,
                            "created": time.ctime(stat.st_ctime),
                            "modified": time.ctime(stat.st_mtime),
                            "path": str(file_path),
                        }
                    )

            return sorted(reports, key=lambda x: x["modified"], reverse=True)

        except Exception as e:
            log_error(e, "Failed to list reports")
            return []

    def delete_report(self, filename: str) -> bool:
        """
        Delete a specific report file.

        Args:
            filename: Name of the report file to delete

        Returns:
            True if file was deleted, False otherwise
        """
        try:
            clean_filename = self._sanitize_filename(filename.replace(".html", ""))
            file_path = self.report_dir / f"{clean_filename}.html"

            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted report: {file_path}")
                return True
            else:
                self.logger.warning(f"Report not found: {file_path}")
                return False

        except Exception as e:
            log_error(e, f"Failed to delete report: {filename}")
            return False


# Pydantic schema for tool validation
class GenerateReportArgsSchema(BaseModel):
    """Schema for generate_report tool arguments."""

    filename: str
    html_content: str


def create_reporting_tools(config: AppConfig) -> list:
    """
    Create reporting tools with proper configuration and error handling.

    Args:
        config: Application configuration

    Returns:
        List of configured reporting tools
    """
    report_service = ReportService(config)

    generate_report_tool = StructuredTool.from_function(
        name="generate_report",
        description=(
            "Write an HTML file to disk. Use this tool whenever someone asks for a report to be generated. "
            "The tool will automatically add proper HTML structure if needed and ensure file safety."
        ),
        func=report_service.generate_html_report,
        args_schema=GenerateReportArgsSchema,
    )

    return [generate_report_tool]
