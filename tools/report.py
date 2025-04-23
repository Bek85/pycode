from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel
import os


def generate_report(filename, html):
    # Get the current script's directory (project root)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Create the reports directory path
    reports_dir = os.path.join(current_dir, "reports")
    # Ensure reports directory exists
    os.makedirs(reports_dir, exist_ok=True)
    # Create the full file path
    file_path = os.path.join(reports_dir, f"{filename}.html")

    # Write the file
    with open(file_path, "w") as f:
        f.write(html)

    print(f"Report saved to: {file_path}")


class GenerateReportArgsSchema(BaseModel):
    filename: str
    html: str


generate_report_tool = StructuredTool.from_function(
    name="generate_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report to be generated.",
    func=generate_report,
    args_schema=GenerateReportArgsSchema,
)
