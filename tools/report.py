from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def generate_report(filename, html):
    with open(filename, "w") as f:
        f.write(html)


class GenerateReportArgsSchema(BaseModel):
    filename: str
    html: str


generate_report_tool = StructuredTool.from_function(
    name="generate_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report to be generated.",
    func=generate_report,
    args_schema=GenerateReportArgsSchema,
)
