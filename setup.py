"""Setup configuration for PyCode package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "PyCode - AI and LangChain Learning Project"

setup(
    name="pycode",
    version="1.0.0",
    author="Alex",
    author_email="alex@example.com",
    description="A collection of Python applications demonstrating AI/ML concepts and LangChain integrations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alex/pycode",  # Update with your actual repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "pycode-chat=pycode.chat_apps.tchat_gpt:main",
            "pycode-agents=pycode.langchain_demos.agents:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pycode": ["*.txt", "*.json", "*.env.example"],
    },
)