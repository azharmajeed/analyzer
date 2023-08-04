# 🦜️🔗 Data Analyzer LLM

This repository contains reference implementations of various LangChain agents interacted with using Streamlit.
The main goal of the project is to build a tool that can connect to various data sources(csv/text/api) and generate the query/code needed to fetch the necessary information pertaining to the user's prompt.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Install git pre-commit hooks
$ poetry shell
```

## Running

```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run streamlit_agent/mrkl_demo.py
```