# 🦜️🔗 Data Analyzer LLM

This repository contains reference implementations of various LangChain agents interacted with using Streamlit.
The main goal of the project is to build a tool that can connect to various data sources(csv/text/api) and generate the query/code needed to fetch the necessary information pertaining to the user's prompt.

Currently using OpenAI's API and embeddings, will eventually support using Llama models locally.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Activate poetry shell
$ poetry shell
```

## Running

```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run main.py
```