from datetime import datetime
import pandas as pd
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from io import StringIO

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents.agent_types import AgentType
from file_handler import PDFHandler


load_dotenv()


def update_counter(desc, files):
    st.session_state.count += 1
    st.session_state.last_updated = datetime.now().time()
    st.session_state.files_uploaded = files
    if len(files) > 0 and (
        (desc not in st.session_state.desc_file_dict) and (files[-1] not in files)
    ):
        st.session_state.desc_file_dict[desc] = files[-1]
        if files[-1].type == "text/csv":
            st.session_state.csv_files.append((desc, pd.read_csv(files[-1])))
        if files[-1].type == "application/pdf":
            st.session_state.pdf_files.append((desc, PDFHandler.read_file(files[-1])))
        st.session_state.desc_list.append(desc)


PROMPT = "You are Python Pandas coding agent, whose job is to convert natural text to pandas "


# Generate LLM response
def generate_response(api_key, dfs, input_query):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=api_key
    )
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(
        llm,
        dfs,
        verbose=True,
        include_df_in_prompt=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    # Perform Query using the Agent
    response = agent.run(input_query)
    return response


st.title("🦜🔗 Quickstart App")
# Session State also supports the attribute based syntax
if "key" not in st.session_state:
    st.session_state.key = "value"
    st.session_state.count = 0
    st.session_state.last_updated = datetime.now().time()
    st.session_state.desc_file_dict = {}
    st.session_state.files_uploaded = []
    st.session_state.desc_list = []
    st.session_state.csv_files = []
    st.session_state.pdf_files = []


# openai_api_key = os.environ.get("OPENAI_API_KEY")

# csv files
desc_list = []
with st.form("file input"):
    file_desc = st.text_input("File Description")
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    increment = st.form_submit_button(
        "Submit", on_click=update_counter(file_desc, uploaded_files)
    )

# write the csv files to streamlit
for i in st.session_state.csv_files:
    with st.expander("See DataFrame"):
        st.write(i[1])

openai_api_key = st.text_input("OpenAI API Key", type="password")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the data:")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = generate_response(
            openai_api_key,
            [i[1] for i in st.session_state.csv_files],
            prompt,
        )
        st.markdown(response)


# st.write("Last Updated = ", st.session_state.last_updated)

# memory of previous converstion
# embeddings from uploaded documents
# ability to query a database
# ability to search the web
# custom prompt from the collation of the above results


# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.get("content", "")
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
