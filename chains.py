from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# llm = OpenAI(
#     temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=True
# )
# tools = load_tools(["ddg-search"])
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

from convo import main

agent = main(os.environ.get("OPENAI_API_KEY"))

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
