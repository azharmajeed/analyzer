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

llm = OpenAI(
    temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=True
)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)


# llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
# search = SerpAPIWrapper()
# llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
# tools = [
#     Tool(
#         name="Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events. You should ask targeted questions",
#     ),
#     Tool(
#         name="Calculator",
#         func=llm_math_chain.run,
#         description="useful for when you need to answer questions about math",
#     ),
#     Tool(
#         name="FooBar DB",
#         func=db_chain.run,
#         description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
#     ),
# ]

# mrkl = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

# mrkl.run("?")
