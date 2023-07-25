from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
import os
from dotenv import load_dotenv

load_dotenv()

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
]


memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)
agent_chain.run(input="hi, i am bob")
agent_chain.run(input="what's my name?")
agent_chain.run("what are some good dinners to make this week, if i like thai food?")
agent_chain.run("could you summarize some good Thai recipies?")
