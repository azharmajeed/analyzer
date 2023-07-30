from langchain.agents import (
    Tool,
    AgentType,
    initialize_agent,
    AgentExecutor,
    LLMSingleActionAgent,
)
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import os
import pinecone
from dotenv import load_dotenv

from prompt_template import CustomOutputParser, CustomPromptTemplate

load_dotenv()

# Set up the base template
template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""


doc_path = str("./penal_code.pdf")
loader = PyPDFLoader(doc_path)
documents = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

docsearch = Chroma.from_documents(texts, embeddings, collection_name="penal_code")
penal_code = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
    Tool(
        name="Penal Code QA System",
        func=penal_code.run,
        description="useful for when you need to answer questions about the Indian Penal Code or if any issue is associated with it.",
    ),
]


memory = ConversationBufferMemory(memory_key="chat_history")

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)

output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# agent_chain = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
#     memory=memory,
# )
agent_executor.run(input="hi, can you tell me how many Indian Penal codes are there?")
agent_executor.run(
    input="elaborate the 6th penal code with an example- 6. Definitions in the Code to be understood subject to exceptions. "
)
agent_executor.run("Do you know the previous discussed penal code number ?")
