from langchain.agents import (
    Tool,
    AgentType,
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
import os
import pinecone
from dotenv import load_dotenv

from prompt_template import CustomOutputParser, CustomPromptTemplate


def main(openai_api_key: str) -> AgentExecutor:
    # Set up the base template
    template = """Answer the following questions as best you can, you are an assistant on Indian Law and Governemnt policy topics. You have access to the following tools:

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

    This is the history of our conversation:
    {history}

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

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    # setup vector store
    docsearch = Chroma.from_documents(texts, embeddings, collection_name="penal_code")
    penal_code = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )

    # setup for web search
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

    # setup for csv agent
    # setup for sql agent

    memory = ConversationBufferWindowMemory(k=3)

    prompt_with_history = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"],
    )

    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
    )
    return agent_executor


# agent_chain = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
#     memory=memory,
# )


if __name__ == "__main__":
    load_dotenv()

    agent_executor = main(os.environ.get("OPENAI_API_KEY"))

    agent_executor.run(
        input="hi, can you tell me how many sections and chapters are there in the Indian Penal code ?"
    )
    agent_executor.run(input="what are those 4 penal codes you found?")
    agent_executor.run(
        input="elaborate the 6th penal code with an example- 6. Definitions in the Code to be understood subject to exceptions. "
    )
    agent_executor.run("Search the web for what is the date today ?")
    agent_executor.run(input="Hi, my name is Azhar Majeed")
