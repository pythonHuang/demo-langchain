from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "搜索有关LangSmith的信息。关于LangSmith的任何问题，您都可以使用这个工具",
)

#另一个搜索工具
#export TAVILY_API_KEY=...
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()


tools = [retriever_tool, search]

#创建智能体来使用这些工具
#pip install langchainhub
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


#调用
agent_executor.invoke({"input": "langsmith如何帮助测试?"})