from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
def loadUrl(url:str):
    loader = WebBaseLoader(url or "https://docs.smith.langchain.com")
    docs = loader.load()
    return docs

from langchain_community.embeddings import HuggingFaceEmbeddings



from langchain_community.vectorstores import FAISS
def getRetriever(docs):
    
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever



from langchain.tools.retriever import create_retriever_tool

def getRagTool(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "搜索有关LangSmith的信息。关于LangSmith的任何问题，您都可以使用这个工具",
    )

#另一个搜索工具
#export TAVILY_API_KEY=...
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

docs=loadUrl()
retriever=getRetriever(docs)
retriever_tool=getRagTool(retriever)
tools = [retriever_tool, search]

#创建智能体来使用这些工具
#pip install langchainhub
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.agents import Tool,initialize_agent

from .onellm import model

def initAgent(prompt=None, model=model,tools=tools):
    # Get the prompt to use - you can modify this!
    if prompt is None:
        prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor



def initAgent2(prompt=None, model=model,tools=tools):
    # Get the prompt to use - you can modify this!
    if prompt is None:
        prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = initialize_agent(
        tools,
        model, 
        agent="zero-shot-react-description",
        agent_kwargs=dict(subffix=""+prompt.suffix),
        verbose=True,
        return_intermediate_steps=True,
    )
    return agent

#调用
def test(query):
    agent_executor=initAgent()
    return agent_executor.stream({"input": query or "langsmith如何帮助测试?"})
