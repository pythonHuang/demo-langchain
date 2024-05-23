from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
embeddings=None
try:
    embeddings = HuggingFaceEmbeddings()
except :
    embeddings = OpenAIEmbeddings()
    pass

# pip install faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    # HumanMessagePromptTemplate,
    # SystemMessagePromptTemplate,
    # AIMessagePromptTemplate,
    # PromptTemplate,
    # MessagesPlaceholder
)

def loadURL(url:str):
    """获取数据

    Args:
        url (str): _description_

    Returns:
        _type_: _description_
    """
    loader = WebBaseLoader(url or "https://docs.smith.langchain.com")

    docs = loader.load()
    return docs

def getEmbeddings():
    """获取嵌入向量模型
向量
    Returns:
        _type_: _description_
    """
    return embeddings

def getVectorStore(docs,embeddings):
    """文档向量化 并存存储到本地向量数据库FAISS  灌库"""
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    return vector

def getVectorChain(prompt,llm):
    """获取问答链"""
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

def getRetrievalChain(vector,document_chain,top:int=2):
    """检索 链"""
    retriever = vector.as_retriever(search_kwargs={"k": top})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def testRaq(query=""):
    docs=loadURL("http://docs.smith.langchain.com")
    embeddings=getEmbeddings()
    
    vector=getVectorStore(docs,embeddings)
    prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")
    
    llm=ChatOpenAI()
    document_chain=getVectorChain(prompt,llm)
    retrieval_chain=getRetrievalChain(vector,document_chain)
    response = retrieval_chain.invoke({"input": query or "langsmith如何帮助测试?"})
    return response
if __name__ == "__main__":
    response=testRaq()
    print(response["answer"])
    