
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
)
from langchain.prompts import (
    ChatPromptTemplate,
    # HumanMessagePromptTemplate,
    # SystemMessagePromptTemplate,
    # AIMessagePromptTemplate,
    # PromptTemplate,
    # MessagesPlaceholder
)
# from langchain_community.retrievers import RetrievalQA
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from langchain.schema import Document
from typing import List

# from textsplitter import ChineseTextSplitter
class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        elif ext == "md" or ext == "markdown":
            return UnstructuredMarkdownLoader(filename)
        elif ext == "txt":
            return TextLoader(filename,autodetect_encoding=True)
        elif ext == "csv":
            return CSVLoader(filename,autodetect_encoding=True)
        elif ext == "json":
            return JSONLoader(filename,autodetect_encoding=True)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")

def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]


def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    
    # ispdf=filename.endswith("pdf")
    # textsplitter = ChineseTextSplitter(pdf=ispdf, sentence_size=100)
    # pages = file_loader.load_and_split(text_splitter=textsplitter)
    pages = file_loader.load_and_split()
    return pages


def importDB(path="./data/LlamaIndex.pdf"):#医保SDK和上传下载调用示例.docx
    
    if path.index("/")==-1:
        path=f"./data/{path}"
    # 加载文档
    # loader = PyPDFLoader(path)
    # pages = loader.load_and_split()
    pages = load_docs(path)
    
    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    texts = text_splitter.create_documents(
        [page.page_content for page in pages[:4]]
    )

    # 灌库
    embeddings = OpenAIEmbeddings()
    curdb = Chroma.from_documents(texts, embeddings)
    return curdb

db=None

def get_retriever(top: int = 2,curDb=db,fileName=""):
    if fileName:
        curDb=importDB(fileName)
    elif curDb is None:
        curDb=importDB()
        
    # 检索 top-1 结果
    curretriever = curDb.as_retriever(search_kwargs={"k": top})
    return curretriever
def get_noneStr(data:str):
    return data

class MyRunnable:
    def __init__(self, name):
        self.name = name
 
    def run(self):
        print(f"Running {self.name}...")
def testRAG(query="",useRqg=True,fileName=""):
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from ..llm.onellm import model
    from langchain.tools import StructuredTool
    # from langchain import Chain, LambdaExecutionPolicy
    # from langchain import PromptTemplate, LambdaExecutor, LambdaChain
    # executor = StructuredTool(
    #     lambda x: "",
    # )
    str_qa_tool = StructuredTool.from_function(
        func=get_noneStr,
        name="get_noneStr",
        description="获取字符串。",
    )
    # Prompt模板
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever=str_qa_tool
    if(useRqg):
        if fileName:
            retriever=get_retriever(fileName=fileName)
        else:
            retriever=get_retriever()
    else:
         # Prompt模板
        template = """你是一个陪聊机器人，积极乐观地回应用户，耐心地一步步帮用户分析问题，让对话持续下去，并且经常询问用户的看法，像一个高情商聊天大师一样，不要冷场。如果知道用户的名字，就要经常亲切地称呼他。如果不知道名字，就称呼\"亲爱的\"

        Question: {question}
        """
    # Chain
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt
        | model
        | StrOutputParser()
    )

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=model,  # 语言模型
    #     chain_type="stuff",  # prompt的组织方式，后面细讲
    #     retriever=db.as_retriever()  # 检索器
    # )
    # response = qa_chain.run(query + "(请用中文回答)")
    
    # return rag_chain.invoke(query or "Llama 2有多少参数")
    return rag_chain.stream(query or "Llama 2有多少参数")
    # for res in rag_chain.stream(query or "Llama 2有多少参数"):
    #     yield res

if __name__ == "__main__":
    testRAG()
