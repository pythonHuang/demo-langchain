from .onellm import model

from  ..tools.funcall import toolsDefault,funcallChain
from  ..tools.rag import get_retriever
from  ..tools.util import list_files_in_directory

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Action(BaseModel):
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")


import json
from typing import List, Optional, Tuple
from langchain.memory.chat_memory import BaseChatMemory

from langchain_core.tools import tool
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.schema.output_parser import StrOutputParser
from pydantic import ValidationError

from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory

def queryRqg(fileName: str, query:str) -> str:
    """向知识库提问"""
    fileName=f'./data/{fileName}'
    retriever=get_retriever(fileName=fileName)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(
            temperature=0,
            model_kwargs={
                "seed": 42
            },
        ),  # 语言模型
        chain_type="stuff",  # prompt的组织方式，后面细讲
        retriever=retriever  # 检索器
    )
    return qa_chain.invoke(query)
@tool
def queryRqg_tool(fileName: str, query:str) -> str:
    """向知识库提问"""
    
    return queryRqg_tool(fileName,query)

from langchain.tools import StructuredTool

rqgIndexList=list_files_in_directory('./data/')
print(rqgIndexList)
document_qa_tool = StructuredTool.from_function(
    func=queryRqg,
    name="AskDocument",
    description="""根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。
    
    文件名列表： 
    {0}""".format(rqgIndexList),
)

class AutoGPT():
    
    llm:BaseChatModel=None
    tools=None
    work_dir:str=""
    main_prompt_file:str=""
    ask_prompt_file:str=""
    raq_prompt_file:str=""
    funcall_prompt_file:str=""
    
    main_prompt:str=""
    funcall_prompt:str=""
    raq_prompt:str=""
    funcall_prompt:str=""
    @staticmethod
    def __chinese_friendly(string) -> str:
        lines = string.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('{') and line.endswith('}'):
                try:
                    lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
                except:
                    pass
        return '\n'.join(lines)

    @staticmethod
    def __format_long_term_memory(task_description: str, memory: BaseChatMemory) -> str:
        return memory.load_memory_variables(
            {"prompt": task_description}
        )["history"]

    @staticmethod
    def __format_short_term_memory(memory: BaseChatMemory) -> str:
        messages = memory.chat_memory.messages
        string_messages = [messages[i].content for i in range(1, len(messages))]
        return "\n".join(string_messages)

    def __init__(self,llm:BaseChatModel=model,
                 tools=toolsDefault,
                 work_dir='./data/',
                 main_prompt_file='./prompts/main/main.txt',
                 raq_prompt_file='./prompts/main/rqg.txt',
                 final_prompt_file='./prompts/main/final_step.txt',
                 funcall_prompt_file='./prompts/main/funcall.txt',
                 max_thought_steps: int= 4,
        ):
        
        self.llm=llm
        self.tools=tools
        
        self.tools.append(queryRqg)
        
        self.work_dir=work_dir
        # OutputFixingParser： 如果输出格式不正确，尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)

        self.main_prompt_file = main_prompt_file
        self.raq_prompt_file = raq_prompt_file
        self.final_prompt_file = final_prompt_file
        self.funcall_prompt_file = funcall_prompt_file
        self.max_thought_steps=max_thought_steps
        
        self.__init_prompt_templates()
        self.__init_chains(tools)
    
    def __init_prompt_templates(self):
        rqgIndexList=self.__getRagIndex_prompt(self.work_dir)
        
        
        self.final_prompt = PromptTemplate.from_file(
            self.final_prompt_file
        )
        self.raq_prompt=PromptTemplate.from_file(
            self.raq_prompt_file
        )
        self.main_prompt = PromptTemplate.from_file(
            self.main_prompt_file
        ).partial(
            work_dir=self.work_dir,
            tools=render_text_description(self.tools),
            format_instructions=self.__chinese_friendly(
                self.output_parser.get_format_instructions(),
            )
        )
        self.funcall_prompt = PromptTemplate.from_file(
            self.funcall_prompt_file
        ).partial(
            tools=render_text_description(self.tools),
            format_instructions=self.__chinese_friendly(
                self.output_parser.get_format_instructions(),
            )
        )
        
    def __init_chains(self,tools):
        # 主流程的chain
        self.main_chain = (self.main_prompt | self.llm | StrOutputParser())
        # 提出问题的chain
        self.raq_chain = (self.raq_prompt | self.llm | StrOutputParser())
        # 最终一步的chain
        self.final_chain = (self.final_prompt | self.llm | StrOutputParser())
        
        self.funcall_chain =  (self.funcall_prompt | funcallChain(self.llm,tools))
        
        
    def run(self,query:str, verbose=False):
        # 初始化短时记忆
        short_term_memory = self.__init_short_term_memory()
        # 连接长时记忆（如果有）
        #long_term_memory = self.__connect_long_term_memory()

        # 思考步数
        thought_step_count = 0
        action=None
        response=""
        
        return self.funcall_chain.stream(query)
        
    
    def __askRAQTool(query:str,type:str):
        """向知识库提问
        """
        fileName=f'./data/{type}.pdf'
        retriever=get_retriever(fileName=fileName)
        return retriever.invoke(query)
    def __getRagIndex_prompt(self,path):
        return list_files_in_directory(path)
    
    def __exec_action(self, action: Action) -> str:
        # 查找工具
        tool = self.__find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "+
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    def __init_short_term_memory(self) -> BaseChatMemory:
        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
        )
        short_term_memory.save_context(
            {"input": "\n初始化"},
            {"output": "\n开始"}
        )
        return short_term_memory

    # def __connect_long_term_memory(self) -> BaseMemory:
    #     if self.memery_retriever is not None:
    #         long_term_memory = VectorStoreRetrieverMemory(
    #             retriever=self.memery_retriever,
    #         )
    #     else:
    #         long_term_memory = None
    #     return long_term_memory

        

def test(query:str="你是谁？"):
    agent=AutoGPT()
    return agent.run(query)

if __name__ == "main":
    result=test("你是谁？")
    print(result)