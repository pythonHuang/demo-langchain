from langchain_core.runnables.utils import ConfigurableField
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import (
    StrOutputParser, 
    AIMessage, 
    HumanMessage, 
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
# from ZhiAINew import ChatZhipuAI
from langchain_community.chat_models import ChatZhipuAI

from langchain_community.chat_models import QianfanChatEndpoint
import os


import os
from dotenv import find_dotenv, load_dotenv

_=load_dotenv(find_dotenv())
# from  ..configInfo import Config

defaultModel=os.getenv('MODEL_TYPE')
if defaultModel is None:
    defaultModel="gpt"
    
defaultModelName=os.getenv('MODEL')
if defaultModelName is None:
    defaultModelName="gpt-3.5-turbo"
    
temperature = 0
# 模型1
ernie_model = QianfanChatEndpoint(
    qianfan_ak=os.getenv('ERNIE_CLIENT_ID'),
    qianfan_sk=os.getenv('ERNIE_CLIENT_SECRET')
)

# 模型2
gpt_model = ChatOpenAI(model=defaultModelName,temperature=temperature)

# gpt_model.config = Config()
# gpt_model.config.model = defaultModel



# 通过 configurable_alternatives 按指定字段选择模型
model = gpt_model.configurable_alternatives(
    ConfigurableField(id="llm"), 
    default_key=defaultModel, 
    ernie=ernie_model,
    zhipuai=ChatZhipuAI(
        model_name="glm-4"  if defaultModelName is None 
        or defaultModelName=="gpt" 
        or defaultModelName=="gpt-3.5-turbo" else defaultModelName
    )
)
# model.config(model=defaultModel,temperature=0)


# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("{instructions}"),
        MessagesPlaceholder(variable_name="conversation"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)


# prompt.format_prompt(
#     # 对 "conversation" 和 "language" 赋值
#     conversation=[
#         HumanMessage(content="Who is Elon Musk?"), 
#         AIMessage(
#         content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
#     )],
#     query="queryname"
# )
# prompt.input_variables("query").partial(format_instructions=parser.get_format_instructions())


# LCEL
chain = (
    {"query": RunnablePassthrough()} 
    | prompt
    | model 
    | StrOutputParser()
)

def getPrompt(template="",DataTypeModel=None):
    
    
    # template = """提取用户输入中的日期。
    # {format_instructions}
    # 用户输入:
    # {query}"""

    # 根据Pydantic对象的定义，构造一个OutputParser
    if DataTypeModel is None:
        curprompt = PromptTemplate(
            template=template,
            input_variables=["query"],
        )
        #.input_variables("query")
        #.partial(format_instructions=parser.get_format_instructions())
    else:
        parser = PydanticOutputParser(pydantic_object=DataTypeModel)

        curprompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            # 直接从OutputParser中获取输出描述，并对模板的变量预先赋值
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
    return curprompt
    

def getModel(modelName=""):
    """
    获取模型配置链。

    参数:
    - modelName: 字符串，指定模型名称，默认为空字符串。

    返回值:
    - chain: 配置链对象，可用于进一步配置和获取指定模型。
    """
    # 构建模型配置链，初始配置包括GPT模型的可配置替代选项、默认键和ERNIE模型
    chain = gpt_model.configurable_alternatives(
        ConfigurableField(id="llm"), 
        default_key=defaultModel, 
        ernie=ernie_model,
    )
    # 根据传入的modelName参数，配置模型名称
    chain=chain.with_config(configurable={"llm": modelName})
    return chain
def getChain(curModel=model, curPrompt=prompt):
    """
    构建并返回一个处理链。

    参数:
    curModel - 当前使用的模型，默认为model。
    curPrompt - 当前的提示信息，默认为prompt。

    返回值:
    返回一个处理链，该链由多个处理组件构成，具体包括：
    - RunnablePassthrough：负责执行可运行的传递操作。
    - curPrompt：当前的提示信息，用作链中的一部分。
    - curModel：当前使用的模型，作为链中的一部分。
    - StrOutputParser：负责解析输出为字符串的处理组件。
    """
    # 构建并初始化处理链
    chain = (
        {"query": RunnablePassthrough()} 
        | curPrompt
        | curModel 
        | StrOutputParser()
    )

    return chain

def querySingle(query="介绍你自己，包括你的生产商",curChain=chain,modelName=''):
    """
    使用当前链curChain执行特定查询，并返回结果。
    
    参数:
    - query: 字符串，指定要执行的查询，默认为"介绍你自己，包括你的生产商"。
    - curChain: 链对象，指定用于执行查询的链，具体类型依赖于实现，默认为chain。
    
    返回值:
    - ret: 执行查询后的结果，其类型和内容依赖于链对象的具体实现。
    """
    
    # 使用指定的模型 "gpt" 对查询进行处理
    if modelName :
        curChain=curChain.with_config(configurable={"llm": modelName})
   
        
    ret =curChain.invoke(query)

    return ret
    
    
if __name__ == '__main__':
    ret=querySingle()
    print(ret)
    