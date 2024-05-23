from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import JsonOutputToolsParser


from typing import Union
from operator import itemgetter
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import tool

from .util import multiply, add, exponentiate

# 名称到函数的映射
toolsDefault = [multiply, add, exponentiate]


import json

def route(response):
    if len(response["functions"]) > 0:
        return response["functions"]
    else:
        return response["text"]
    
def funcall_getTool(tools=toolsDefault):
    
    # 名称到函数的映射
    tool_map = {tool.name: tool for tool in tools}

    def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
        """根据模型选择的 tool 动态创建 LCEL"""
        tool = tool_map[tool_invocation["type"]]
        return RunnablePassthrough.assign(
            output=itemgetter("args") | tool
        )

    # .map() 使 function 逐一作用与一组输入
    call_tool_list = RunnableLambda(call_tool).map()
    return call_tool_list


def funcallGetResult(model,tools=toolsDefault):
    # 带有分支的 LCEL
    llm_with_tools = model.bind_tools(tools) | {
        "functions": JsonOutputToolsParser(),
        "text": StrOutputParser()
    }
    return llm_with_tools


def funcallByTool(model,query,tools=toolsDefault):
    call_tool_list=funcall_getTool(tools)
    # llm_with_tools = model.bind_tools(tools) | {
    #     "functions": JsonOutputToolsParser() | call_tool_list,
    #     "text": StrOutputParser()
    # } | RunnableLambda(route)
    
    llm_with_tools = model.bind_tools(tools) 
    llm_with_tools =llm_with_tools | {
        "functions": JsonOutputToolsParser() | call_tool_list,
        "text": StrOutputParser()
    } 
    llm_with_tools =llm_with_tools | RunnableLambda(route)
     

    #result = llm_with_tools.invoke("1024的平方是多少")
    #print(result)

    #result = llm_with_tools.invoke("你好")
    #print(result)
    return llm_with_tools.invoke(query)

def test():
    from ..llm.onellm import model
    result=funcallByTool(model=model,query="你好")
    print(result)
    
if __name__ == "__main__":
    test()