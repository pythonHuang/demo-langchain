from langchain.globals import set_debug
# set_debug(True)
import os
import gradio as gr

from server.configInfo import config
from server.llm.onellm import model
from server.tools.funcall import test as test1
from server.tools.rqgSearch import testRaq as testRaq1
from server.tools.rag import testRAG as testRaq2

import shutil
test1("你是谁？")
# testRaq1("你是谁？")
# testRaq2("你是谁？",True)

# PORT=os.getenv("GRADIO_SERVER_PORT")

# 初始化 OpenAI 客户端
messages=[
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "你是谁？"
    }
]

# client = model.with_config(
#      model="gpt-3.5-turbo",
#      input=messages,
#      stream=True
# )

client = model
# client = model.with_config(configurable={"llm": 'ernie'})
# client = model.with_config(configurable={"llm": 'zhipuai'})

# res=client.invoke(messages)
print(client)

# 生成对话响应
def generate_response(messages):
    response = client.astream(messages)
    return response

# 向 OpenAI 发起查询


def openai_query(query_messages, chat_histroy):

    # 初始化转换后的数据列表
    transformed_data = []

    # 遍历 history 数据
    for pair in chat_histroy:
        # 对每个子列表中的元素，创建两个字典
        user_dict = {'role': 'user', 'content': pair[0]}
        assistant_dict = {'role': 'assistant', 'content': pair[1]}

        # 将这两个字典添加到转换后的数据列表中
        transformed_data.extend([user_dict, assistant_dict])

    combined_messages = query_messages + transformed_data

    # 方便查看输出的数据
    print("\n ===query_messages===", query_messages)
    print("\n ===chat_histroy===", chat_histroy)
    print("\n ===transformed_initial_data===", transformed_data)
    print("\n ===combined_messages===", combined_messages)

    # 向 OpenAI 发起查询
    # openai_res = generate_response(combined_messages)
    return combined_messages

# 生成文本回复


def generate_text(prompt, chat_histroy):
    msg = [
        {
            "role": "system",
            "content": "你是一个陪聊机器人，积极乐观地回应用户，耐心地一步步帮用户分析问题，让对话持续下去，并且经常询问用户的看法，像一个高情商聊天大师一样，不要冷场。如果知道用户的名字，就要经常亲切地称呼他。如果不知道名字，就称呼\"亲爱的\""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    results = openai_query(msg, chat_histroy)
    return results


async def stream_echo(message, chat_histroy,userRag=False,file_obj=None):
    # combined_messages = generate_text(message, chat_histroy)
    # res=client.invoke(combined_messages)
    if file_obj is not None:
        baseName=os.path.basename(file_obj.name)
        print(f"上传文件：{baseName}")
        fileName="./data/"+baseName
        shutil.copyfile(file_obj.name,fileName)
        res=testRaq2(message,userRag,fileName)
    else:
        res=testRaq2(message,userRag)
        
        
    print(res)
    if isinstance(res,str):
        yield res
    elif hasattr(res,"content"):
        print(res.content)
        yield res.content
    else:
        response = []
        for chunk in res:
            if chunk is not None:
                if isinstance(chunk,str):
                    response.append(chunk or "") 
                else:
                    response.append(chunk.content or "") 
                yield "".join(response).strip()
            else:
                break
            
        print("".join(response).strip())
    # response = []
    # for chunk in client.stream(combined_messages):
    #     if chunk is not None:
    #         response.append(chunk.content or "") 
    #         yield "".join(response).strip()
    #     else:
    #         break
    
    # print("".join(response).strip())

    # response = []
    # for chunk in client.stream(message):
    #     if chunk is not None:
    #         response.append(chunk.content or "") 
    #         # yield "".join(response).strip()
    #     else:
    #         break
    
    # print("".join(response).strip())
    # return "".join(response).strip()
    
    # accumulated_response = ''
    # for chunk in res:
    #     # 检查 chunk 是否为 None
    #     if chunk is None or chunk.choices is None or chunk.choices[0].delta is None or chunk.choices[0].delta.content is None:
    #         break

    #     # 获取当前chunk的内容
    #     current_content = chunk.choices[0].delta.content

    #     if current_content is not None:
    #         accumulated_response += current_content
    #         yield accumulated_response

useRag=gr.components.Checkbox(label="是否使用RAG")
file=gr.components.File(label="上传文档")
outputs=gr.components.File(label="下载文档")
# 创建 Gradio 聊天界面
demo = gr.ChatInterface(stream_echo,
                        additional_inputs=[useRag ,file],
                        additional_inputs_accordion_name="Additional Inputs",
                        # outputs=outputs,
                        title="陪聊机器人",
                        description="你有什么想聊的事情吗？不妨先告诉我，你叫什么名字？",
                        # examples=[["今天发工资了，真高兴！快叫我小财神",None],
                        #           ["晋升答辩失败了，我有些难过",None],
                        #           ["又和老婆吵架了，真不知道何时是个头",None],
                        #           ["累死了！累死了！累死了！",None]],
                        
                        # examples=["今天发工资了，真高兴！快叫我小财神",
                        #           "晋升答辩失败了，我有些难过",
                        #           "又和老婆吵架了，真不知道何时是个头",
                        #           "累死了！累死了！累死了！"],
                        ).queue()

if __name__ == "__main__":
    demo.launch(server_port=7801,inbrowser=True)
    # demo.launch(server_port=7801,root_path="/chat/")
