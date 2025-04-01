
#用户登录后可以获取自己的数据库，并且选择上传自己的key，选择模型，也可以选择默认使用本地提供的key；

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 登录逻辑处理
import gradio as gr
from user.sign_up import load_users, save_user
import traceback
import logging

def login(username, password, state):
    try :
        users = load_users() #json文件中读取用户信息
        # 检查用户名和密码是否匹配
        if username in users and users[username] == password:
            # 登录成功，修改状态
            state["logged_in"] = True
            state["username"] = username
            # 返回提示信息、更新状态、隐藏登录界面、显示主应用界面，显示embedding_key和llm_key
            return "登录成功！", state, gr.update(visible=False), gr.update(visible=True),\
            gr.update(visible=True),gr.update(visible=True),gr.update(value=f"""<h1><center>Rag_Assistant</center></h1>
                    <center>当前用户：{state["username"]}</center> 
                    """) #针对rag_assistant增加的，非一般性代码
        else:
            # 登录失败，保持注册和登录界面可见
            return "用户名或密码错误！", state, gr.update(visible=True), gr.update(visible=False),\
                gr.update(visible=True),gr.update(visible=True),gr.update() #针对rag_assistant增加的，非一般性代码
    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"用户登录发生错误：\n{error_info}")  # 错误输出到log.txt, 存在于项目根目录
        print(error_info)
        return f"用户登录失败：{str(e)}",gr.update(visible=True), gr.update(visible=False),\
            gr.update(visible=False),gr.update(visible=False),gr.update()


# 退出登录逻辑
def logout(state):
    state["logged_in"] = False
    state["username"] = ""
    # 退出后显示注册和登录界面，隐藏应用


    return state, gr.update(visible=False), gr.update(visible=True),"",gr.update(visible=True),gr.update(visible=True) # 清空登录输入框

def loginToRegister():

    # 退出后显示注册和登录界面，隐藏应用界面
    return  gr.update(visible=False), gr.update(visible=True),""
def guest_login(state):
    try:
        state["logged_in"] = True
        state["username"] = "guest"
        # 登录成功，修改状态,隐藏登陆界面，显示主应用界面
        return "登录成功！", state, gr.update(visible=False), gr.update(visible=True) ,\
              gr.update(value=f"""<h1><center>Rag_Assistant</center></h1>
                            <center>当前用户：游客</center> 
                            """)

    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"游客登录发生错误：\n{error_info}")  # 错误输出到log.txt, 存在于项目根目录
        print(error_info)
        return f"游客登录失败：{str(e)}", gr.update(visible=True), gr.update(visible=False),\
            gr.update()

