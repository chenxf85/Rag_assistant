# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import KnowledgeDB
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
from user.run_user import login_block,register_block,run_rag_assistant,MyBlocks
import traceback
import logging
from user.log_in import logout
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量

SPARK_MODEL_DICT={"SPARKLITE":"generalv1.1",
                  "SPARKMAX":"generalv3.1",
                  "SPARKPRO":"generalv3.5",
                  "SPARKULTRA":"generalv4.0"}




LLM_MODEL_DICT = {
    "OPENAI": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4o-mini"],
    "WENXIN": ["ernie-4.0-8k-latest","ernie-4.0-turbo-8k-latest","ernie-4.0-turbo-128k","ernie-lite-pro-128k","ernie-speed-pro-128k"],
    "SPARK": ["SPARKLITE","SPARKMAX", "SPARKPRO","SPARKULTRA"],
    "ZHIPUAI": ["glm-4-flash","glm-4-air","glm-4-long","glm-4-plus","glm-zero-preview"],
    "KIMI":["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"],
    "DEEPSEEK":["deepseek-chat","deepseek-reasoner"],
    "QWEN":["qwen-max","qwen-plus","qwen-turbo","qwen-long"]
}

_ = load_dotenv(find_dotenv())


INIT_LLM = "gpt-3.5-turbo"
EMBEDDING_MODEL_DICT = {'ZHIPUAI':["embedding-2","embedding-3"],
'OPENAI':["default"], #GPT-FREE没有指明调用的模型
'SPARK':["default"], #SPARK没有指明调用的模型
 'WENXIN':[
        "tao-8k",
        "embedding-v1",
        "bge-large-zh",
        "bge-large-en"],
"QWEN":["text-embedding-v3","text-embedding-v2","text-embedding-v1"]
}
INIT_EMBEDDING_MODEL = "openai"
DEFAULT_DB_PATH = "../../data_base/knowledge_db"
DEFAULT_PERSIST_PATH = "../../data_base/vector_db/chroma"
AIGC_AVATAR_PATH = "aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "datawhale_avatar.png"
AIGC_LOGO_PATH = "../../figures/RAG_Assistant.png"
image1_path = "../../figures/image1.png"
#DATAWHALE_LOGO_PATH = "../../figures/datawhale_logo.png"

knowledgeDB = KnowledgeDB()


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "") # 返回 LLM_MODEL_DICT 中 platform 对应的模型列表，如果没有则返回空列表


class Model_center():
    """
    存储问答 Chain 的对象 

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model_type: str = "OPENAI", model:str="gpt-3.5-turbo",
                                  embedding_type:str="OPENAI",embedding: str = None, temperature: float = 0.0, top_k: int = 4, history_len: int = 3,
                                  file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH,
                                  api_key:str=None,embedding_key:str=None,spark_app_id:str=None,spark_api_secret:str=None,
                                  api_base:str=None,embedding_base:str=None):

        """
        调用带历史记录的问答链进行回答;
        """

        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = \
                    Chat_QA_chain_self(model_type=model_type,model=model, temperature=temperature,
                                    top_k=top_k, chat_history=chat_history, file_path=file_path, persist_path=persist_path,
                                    embedding_type=embedding_type,embedding=embedding,api_key=api_key,
                                    spark_app_id=spark_app_id,spark_api_secret=spark_api_secret,
                                    embedding_key=embedding_key,

                                       api_base=api_base,embedding_base=embedding_base)

            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history


    def qa_chain_self_answer(self, question: str, chat_history: list = [], model_type="OPENAI", model: str = "gpt-3.5-turbo",
                             embedding_type="OPENAI",embedding=None, temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH,
                             persist_path: str = DEFAULT_PERSIST_PATH,api_key:str=None,embedding_key:str=None,spark_app_id:str=None,spark_api_secret:str=None,
                             embedding_base:str=None,api_base:str=None
                             ):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model_type=model_type,model=model, temperature=temperature,
                                                                       top_k=top_k, file_path=file_path, persist_path=persist_path,
                                                                       embedding_type=embedding_type, embedding=embedding,api_key=api_key,
                                                                       spark_app_id=spark_app_id,spark_api_secret=spark_api_secret,
                                                                       embedding_key=embedding_key,embedding_base=embedding_base,
                                                                       api_base=api_base
                                                                      )

            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append(
                (question, chain.answer(question, temperature, top_k)))
            return "", chat_history
        except Exception as e:
            return e, chat_history


    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt



def respond(message, chat_history,type, llm, history_len=3, temperature=0.1, max_tokens=2048,llm_key=None):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复,
    chat_history: 更新后的聊天历史记录;
    """

    if message == None or len(message) < 1:
            return "", chat_history

    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt,type, llm, temperature=temperature, max_tokens=max_tokens,api_key=llm_key)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history

    except Exception as e:
        error_info = traceback.format_exc() #将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt", format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',level= logging.ERROR) #设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"chat_process发生错误：\n{error_info}")  #错误输出到控制台，存在于项目根目录
        print(error_info)
        return e, chat_history

# 定义一个函数，用于更新 LLM 类型下拉框的内容。
def update_llm_dropdown(selected_type):
    models = LLM_MODEL_DICT.get(selected_type, [])
    return gr.update(choices=models, value=models[0] if models else None) #将LLM下拉选项更新为：

def update_embedding_dropdown(selected_type,state):
    embeddings = EMBEDDING_MODEL_DICT.get(selected_type, [])
    if selected_type=="SPARK" and state["username"]!="guest":
        return gr.update(choices=embeddings, value=embeddings[0] if embeddings else None), gr.update(visible=True),gr.update(visible=True)
    else:
        return gr.update(choices=embeddings, value=embeddings[0] if embeddings else None),gr.update(visible=False),gr.update(visible=False)

model_center = Model_center()


def Rag_assistant_block(state):
    with gr.Group(visible=False) as demo:
        DB=gr.State(knowledgeDB)
        with gr.Row(equal_height=True):
            gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=1, show_label=False, show_download_button=False,
                     container=False)
            gr.Column(scale=4) #为了使图片出现在合适的位置，用column占位
            with gr.Column(scale=2):
                title=gr.Markdown(f"""<h1><center>Rag_Assistant</center></h1>
                    <center>当前用户：</center>    """)

            gr.Column(scale=4)
          #  gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False,
            #         container=False)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True,
                                     avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))
                # 创建一个文本框组件，用于输入 prompt。

                msg = gr.Textbox(label="Prompt/问题")

                with gr.Row():
                    # 创建提交按钮。
                    db_with_his_btn = gr.Button("Chat db with history")
                    db_wo_his_btn = gr.Button("Chat db without history")
                    llm_btn = gr.Button("Chat with llm")

                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear console")
                with gr.Row():
                    # 登出
                    logout_button = gr.Button("退出登录")

            with gr.Column(scale=1):
                file = gr.File(label='请选择知识库目录', file_count='multiple',
                               file_types=['.txt', '.md', '.docx', '.pdf'])
                with gr.Row():
                    init_db = gr.Button("知识库文件向量化")
                model_argument = gr.Accordion("参数配置", open=False)

                with model_argument:
                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="llm temperature",
                                            interactive=True)

                    top_k = gr.Slider(1,
                                      10,
                                      value=3,
                                      step=1,
                                      label="vector db search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history length",
                                            interactive=True)

                chat_model_select = gr.Accordion("聊天模型选择")  #
                embedding_model_select = gr.Accordion("Embedding模型选择")
                with chat_model_select:
                    model_type = gr.Dropdown(
                        list(LLM_MODEL_DICT.keys()),
                        label="companys",
                        value="OPENAI",
                        interactive=True)

                    llm = gr.Dropdown(LLM_MODEL_DICT["OPENAI"],
                                      label="large language model",
                                      value=INIT_LLM,
                                      interactive=True)

                    llm_key= gr.Textbox(label="llm key", value=None,type="password",visible=False) #只对用户可见



                with embedding_model_select:
                    embedding_type = gr.Dropdown(EMBEDDING_MODEL_DICT,
                                                 label="companys",
                                                 value="OPENAI")
                    embedding = gr.Dropdown(EMBEDDING_MODEL_DICT["OPENAI"],
                                            label="Embedding model",
                                            value=None,
                                            interactive=True)

                    embedding_key= gr.Textbox(label="embedding key", value=None,type="password",visible=False)
                    #只对spark embedding可见
                    spark_app_id = gr.Textbox(label="spark app id", value=None,type="password",visible=False)
                    spark_api_secret = gr.Textbox(label="spark api secret", value=None,type="password",visible=False)



            model_type.change(update_llm_dropdown, inputs=model_type, outputs=llm)  # 更新LLM下拉选项
            embedding_type.change(update_embedding_dropdown, inputs=[embedding_type,state], outputs=[embedding,spark_app_id,spark_api_secret])

            # 设置初始化向量数据库按钮的点击事件。当点击时，调用 create_db_info 函数，并传入用户的文件和希望使用的 Embedding 模型。
            init_db.click(knowledgeDB.create_db_info,
                          inputs=[embedding_type, embedding], outputs=[msg])

            # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                msg, chatbot, model_type, llm, embedding_type, embedding, temperature, top_k, history_len, llm_key,embedding_key,spark_app_id,spark_api_secret],
                                  outputs=[msg, chatbot])

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                msg, chatbot, model_type, llm, embedding_type, embedding, temperature, top_k,
                llm_key,embedding_key,spark_app_id,spark_api_secret], outputs=[msg, chatbot])
            # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            llm_btn.click(respond, inputs=[
                msg, chatbot, model_type, llm, history_len, temperature,llm_key], outputs=[msg, chatbot],
                          show_progress="minimal")

            # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
            msg.submit(respond, inputs=[
                msg, chatbot, model_type, llm, history_len, temperature], outputs=[msg, chatbot],
                       show_progress="hidden")
            # 点击后清空后端存储的聊天记录
            clear.click(model_center.clear_history)


        with gr.Row():
            gr.Image(value=image1_path, scale=1, min_width=5, show_label=False, show_download_button=False,
                         container=False)
            gr.Column(scale=4)

        gr.Markdown("""提醒：<br>
                         1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
                         2. 初始化数据库时间可能较长，请耐心等待。
                         3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
                         """)
            #gr.Column(scale=1) #占位
    # threads to consume the request

    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。

    return MyBlocks(demo,("logout_button",logout_button),
    ("llm_key",llm_key),
                    ("embedding_key",embedding_key),
                    ("spark_app_id",spark_app_id),
                    ("spark_api_secret",spark_api_secret),
                    ("title",title)
                    )


if __name__ == "__main__":
    with gr.Blocks(title="Rag_assistant") as Rag_assistant:

        state=gr.State({"username":{},"logged_in":False})
        login_block=login_block()
        register_block=register_block()
        app_block=Rag_assistant_block(state)
        run_rag_assistant(state,login_block,register_block,app_block)
    Rag_assistant.launch(share=True)


