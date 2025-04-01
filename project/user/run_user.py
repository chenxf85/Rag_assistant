
import sys
import os
import gradio as gr
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from user.log_in import login, logout, guest_login,loginToRegister
from user.sign_up import register


# 创建 应用登录和注册页面，可以选择游客访问和用户访问（支持保存此前创建的数据库）
#gr.State是gradio中定义的一个可以在交互中持久化的变量（组件），可以用作输入输出组件，可以用变量本身的方式去访问。用于记录当前界面的状态，登录的用户，以及是否登录
#gr.update()是动态更新组件属性的方法。
#没有直接写整个run_user函数而是分了登录和注册block，是因为app界面如果要增加返回登录的按钮，需要以登录block作为输出组件（可见），隐藏app_block。
#而如果写成整体，run_user需要使用app_block，而app_block的布局由于返回登录按钮，所以需要以登录block作为输出组件，这样会出现问题。
#所以需要把登录block和登陆界面按钮进入app_block的逻辑分开写，这样app_block的布局就不会受到影响。
#代码复杂了一些，但是保证了模块的分离型。



class MyBlocks():

   # 自定义类，是为了可以以字典的形式访问布局内部组件（这些组件在不同block之间需要传递）
    def __init__(self, block,*args,**kwargs):
        self.block=block

        self._component_registry = {}

        for (name, component) in args:

            self.register(name, component)
            # 也支持关键字形式
        for name, component in kwargs.items():
            self.register(name, component)
    def register(self, name, component):
        self._component_registry[name] = component
        return component

    def __getitem__(self, name):
        return self._component_registry[name]
def login_block() ->MyBlocks:  #在block之间传递信息

        # 初始化状态，记录登录状态和当前用户名;



        # 登录界面（默认可见）
        with gr.Group(visible=True) as login_block:
            gr.Markdown("<h1><center>Rag_Assistant 用户登录系统<center></h1>")
            with gr.Column(scale=1):
                login_username = gr.Textbox(label="用户名")
                login_password = gr.Textbox(label="密码", type="password")
                login_button1 = gr.Button("登录")
                reg_button1= gr.Button("前去注册")
                guest_button =gr.Button("游客访问")
                login_output = gr.Textbox(label="提示")

            gr.Markdown("""
                提醒：<br>
               1. 请先注册用户，再登录。 <br>
               2. 游客访问不需要注册，但无法保存知识库。 <br>
               3. 登录后访问，可以管理此前的知识库 <br>
               """)
            return MyBlocks(login_block,
                            ("login_username",login_username),
                            ("login_password",login_password),
                            ("login_button1",login_button1),
                            ("reg_button1",reg_button1),
                            ("guest_button",guest_button),
                            ("login_output",login_output))

def register_block()->MyBlocks:
        # 注册界面（默认不可见）

        with gr.Group(visible=False) as register_block:   #Group 是 Blocks 中的一个布局元素，它将子项分组在一起，以便它们之间没有任何填充或边距。
            with gr.Column(scale=1):
                gr.Markdown("<h1><center>Rag_Assistant 用户注册系统<center></h1>")
                reg_username = gr.Textbox(label="用户名")
                reg_password1 = gr.Textbox(label="密码", type="password")
                reg_password2 = gr.Textbox(label="确认密码", type="password")
                reg_button2 = gr.Button("注册")
                log_button2= gr.Button("返回登录")
                reg_output = gr.Textbox(label="提示")

        return MyBlocks(register_block,
                        ("reg_username",reg_username),
                        ("reg_password1",reg_password1),
                        ("reg_password2",reg_password2),
                        ("reg_button2",reg_button2),
                        ("log_button2",log_button2),
                        ("reg_output",reg_output))
            # 登录按钮绑定登录函数，登录后根据返回值控制界面显示

def demo_app():
    with gr.Group(visible=False) as app_block:
        welcome_text = gr.Markdown("欢迎使用应用！")
        app_info = gr.Textbox(label="这里是你的应用内容")  # 可以放任何应用内容
        logout_button = gr.Button("退出登录")
        # 退出登录按钮绑定逻辑，退出后返回到登录和注册界面

        # 登录按钮绑定登录函数，登录后根据返回值控制界面显示

    return MyBlocks(app_block,("logout_button",logout_button))


def run_user(login_block:MyBlocks, register_block:MyBlocks,app_block:MyBlocks): #demo

        state = gr.State({"username": "", "logged_in": False})

        log_button2 = register_block["log_button2"]
        login_button1 = login_block["login_button1"]
        reg_button1 = login_block["reg_button1"]
        guest_button = login_block["guest_button"]
        reg_button2 = register_block["reg_button2"]
        logout_button = app_block["logout_button"]



        login_button1.click(
            login,
            inputs=[login_block["login_username"],login_block["login_password"] , state],
            outputs=[login_block["login_output"], state, login_block.block, app_block.block]
        )

        log_button2.click(loginToRegister, inputs=[],
                          outputs=[register_block.block, login_block.block, login_block["login_output"]])  # 返回登录界面

        guest_button.click(guest_login, inputs=[state],
                           outputs=[login_block["login_output"]
                               , state, login_block.block, app_block.block]
                           )

        reg_button1.click(loginToRegister, inputs=[],
                          outputs=[login_block.block,register_block.block, login_block["login_output"]])   #登录跳转到注册界面
        reg_button2.click(register, inputs=[register_block["reg_username"],register_block["reg_password1"],register_block["reg_password2"] ],
                          outputs=[register_block["reg_output"], register_block.block, login_block.block, login_block["login_output"]])  #注册成功后跳转到登录界面
        logout_button.click(
            logout,
            inputs=[state],
            outputs=[state, app_block.block, login_block.block,login_block["login_output"]]
        )

def run_rag_assistant(state,login_block, register_block, app_block):

  #  state=gr.State({"username":{},"logged_in":False})
    log_button2 = register_block["log_button2"]
    login_button1 = login_block["login_button1"]
    reg_button1 = login_block["reg_button1"]
    guest_button = login_block["guest_button"]
    reg_button2 = register_block["reg_button2"]
    logout_button = app_block["logout_button"]

    login_button1.click(
        login,
        inputs=[login_block["login_username"], login_block["login_password"], state],
        outputs=[login_block["login_output"], state, login_block.block, app_block.block,
                 app_block["llm_key"],app_block["embedding_key"],app_block["title"]]
    )

    log_button2.click(loginToRegister, inputs=[],
                      outputs=[register_block.block, login_block.block, login_block["login_output"]])  # 返回登录界面

    guest_button.click(guest_login, inputs=[state],
                       outputs=[login_block["login_output"]
                           , state, login_block.block, app_block.block,app_block["title"]])

    reg_button1.click(loginToRegister, inputs=[],
                      outputs=[login_block.block, register_block.block, login_block["login_output"]])  # 登录跳转到注册界面
    reg_button2.click(register, inputs=[register_block["reg_username"], register_block["reg_password1"],
                                        register_block["reg_password2"]],
                      outputs=[register_block["reg_output"], register_block.block, login_block.block,
                               login_block["login_output"]])  # 注册成功后跳转到登录界面
    logout_button.click(
        logout,
        inputs=[state],
        outputs=[state, app_block.block, login_block.block, login_block["login_output"]]
    )

#这是结合登录，注册和app_block使用的一个demo
if __name__ == "__main__":

    with gr.Blocks() as demo:

        login_block = login_block()
        register_block = register_block()
        app_block = demo_app()
        run_user(login_block, register_block,app_block)

    demo.launch()

