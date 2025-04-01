import sys 
sys.path.append("../llm")
from openai import OpenAIError
#导入chatopenai，ErnieBotChat,
from langchain_community.chat_models import ChatOpenAI #,ErnieBotChat,ChatZhipuAI,ChatSparkLLM,MoonshotChat, ChatTongyi
from call_llm import parse_llm_api_key
from call_llm import parse_llm_api_base


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
def model_to_llm(type:str=None,model:str=None, temperature:float=0.0, api_key:str=None,api_base:str=None,max_tokens:int =None):


        if type  in LLM_MODEL_DICT and model in LLM_MODEL_DICT[type]:
            if api_key == None:
                api_key = parse_llm_api_key(type, model)
            if api_base == None:
                api_base = parse_llm_api_base(type)
            if type == "SPARK":
                model = SPARK_MODEL_DICT[model]
        #chatopenai可以统一调用聊天模型，除了deepseek之外，其他模型也有各自 的API，为了统一这里全部使用OPENAI
            llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key,
                             openai_api_base=api_base,max_tokens=max_tokens)
        else:
            raise ValueError(f"model{model} not support!!!")

        return llm
