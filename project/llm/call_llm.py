#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   call_llm.py
@Time    :   2023/10/18 10:45:00
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   将各个大模型的原生接口封装在一个接口
'''

from openai import OpenAI
import json
import requests
import _thread as thread
import base64
import datetime
from dotenv import load_dotenv, find_dotenv
import hashlib
import hmac
import os
import queue
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import zhipuai
from langchain.utils import get_from_dict_or_env

import websocket  # 使用websocket_client




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
def get_completion(prompt :str, type :str,model:str, temperature=0.1,api_key:str=None,api_base:str=None, max_tokens=2048):
    #大模型调用可以用：模型自身API,OPENAI SDK,langchain SDK
    #由于OPENAI SDK兼容性最好，优先使用OPENAI SDK，其次再用模型自身API。
    #现考虑支持GPT,文心,Deepseek，Kimi，Qwen，Spark,zhipuai


    # arguments:
    # prompt: 输入提示
    # model：模型名
    # temperature: 温度系数
    # api_key：如名

    # max_tokens : 返回最长序列
    # return: 模型返回，字符串；
    # 调用 GPT

    if api_key == None:
        api_key = parse_llm_api_key(type,model)
    if api_base==None:
        api_base=parse_llm_api_base(type)

    if type=="SPARK":
        model=SPARK_MODEL_DICT[model]

    if type in LLM_MODEL_DICT and model in LLM_MODEL_DICT[type]:
        return get_completion_openai(prompt, model, temperature, api_key, max_tokens,api_base)
    else:
        return "不正确的模型"

def get_completion_openai(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int,api_base:str=None):

    
    client = OpenAI(api_key=api_key, base_url=api_base)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
       #     {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        stream=False
    )


    return  response.choices[0].message.content



def parse_llm_api_key(type:str, model:str=None,env_file:dict()=None):
    """
    通过 model 和 env_file 的来解析平台参数
    model参数只有spark才用到，因为spark不同模型的key不一样。
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ


    if type not in LLM_MODEL_DICT :
        raise ValueError(f"model{type} not support!!!")
    else:
        if type =="SPARK" :
            if model !="default":
                return env_file[model + "_API_KEY"]
            else:#spark嵌入模型 #model=default, 则调用的是嵌入模型
                return (env_file["SPARKEMBEDDING_APP_ID"], env_file["SPARKEMBEDDING_API_SECRET"], env_file["SPARKEMBEDDING_API_KEY"])

        else:
            return env_file[type + "_API_KEY"]




def parse_llm_api_base(type:str, env_file:dict()=None):
    """
    通过 model 和 env_file 的来解析平台参数
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ

    if type not in LLM_MODEL_DICT :
        raise ValueError(f"model{type} not support!!!")
    else:
        return env_file[type+"_BASE_URL"]

