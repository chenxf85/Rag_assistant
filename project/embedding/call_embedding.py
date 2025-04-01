import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_community.embeddings import SparkLLMTextEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import parse_llm_api_key
from llm.call_llm import parse_llm_api_base
from langchain.utils import get_from_dict_or_env
EMBEDDING_MODEL_LIST = {'ZHIPUAI':["embedding-2","embedding-3"],
'OPENAI':["default"],
'SPARK':["default"],
 'WENXIN':[
        "tao-8k",
        "embedding-v1",
        "bge-large-zh",
        "bge-large-en"],
"QWEN":["text-embedding-v3","text-embedding-v2","text-embedding-v1"]
}
def get_embedding(type:str=None,embedding: str=None, embedding_key: str=None,embedding_base:str=None,spark_app_id:str=None, spark_api_secret:str=None):

    if type  in EMBEDDING_MODEL_LIST and  (embedding in EMBEDDING_MODEL_LIST[type]):

        if embedding_key == None:
            if type!= "SPARK":
                embedding_key = parse_llm_api_key(type,embedding)
            else:
                spark_app_id,spark_api_secret,embedding_key = parse_llm_api_key(type,embedding)

        if embedding_base== None  :
            embedding_base = parse_llm_api_base(type)

        # 因为SPARK和OPENAI只支持一种免费的嵌入模型调用，所以这里单独写，而不传入模型名称
        if type == "SPARK":
            return SparkLLMTextEmbeddings(spark_app_id=spark_app_id, spark_api_secret=spark_api_secret,
                                          spark_api_key=embedding_key)
        elif type =="OPENAI":
            return OpenAIEmbeddings(openai_api_key=embedding_key,
                                    openai_api_base=embedding_base)
        else:
            return OpenAIEmbeddings(
                model=embedding,
                openai_api_key=embedding_key,
                openai_api_base=embedding_base)

    else:
        raise ValueError(f"embedding {embedding} not support ")




