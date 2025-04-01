import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import tempfile
from embedding.call_embedding import get_embedding
import gradio as gr

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


# 首先实现基本配置




from langchain_chroma import Chroma



DEFAULT_DB_PATH = "../../data_base/knowledge_db"
DEFAULT_PERSIST_PATH = "../database/vector_data_base"


class KnowledgeDB:
    # 将create_db和get_vectordb合并为一个类
    # 数据库类，该类用于管理知识库，包括获取文件目录下的所有文件，加载文件；
    # 进行文本切割，创建持久化知识库。
    # 可视化一个知识库列表
    # 可以选择加载已有知识库，或者重新创建新的知识库。
    # 为不同的嵌入模型维护不同的数据库（不同的embedding，相同的文本也会有不同的输出和可能不同的输出维度）
    # 用户可以在后端持久化数据库，并且可以加载用户自身已有数据库，所以无需加载数据库到内存（变量），游客无法在后端保存数据库，只能使用当前页面缓存的数据库。

    def __init__(self,state):

        self.vectordbs = {}        #维护不同embedding的数据库,
        self.state=state
        self.DB_PATH= DEFAULT_DB_PATH                  #在app交互界面由用户选择本地文件路径
        self.persist_directory = DEFAULT_PERSIST_PATH  #+"/"+user #如果是用户访问为每个用在后端数据库目录：包括不同embedding的数据库

    def get_files(self,dir_path):
        file_list = []
        for filepath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
        return file_list


    def file_loader(self,file, loaders):
        if isinstance(file, tempfile._TemporaryFileWrapper):
            file = file.name
        if not os.path.isfile(file):
            [self.file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
            return
        file_type = file.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file))
        elif file_type == 'txt':
            loaders.append(UnstructuredFileLoader(file))
        return


    def create_db_info(self, files=DEFAULT_DB_PATH,embedding_type="OPENAI",embeddings="default",
                       embedding_key:str=None,embedding_base:str=None,spark_app_id:str=None,spark_api_secret:str=None):
        if embedding_type not in self.vectordbs:
            vector_db=self.create_db(files,embedding_type,embeddings,
                           embedding_key,embedding_base,spark_app_id,spark_api_secret)
            vector_db.persist()  #需要优化游客不持久化数据库
            self.vectordbs[embedding_type] = vector_db

        else:
            self.vectordbs[embedding_type].add_documents(files)

        return ""

    def create_db(self,  files=DEFAULT_DB_PATH,embedding_type:str="OPENAI", embeddings:str=None,
                  embedding_key:str=None,embedding_base:str=None,spark_app_id:str=None,spark_api_secret:str=None):
        """
        该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。

        参数:
        file: 存放文件的路径。
        embeddings: 用于生产 Embedding 的模型

        返回:
        vectordb: 创建的数据库。

        """

        persist_directory=self.persist_directory

        if files == None:
            return "can't load empty file"

        if type(files) != list:
            files = [files]
        loaders = []
        [self.file_loader(file, loaders) for file in files]
        docs = []
        for loader in loaders:
            if loader is not None:
                docs.extend(loader.load())
        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs[:10])

        # 定义持久化路径
        # persist_directory = '../../data_base/vector_db/chroma'
        if type(embeddings) == str:
            embeddings = get_embedding(embedding_type,embedding=embeddings, embedding_key=embedding_key,
                                       embedding_base=embedding_base,spark_app_id=spark_app_id, spark_api_secret=spark_api_secret)

        # 加载数据库
        #不同的embedding得到的文本向量维度和数据不同，应该维护不同的数据库；

        for i in range(len(embeddings)):
            self.vectordb[embeddings[i]] = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings[i],
                persist_directory=persist_directory+f"/{embedding_type}"   #允许我们将persist_directory目录保存到磁盘上
            )
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_directory    #允许我们将persist_directory目录保存到磁盘上
        )

        vectordb.persist()
        return vectordb

    def del_db(self):
        pass


    def presit_knowledge_db(self,vectordb):
        """
        该函数用于持久化向量数据库。

        参数:
        vectordb: 要持久化的向量数据库。
        """
        vectordb.persist()


    def load_knowledge_db(path, embeddings):
        """
        该函数用于加载向量数据库。

        参数:
        path: 要加载的向量数据库路径。
        embeddings: 向量数据库使用的 embedding 模型。

        返回:
        vectordb: 加载的数据库。
        """
        vectordb = Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
        return vectordb

    def get_vectordb(self,file_path: str = None, persist_path: str = None, type="OPENAI", embedding="dafault",
                     embedding_key: str = None, embedding_base: str = None,
                     spark_app_id: str = None, spark_api_secret: str = None):
        """
        返回向量数据库对象
        输入参数：
        question：
        llm:
        vectordb:向量数据库(必要参数),一个对象
        template：提示模版（可选参数）可以自己设计一个提示模版，也有默认使用的
        embedding：可以使用zhipuai等embedding，不输入该参数则默认使用 openai embedding，注意此时api_key不要输错
        """
        embedding = get_embedding(type=type, embedding=embedding, embedding_key=embedding_key,
                                  embedding_base=embedding_base, spark_app_id=spark_app_id,
                                  spark_api_secret=spark_api_secret)
        if os.path.exists(persist_path):  # 持久化目录存在
            contents = os.listdir(persist_path)
            if len(contents) == 0:  # 但是下面为空
                # print("目录为空")
                vectordb = self.create_db(file_path, persist_path, embedding)
                # presit_knowledge_db(vectordb)
                vectordb = self.load_knowledge_db(persist_path, embedding)
            else:
                # print("目录不为空")
                vectordb = self.load_knowledge_db(persist_path, embedding)
        else:  # 目录不存在，从头开始创建向量数据库
            vectordb = self.create_db(file_path, persist_path, embedding)
            # presit_knowledge_db(vectordb)
            vectordb = self.load_knowledge_db(persist_path, embedding)

        return vectordb


