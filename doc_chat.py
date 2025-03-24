# pip install streamlit==1.39.0
# pip install toml
# pip install requests
# pip install langchain
# pip install langchain_community
# pip install langchain_chroma
# pip install openai
import streamlit as st
import tempfile
import os
import requests
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain.tools.retriever import create_retriever_tool
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
import psutil  # 需要安装: pip install psutil
import traceback  # 用于详细错误追踪
from typing import List, Optional
from langchain_core.documents import Document

# 添加日志记录功能
def log_error(title, error):
    """记录错误信息到UI和日志"""
    error_msg = f"{title}: {str(error)}"
    st.error(error_msg)
    print(f"ERROR: {error_msg}")
    
    # 记录详细堆栈跟踪
    trace = traceback.format_exc()
    print(f"TRACE: {trace}")
    
    # 在开发环境中，也可以显示堆栈跟踪
    with st.expander("错误详情"):
        st.code(trace)

# 文档问答应用程序
# 主要功能：支持多文档上传、语义检索、基于LLM的问答

# 导入必要的库
# 提供Web应用交互、文件处理、网络请求、AI模型交互等功能

# 配置Streamlit应用页面
# 设置页面标题和布局，提供良好的用户体验
st.set_page_config(page_title="文档问答", layout="wide")
st.title("文档问答")  # 显示应用标题

# 自定义嵌入类：硅基流动嵌入
# 封装文本嵌入API调用，将文本转换为向量表示
class SiliconFlowEmbeddings(Embeddings):
    """硅基流动文本嵌入模型封装类"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "BAAI/bge-m3"
        self.url = "https://api.siliconflow.cn/v1/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档嵌入"""
        embeddings = []
        for text in texts:
            result = self._embed_text(text)
            if result:
                embeddings.append(result)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """单个查询文本嵌入"""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> Optional[List[float]]:
        """调用API生成文本嵌入向量"""
        payload = {
            "input": text,
            "encoding_format": "float",
            "model": self.model
        }
        headers = {
            "Authorization": self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request("POST", self.url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            st.error(f"API请求失败: {str(e)}")
            if hasattr(e.response, 'text'):
                st.error(f"错误详情: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"处理响应时出错: {str(e)}")
            return None

    def __call__(self, input):
        """适配 Chroma 接口"""
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise ValueError("输入必须是字符串或字符串列表")

# 聊天历史管理
# 初始化和管理聊天会话状态
def init_session():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None

# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()

# 自定义DeepSeek语言模型
# 封装DeepSeek API调用，提供对话能力
class DeepSeekLLM:
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key = api_key
        self.model = model
        # 使用OpenAI SDK调用DeepSeek API
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def __call__(self, prompt):
        """调用DeepSeek API生成回复"""
        try:
            # 为文档问答任务提供明确的系统指令
            system_message = """你是一个专业的文档问答助手。
根据提供的文档内容准确回答用户问题。
如果文档中包含答案，请基于文档内容进行回答，不要添加额外信息。
如果文档中没有包含问题的答案，请回答"抱歉，这个问题我无法在文档中找到相关信息。"
不要编造不在文档中的信息。
保持回答简洁、准确、有帮助。"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度以提高回答的确定性
                max_tokens=1000,   # 控制回答的长度
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"DeepSeek API调用失败: {str(e)}")
            return "抱歉，我无法获取回复。"

# 初始化DeepSeek语言模型
deepseek_api_key = "sk-ff6c3fa4fc4e453b92b7d023cd9efc4e"
llm = DeepSeekLLM(api_key=deepseek_api_key)

# 处理文档的函数
def process_documents(input_files: List[str], output_dir: str, save_to_local=False, local_path=None) -> Optional[Chroma]:
    """处理文档并创建向量数据库"""
    # 确定最终的输出目录
    final_output_dir = local_path if save_to_local and local_path else output_dir
    
    # 初始化硅基流动嵌入模型
    embeddings = SiliconFlowEmbeddings(api_key="sk-nhyeljqothggnyzntjdzecdhivhvstzyqubhtafplbrxcjhi")
    
    # 加载文档
    loaded_docs = []
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                loaded_docs.append(Document(page_content=text, metadata={"source": file_path}))
            st.info(f"成功加载文件: {file_path}")
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                    loaded_docs.append(Document(page_content=text, metadata={"source": file_path}))
                st.info(f"成功加载文件(GBK编码): {file_path}")
            except Exception as e:
                st.error(f"无法加载文件 {file_path}: {str(e)}")
                continue
    
    if not loaded_docs:
        st.error("没有成功加载任何文档")
        return None
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(loaded_docs)
    st.info(f"文档已分割为 {len(splits)} 个文本块")
    
    try:
        # 创建向量数据库
        from langchain_chroma import Chroma
        
        # 确保输出目录存在
        os.makedirs(final_output_dir, exist_ok=True)
        
        # 创建向量数据库
        db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=final_output_dir
        )
        
        st.success(f"向量数据库创建成功，保存在: {final_output_dir}")
        return db
        
    except Exception as e:
        st.error(f"创建向量数据库失败: {str(e)}")
        return None

def load_vectordb(db_path: str) -> Optional[Chroma]:
    """加载本地向量数据库"""
    st.info(f"尝试加载向量数据库: {db_path}")
    
    if not os.path.exists(db_path):
        st.error(f"向量数据库路径不存在: {db_path}")
        return None
    
    try:
        # 初始化嵌入模型
        embeddings = SiliconFlowEmbeddings(api_key="sk-nhyeljqothggnyzntjdzecdhivhvstzyqubhtafplbrxcjhi")
        
        # 加载向量数据库
        from langchain_chroma import Chroma
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        st.success("向量数据库加载成功")
        return db
        
    except Exception as e:
        st.error(f"加载向量数据库失败: {str(e)}")
        return None

def check_directory_writable(directory):
    """检查目录是否可写"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception:
            return False
    else:
        # 检查是否有写权限
        try:
            test_file = os.path.join(directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False

def load_multiple_vectordbs(vectordb_dirs):
    """
    加载多个向量数据库并合并为一个查询接口
    
    参数:
    - vectordb_dirs: 向量数据库目录列表
    
    返回:
    - combined_retriever: 合并后的检索器，失败时返回None
    """
    if not vectordb_dirs:
        st.warning("未选择任何向量库")
        return None
        
    # 记录成功加载的向量库
    loaded_vectordbs = []
    
    # 加载每个向量库
    for db_dir in vectordb_dirs:
        try:
            st.info(f"正在加载向量库: {db_dir}")
            vectordb = load_vectordb(db_dir)
            if vectordb:
                loaded_vectordbs.append(vectordb)
                st.success(f"成功加载向量库: {db_dir}")
            else:
                st.error(f"加载向量库失败: {db_dir}")
        except Exception as e:
            st.error(f"加载向量库 {db_dir} 时出错: {str(e)}")
    
    if not loaded_vectordbs:
        st.error("没有成功加载任何向量库")
        return None
    
    if len(loaded_vectordbs) == 1:
        # 只有一个向量库，直接返回
        return loaded_vectordbs[0]
    else:
        # 有多个向量库，创建多重检索器
        try:
            from langchain.retrievers import MergerRetriever
            
            # 创建检索器列表
            retrievers = []
            for i, vectordb in enumerate(loaded_vectordbs):
                retrievers.append(vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                ))
            
            # 合并检索器
            merged_retriever = MergerRetriever(retrievers=retrievers)
            st.success(f"成功合并 {len(retrievers)} 个向量库为统一检索接口")
            return merged_retriever
        except Exception as e:
            st.error(f"合并检索器时出错: {str(e)}")
            # 如果合并失败，返回第一个向量库
            st.warning("合并失败，将使用第一个成功加载的向量库")
            return loaded_vectordbs[0]

# 查找本地向量库目录
def find_vectordb_dirs(base_dir="/Users/yeziyin"):
    """
    在基础目录下查找可能的向量数据库目录
    
    参数:
    - base_dir: 基础目录路径
    
    返回:
    - vectordb_dirs: 可能的向量数据库目录列表
    """
    vectordb_dirs = []
    
    # 首先检查常用位置
    common_dirs = [
        os.path.join(base_dir, "vectordb"),
        os.path.join(base_dir, "langchian", "langchain-rag", "vectordb"),
        os.path.join(base_dir, "langchian", "langchain-rag", "temp_vectordb"),
        "vectordb",
        "temp_vectordb"
    ]
    
    # 添加确实存在并包含Chroma数据库文件的目录
    for dir_path in common_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # 检查是否包含Chroma数据库文件
            if os.path.exists(os.path.join(dir_path, "chroma.sqlite3")):
                vectordb_dirs.append(dir_path)
    
    return vectordb_dirs

# 主应用逻辑
def main():
    # 设置页面标题
    st.title("文档问答系统")
    
    # 侧边栏内容
    with st.sidebar:
        # 显示调试信息
        st.info("系统状态: 正常")
        
        # 添加清除按钮
        if st.button("🗑️ 清除所有记录", help="清除聊天记录和加载的文档"):
            # 清除所有会话状态，但不使用experimental_rerun
            st.session_state["messages"] = []
            st.session_state["retriever"] = None
            st.session_state["last_query"] = None
            if "last_context" in st.session_state:
                del st.session_state["last_context"]
            st.success("已清除所有记录！")
            # 返回，避免继续执行下面的代码
            return
        
        # 调试模式开关
        debug_mode = st.checkbox("开启调试模式", value=False, help="显示详细的处理过程")
        if debug_mode:
            st.session_state["debug_mode"] = True
        else:
            st.session_state["debug_mode"] = False
    
    # 主内容区
    # 强制显示上传功能
    st.markdown("## 选择操作模式")
    operation_mode = st.radio(
        "操作方式:",
        ["上传新文档", "加载本地向量库"]
    )
    
    retriever = None
    
    if operation_mode == "上传新文档":
        st.markdown("### 📁 上传文件区域")
        
        # 直接在主界面显示上传组件
        uploaded_files = st.file_uploader(
            "选择要上传的文档", 
            accept_multiple_files=True, 
            type=["txt", "md", "pdf"],
            help="支持txt、md和pdf格式的文档"
        )
        
        # 显示上传状态
        if uploaded_files:
            st.success(f"已选择 {len(uploaded_files)} 个文件")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size} 字节)")
        else:
            st.warning("尚未选择任何文件，请点击上方'Browse files'按钮选择文件")
        
        # 是否保存到本地
        save_locally = st.checkbox("保存到本地向量库")
        local_path = None
        
        if save_locally:
            local_path = st.text_input("本地向量库保存路径:", value="/Users/yeziyin/vectordb")
            
            # 检查目录权限
            if local_path:
                if check_directory_writable(local_path):
                    st.success(f"✅ 目录 {local_path} 可写入")
                else:
                    st.error(f"❌ 无法写入目录 {local_path}，请检查权限或选择其他目录")
            
        # 处理上传的文档
        process_button = st.button("📄 处理文档")
        if process_button:
            if not uploaded_files:
                st.error("请先上传文件再处理")
            else:
                with st.spinner("正在处理文档..."):
                    # 保存上传的文件到临时目录
                    temp_dir = "temp_docs"
                    if not check_directory_writable(temp_dir):
                        st.error(f"临时目录 {temp_dir} 不可写入，请检查应用权限")
                        return
                        
                    # 清理临时目录
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            st.error(f"删除文件时出错: {str(e)}")
                    
                    # 保存上传的文件
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        try:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)
                            st.info(f"已保存文件: {file_path}")
                        except Exception as e:
                            st.error(f"保存文件 {uploaded_file.name} 时出错: {str(e)}")
                    
                    if not file_paths:
                        st.error("没有成功保存任何文件，无法继续处理")
                        return
                        
                    # 创建临时向量数据库目录
                    vectordb_dir = "temp_vectordb"
                    
                    # 处理文档
                    retriever = process_documents(
                        file_paths, 
                        vectordb_dir,
                        save_to_local=save_locally,
                        local_path=local_path
                    )
                    
                    if retriever:
                        st.session_state.retriever = retriever
                        st.success("文档处理完成，可以开始提问")
                    else:
                        st.error("处理文档时出错，请查看日志")
    
    elif operation_mode == "加载本地向量库":
        # 查找可能的向量库目录
        vectordb_dirs = find_vectordb_dirs()
        
        # 提供自定义路径输入
        custom_path = st.text_input("添加本地向量库路径:", value="/Users/yeziyin/vectordb")
        if custom_path and custom_path not in vectordb_dirs and os.path.exists(custom_path):
            vectordb_dirs.append(custom_path)
        
        # 没有找到任何向量库目录时的提示
        if not vectordb_dirs:
            st.warning("未找到任何向量库目录，请手动输入路径")
        else:
            st.success(f"找到 {len(vectordb_dirs)} 个可能的向量库目录")
        
        # 显示所有可选向量库
        options = []
        for db_dir in vectordb_dirs:
            # 检查是否是有效的Chroma数据库
            has_db_file = os.path.exists(os.path.join(db_dir, "chroma.sqlite3"))
            status = "✅" if has_db_file else "❓"
            options.append(f"{status} {db_dir}")
        
        # 多选框组件
        if options:
            st.write("### 选择要使用的向量库:")
            selected_options = st.multiselect(
                "可用的向量库列表",
                options,
                default=[options[0]] if options else None,
                help="选择一个或多个向量库作为知识来源"
            )
            
            # 提取路径
            selected_dirs = [opt.split(" ", 1)[1] for opt in selected_options]
        else:
            selected_dirs = []
            if custom_path and os.path.exists(custom_path):
                selected_dirs = [custom_path]
        
        # 加载向量库
        if st.button("加载选中的向量库"):
            # 清除可能的旧状态
            if "retriever" in st.session_state:
                del st.session_state["retriever"]
            
            with st.spinner("正在加载向量库..."):
                if selected_dirs:
                    # 加载多个向量库
                    retriever = load_multiple_vectordbs(selected_dirs)
                    
                    if retriever:
                        st.session_state.retriever = retriever
                        st.success("✅ 向量库加载成功，可以开始提问")
                    else:
                        st.error("❌ 加载向量库失败")
                else:
                    st.error("请至少选择一个向量库")

    # 使用一个容器来组织问答区域
    qa_container = st.container()
    
    with qa_container:
        # 主界面：提问部分
        if 'retriever' in st.session_state:
            retriever = st.session_state.retriever
            st.success("📚 向量数据库已加载")
            
            # 用户提问区域
            st.markdown("---")
            st.header("向文档提问")
            user_query = st.text_input("输入您的问题:")
            
            # 添加查询状态管理，避免重复处理
            if user_query:
                # 创建两个列来分别显示结果和调试信息
                if st.session_state.get("debug_mode", False):
                    # 调试模式下使用一列
                    result_col = st.container()
                else:
                    # 普通模式下使用两列
                    result_col, debug_col = st.columns([3, 1])
                
                # 检查是否是新查询
                is_new_query = user_query != st.session_state.get("last_query")
                
                if is_new_query:
                    # 处理用户查询
                    with st.spinner("正在思考..."):
                        # 如果开启调试模式，直接在页面上显示调试信息
                        if st.session_state.get("debug_mode", False):
                            response_text, context = process_query(user_query, retriever)
                        else:
                            # 非调试模式下，创建一个捕获调试输出的上下文管理器
                            with st.echo() as debug_output:
                                # 临时关闭st.write输出
                                def silent_write(*args, **kwargs):
                                    pass
                                original_write = st.write
                                st.write = silent_write
                                
                                try:
                                    response_text, context = process_query(user_query, retriever)
                                finally:
                                    # 恢复st.write
                                    st.write = original_write
                    
                    # 保存查询状态和聊天历史
                    st.session_state["last_query"] = user_query
                    st.session_state["messages"].append({"role": "user", "content": user_query})
                    st.session_state["messages"].append({"role": "assistant", "content": response_text})
                    st.session_state["last_context"] = context
                else:
                    # 使用保存的回答
                    response_text = st.session_state["messages"][-1]["content"]
                    context = st.session_state.get("last_context", "")
                
                # 显示结果
                with result_col:
                    st.subheader("回答:")
                    st.markdown(response_text)
                    
                    # 可折叠的参考文档区域
                    with st.expander("参考文档内容"):
                        st.markdown(context)
        
            # 聊天历史显示
            if "messages" in st.session_state and st.session_state["messages"]:
                st.markdown("---")
                st.subheader("聊天历史")
                for i, msg in enumerate(st.session_state["messages"]):
                    if msg["role"] == "user":
                        st.markdown(f"**问题**: {msg['content']}")
                    else:
                        st.markdown(f"**回答**: {msg['content']}")
                
                # 除了最后一条消息外，每条消息后添加分隔线
                if i < len(st.session_state["messages"]) - 1:
                    st.markdown("---")
                    
# 添加页脚
st.markdown("---")
st.markdown("文档问答系统 | 基于LangChain和DeepSeek")

# 处理用户查询
def process_query(query_text, retriever):
    """
    处理用户查询并返回回答和上下文
    
    参数:
    - query_text: 用户查询文本
    - retriever: 检索器对象，可以是向量数据库或合并检索器
    
    返回:
    - response: 模型回答
    - context: 检索到的上下文
    """
    if not query_text:
        return "请输入问题", ""
        
    try:
        # 打印调试信息
        st.info(f"开始处理查询: {query_text}")
        
        # 检查检索器类型
        retriever_type = type(retriever).__name__
        st.write(f"检索器类型: {retriever_type}")
        
        # 获取相关文档
        relevant_docs = []
        
        # 根据检索器类型使用不同的检索方法
        if hasattr(retriever, "as_retriever") and callable(retriever.as_retriever):
            # 是向量数据库对象，需要转换为检索器
            st.write("检测到向量数据库对象，转换为检索器...")
            try:
                # 尝试MMR检索
                st.write("尝试MMR检索...")
                mmr_docs = retriever.max_marginal_relevance_search(
                    query_text, 
                    k=5,
                    fetch_k=10
                )
                if mmr_docs:
                    st.write(f"MMR检索成功，找到 {len(mmr_docs)} 个文档")
                    relevant_docs = mmr_docs
            except Exception as e:
                st.write(f"MMR检索失败: {str(e)}")
                try:
                    # 尝试相似度检索
                    st.write("尝试相似度检索...")
                    sim_docs = retriever.similarity_search(
                        query_text, 
                        k=5
                    )
                    if sim_docs:
                        st.write(f"相似度检索成功，找到 {len(sim_docs)} 个文档")
                        relevant_docs = sim_docs
                except Exception as e2:
                    st.write(f"相似度检索失败: {str(e2)}")
        elif hasattr(retriever, "get_relevant_documents") and callable(retriever.get_relevant_documents):
            # 已经是检索器对象
            st.write("使用检索器直接获取文档...")
            try:
                docs = retriever.get_relevant_documents(query_text)
                st.write(f"检索成功，找到 {len(docs)} 个文档")
                relevant_docs = docs
            except Exception as e:
                st.write(f"检索失败: {str(e)}")
                
                # 如果是合并检索器，尝试单独调用每个检索器
                if hasattr(retriever, "retrievers"):
                    st.write("尝试单独调用每个检索器...")
                    for i, sub_retriever in enumerate(retriever.retrievers):
                        try:
                            sub_docs = sub_retriever.get_relevant_documents(query_text)
                            st.write(f"检索器 {i+1} 返回 {len(sub_docs)} 个文档")
                            relevant_docs.extend(sub_docs)
                        except Exception as sub_e:
                            st.write(f"检索器 {i+1} 失败: {str(sub_e)}")
        else:
            # 尝试直接使用get方法
            st.write("尝试直接获取文档集合...")
            try:
                if hasattr(retriever, "get") and callable(retriever.get):
                    collection_docs = retriever.get()
                    if collection_docs and 'documents' in collection_docs:
                        st.write(f"获取到 {len(collection_docs['documents'])} 个文档")
                        from langchain_core.documents import Document
                        relevant_docs = [
                            Document(
                                page_content=doc,
                                metadata=meta if meta else {}
                            ) 
                            for doc, meta in zip(
                                collection_docs['documents'][:5], 
                                collection_docs.get('metadatas', [{}] * len(collection_docs['documents']))[:5]
                            )
                        ]
            except Exception as e:
                st.write(f"直接获取文档失败: {str(e)}")
                
        # 检查是否找到相关文档
        if not relevant_docs:
            st.warning("未找到相关文档，无法回答问题")
            return "我无法从文档中找到这个问题的答案。可能文档中不包含相关信息，或数据库访问出现问题。", ""
        
        # 对检索到的文档进行排序和去重
        try:
            # 去掉完全相同的文档
            unique_content = set()
            filtered_docs = []
            for doc in relevant_docs:
                if doc.page_content not in unique_content:
                    unique_content.add(doc.page_content)
                    filtered_docs.append(doc)
            
            relevant_docs = filtered_docs[:5]  # 限制最多5个文档
            st.write(f"过滤后剩余 {len(relevant_docs)} 个文档")
        except Exception as e:
            st.write(f"文档去重失败: {str(e)}")
        
        # 显示检索到的文档内容
        st.write("已检索到以下文档:")
        for i, doc in enumerate(relevant_docs[:3]):
            st.write(f"文档 {i+1}: {doc.page_content[:100]}...")
        
        # 将检索到的文档内容组合成上下文
        context = "\n\n".join([f"文档片段 {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # 构建更强的提示模板
        prompt = f"""请根据以下信息回答用户的问题。
如果无法从提供的信息中找到答案，请直接回复"我无法从文档中找到这个问题的答案"。
不要编造信息，只使用提供的内容作答。
先分析文档内容与问题的相关性，整合相关信息，然后给出具体答案。

文档内容：
{context}

用户问题：{query_text}

请提供详细、准确、有条理的回答："""

        # 发送到DeepSeek模型获取回答
        st.write("发送请求到DeepSeek模型...")
        api_key = "sk-ff6c3fa4fc4e453b92b7d023cd9efc4e"
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的文档问答助手，根据提供的文档内容回答用户问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 降低温度，提高回答的一致性和准确性
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        st.write("模型已返回回答")
        return answer, context
        
    except Exception as e:
        log_error("处理查询时出错", e)
        return "抱歉，处理您的查询时发生错误。", ""

# 确保主函数在启动时执行
if __name__ == "__main__":
    init_session()
    main()
