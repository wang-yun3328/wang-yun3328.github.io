# 本地知识库数字人问答系统
## 项目简介
基于 **RAG 检索增强生成** 技术实现的纯本地知识库数字人系统，支持上传自定义文档构建专属知识库，结合静态人脸图片自动生成口型同步、带自然眨眼动作的数字人回答视频。全程本地离线部署，无需调用云端API，保障数据隐私。

## 技术栈
| 模块 | 技术选型 |
|------|----------|
| 前端交互 | Streamlit |
| RAG 应用框架 | LangChain（LCEL 表达式语法） |
| 向量数据库 | Chroma |
| 本地大模型引擎 | Ollama + Qwen:7B |
| 数字人驱动 | dlib + OpenCV（68点人脸关键点检测 + 三角网格形变） |
| 离线语音合成 | pyttsx3 |
| 音视频合成 | FFmpeg |

## 核心功能
1. **本地知识库问答（RAG）**
   - 支持 TXT、PDF 格式文档导入，自动构建本地私有知识库
   - 基于向量相似度检索实现精准文本召回，大模型仅基于知识库内容回答，大幅降低幻觉
   - 向量数据库持久化存储，重启应用无需重复构建知识库

2. **离线语音合成**
   - 纯本地语音生成，支持调节语速、音调、音量
   - 生成音频自动驱动数字人口型动作，实现音画同步

3. **数字人视频生成**
   - 单张静态人脸图片即可驱动，自动检测68个人脸关键点
   - 实时生成自然的嘴巴开合、随机眨眼动态效果
   - 肤色融合 + 边缘平滑算法，消除拼接黑边与违和感
   - 自动完成音视频合成，输出完整 MP4 格式视频

---

## 核心原理：RAG 与 LangChain 的关系与代码实现
### 1. 基础概念
- **RAG（检索增强生成）**：是一种大模型应用的**技术方案/架构**。核心逻辑是「先检索、后生成」：用户提问时，先从本地知识库中召回最相关的文本片段，再把上下文和问题一起传给大模型，让大模型基于给定内容生成回答，从根源上减少幻觉，同时支持私有数据落地。
- **LangChain**：是一个大模型应用开发**工具框架**。它把 RAG 全流程需要的所有环节（文档加载、文本分割、向量化、向量存储、检索、Prompt管理、模型调用、输出解析）都做了标准化封装，开发者不用从零实现每个模块，通过链式调用就能快速搭建完整的 RAG 系统。

简单说：**RAG 是做事情的思路，LangChain 是实现这个思路的现成工具箱**。

### 2. 代码中的对应实现
本项目基于 LangChain 的 LCEL（LangChain Expression Language）表达式语法，完整实现了 RAG 全链路，核心逻辑都在 `build_knowledge_base()` 函数中，分层对应如下：

#### ① 数据接入层：文档加载
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
```
- 对应 RAG 的「数据准备」阶段
- 自动读取、解析 TXT/PDF 格式的知识库文档，转换成 LangChain 标准的 `Document` 对象格式

#### ② 预处理层：文本分片
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
texts = splitter.split_documents(documents)
```
- 对应 RAG 的「文本分块」阶段
- 把长文档切分成固定大小的文本块（chunk），保留重叠字符避免上下文截断
- 分片大小直接决定后续的检索精度和生成质量

#### ③ 检索核心层：向量化 + 向量存储 + 检索器
```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# 文本转向量
embeddings = OllamaEmbeddings(model="qwen:7b")
# 存入向量数据库
db = Chroma.from_documents(texts, embeddings, persist_directory=DB_DIR)
# 封装为检索器
retriever = db.as_retriever(search_kwargs={"k": 3})
```
- 对应 RAG 的「向量化 → 存储 → 检索」完整链路
- 用本地 Ollama 模型把文本块转换成高维向量（Embedding）
- 存入 Chroma 向量数据库，用于后续的相似度匹配检索
- 封装成统一的检索器，用户提问时自动召回最相关的 3 条文本片段

#### ④ 生成层：Prompt 模板 + 大模型 + 输出解析
```python
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 提示词模板：约束大模型只基于上下文回答
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="根据以下上下文回答问题：\n上下文：{context}\n问题：{question}\n回答："
)
# 本地大模型
llm = Ollama(model="qwen:7b", temperature=0.1)
# 输出解析器：把模型输出转为纯字符串
parser = StrOutputParser()
```
- 对应 RAG 的「生成回答」阶段
- PromptTemplate 统一管理提示词格式，强制大模型遵循「基于给定上下文回答」的规则
- Ollama 负责本地推理生成最终回答
- StrOutputParser 标准化输出格式，方便后续业务逻辑调用

#### ⑤ LangChain 核心：链式组装
```python
from langchain_core.runnables import RunnablePassthrough

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- 这是 LangChain 最核心的价值：用 `|` 管道符把 RAG 的所有环节串联成一条完整的执行链
- 执行流程：用户问题 → 检索器召回上下文 → 自动填充 Prompt 模板 → 大模型生成回答 → 字符串解析输出
- 调用时只需要一行代码 `qa_chain.invoke(question)`，即可完成完整的 RAG 问答

---

## 环境准备
1. **安装 Python 依赖**
```bash
pip install streamlit opencv-python numpy pillow pyttsx3 dlib langchain langchain-community langchain-chroma langchain-text-splitters
```

2. **配置 FFmpeg**
- 下载 Windows 版 FFmpeg，修改代码中 `FFMPEG_PATH` 和 `FFPROBE_PATH` 为你的本地路径

3. **下载 dlib 人脸模型**
- 下载 `shape_predictor_68_face_landmarks.dat` 关键点模型，放在项目根目录

4. **安装 Ollama 并拉取模型**
```bash
# 安装 Ollama 后执行
ollama pull qwen:7b
```

## 运行方式
```bash
streamlit run app.py
```
运行后浏览器自动打开页面：
1. 左侧上传一张清晰的正面人脸照片
2. 右侧输入问题，点击生成按钮
3. 系统自动完成「知识库检索 → 回答生成 → 语音合成 → 数字人视频合成」全流程

## 项目结构
```
.
├── app.py                              # 主程序入口
├── knowledge_files/                    # 知识库文档存放目录
├── chroma_db/                          # 向量数据库持久化目录
├── tmp/                                # 临时文件目录（图片、语音、视频）
└── shape_predictor_68_face_landmarks.dat  # dlib 人脸关键点模型
```

## 优化亮点
- 纯本地离线运行，无数据泄露风险，适配内网环境
- 口型动作采用多频率正弦波叠加，效果自然不机械
- 肤色过渡 + 边缘平滑算法，消除人脸拼接黑边与重影
- 知识库持久化存储，重启应用无需重新向量化
- 基于 LangChain LCEL 架构，链路清晰，易于扩展新功能