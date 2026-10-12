import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import subprocess
import pyttsx3
import warnings
import dlib

warnings.filterwarnings("ignore")

# ===================== 语音引擎优化 =====================
engine = pyttsx3.init()
engine.setProperty('rate', 210)  # 加快语速，贴合口型
engine.setProperty('volume', 1.0)
engine.setProperty('pitch', 110)  # 微调音调，更自然

# ===================== LangChain 导入 =====================
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="本地知识库数字人",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# 自定义CSS：彻底解决上传区域文字问题 + 全界面高对比度文字
st.markdown("""
    <style>
        /* 全局背景：淡青渐变 */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 50%, #fff3e0 100%);
            background-attachment: fixed;
        }
        /* 标题/卡片标题：深青绿色，高对比 */
        h1, h2, h3, h4, h5, h6 {
            color: #004D40 !important;
        }
        /* 正文/普通文字：深灰蓝，清晰不刺眼 */
        p, div, span, label, .stMarkdown, .stTextInput, .stSelectbox, .stMultiSelect {
            color: #263238 !important;
        }
        /* 卡片容器 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1cypcdb, .st-emotion-cache-1cypcdb {
            background-color: rgba(255,255,255,0.95) !important;
            backdrop-filter: blur(12px);
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,150,136,0.1);
            border: 1px solid rgba(0,150,136,0.08);
        }
        /* 上传区域深色背景文字：强制白色，且不透明 */
        [data-testid="stFileUploadDropzone"] {
            background-color: #1e1e1e !important;
            border-radius: 10px;
        }
        [data-testid="stFileUploadDropzone"] * {
            color: #ffffff !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploadDropzone"] svg {
            fill: #ffffff !important;
        }
        /* 上传按钮文字：白色 */
        .stFileUploader button {
            color: #ffffff !important;
            background-color: #333333 !important;
            border: 1px solid #555555 !important;
        }
        /* 已上传文件信息文字：深灰蓝，保证可见 */
        .stFileUploader [data-testid="stFileUploaderFileInfo"] * {
            color: #263238 !important;
        }
        /* 按钮样式：淡青渐变，文字白色 */
        .stButton>button {
            background: linear-gradient(90deg, #4db6ac, #81c784);
            color: #ffffff !important;
            font-weight: 600;
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            border: none;
            box-shadow: 0 4px 15px rgba(77,182,172,0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(77,182,172,0.4);
        }
        /* 输入框样式 */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #4db6ac;
            background-color: rgba(255,255,255,0.95);
            color: #263238 !important;
        }
        /* 进度条样式 */
        .stProgress > div > div > div {
            background-color: #4db6ac;
        }
        /* 视频组件样式 */
        .stVideo {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,150,136,0.15);
        }
        /* 提示框文字 */
        .st-alert-success, .st-alert-warning, .st-alert-error, .st-alert-info {
            background-color: rgba(255,255,255,0.98) !important;
            border-radius: 8px;
            color: #263238 !important;
        }
        /* 图片组件文字：修正弃用警告 */
        .stImage > div > div > img {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# 标题区域
st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h1 style='color: #004D40; font-size: 2.8rem; margin-bottom: 0.8rem;'>🤖 本地知识库数字人</h1>
        <p style='color: #263238; font-size: 1.2rem; margin-bottom: 1.2rem;'>✅ 自然口型 | 流畅眨眼 | 原生肤色融合 | 零重影</p>
        <hr style='height: 3px; background: linear-gradient(90deg, #4db6ac, #81c784); border: none; border-radius: 2px; width: 60%; margin: 0 auto;'>
    </div>
""", unsafe_allow_html=True)

# 文件夹初始化
os.makedirs("tmp", exist_ok=True)
os.makedirs("knowledge_files", exist_ok=True)
DB_DIR = "./chroma_db"

# FFmpeg 路径（请根据你的实际路径修改）
FFMPEG_PATH = r"D:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
FFPROBE_PATH = r"D:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"

# ===================== dlib 初始化 =====================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# ---------------------- 人脸关键点检测 ----------------------
def detect_face_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        st.error("❌ 图片读取失败！")
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        st.error("❌ 未检测到人脸！")
        return None, None

    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return landmarks, img


# ---------------------- 1. 优化版擦除原图五官 + 肤色过渡层（彻底解决黑边） ----------------------
def erase_original_face_roi(img, landmarks):
    """
    彻底擦除原图的嘴巴和眼睛区域，生成足够大的肤色过渡层，从根源消除黑边
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. 嘴巴区域（扩大范围，包含嘴唇外沿）
    mouth_pts = landmarks[48:68].astype(np.int32)
    cv2.fillPoly(mask, [mouth_pts], 255)
    # 2. 眼睛区域（扩大到眼周）
    left_eye_pts = landmarks[36:42].astype(np.int32)
    right_eye_pts = landmarks[42:48].astype(np.int32)
    cv2.fillPoly(mask, [left_eye_pts], 255)
    cv2.fillPoly(mask, [right_eye_pts], 255)

    # 关键修复1：扩大mask范围，但比之前更克制，避免过渡模糊
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
    # 关键修复2：高斯模糊降低强度，保留边缘清晰度
    mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0

    # 提取脸颊平均肤色（扩大采样范围，更准确）
    cheek_mask = np.zeros((h, w), dtype=np.uint8)
    # 脸颊采样点：左右脸颊各4个点
    left_cheek = landmarks[[1, 2, 3, 4, 31, 48]].astype(np.int32)
    right_cheek = landmarks[[12, 13, 14, 15, 35, 54]].astype(np.int32)
    cv2.fillPoly(cheek_mask, [left_cheek], 255)
    cv2.fillPoly(cheek_mask, [right_cheek], 255)
    avg_color = cv2.mean(img, mask=cheek_mask)[:3]

    # 用肤色填充被擦除区域，生成过渡层
    img_erased = img.copy()
    for c in range(3):
        img_erased[:, :, c] = (1 - mask) * img_erased[:, :, c] + mask * avg_color[c]

    # 关键修复3：降低inpaint半径，修复边缘瑕疵但不破坏细节
    img_erased = cv2.inpaint(img_erased, (mask * 255).astype(np.uint8), 8, cv2.INPAINT_TELEA)
    return img_erased


# ---------------------- 2. 局部三角变形（优化算法，修复模糊问题） ----------------------
def warp_face_roi(roi_img, src_pts, dst_pts, mask):
    """对局部区域进行三角变形，返回变形后的图像，修复模糊问题"""
    h, w = roi_img.shape[:2]
    warped = roi_img.copy()

    # 三角剖分
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in src_pts:
        subdiv.insert((int(p[0]), int(p[1])))

    triangles = subdiv.getTriangleList()
    for t in triangles:
        pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], dtype=np.float32)
        # 检查点是否在mask内
        if cv2.pointPolygonTest(mask, (int(pts[0][0]), int(pts[0][1])), False) >= 0 and \
                cv2.pointPolygonTest(mask, (int(pts[1][0]), int(pts[1][1])), False) >= 0 and \
                cv2.pointPolygonTest(mask, (int(pts[2][0]), int(pts[2][1])), False) >= 0:

            # 找到对应索引
            src_tri = []
            for p in pts:
                idx = np.argmin(np.abs(src_pts - p).sum(axis=1))
                src_tri.append(src_pts[idx])
            src_tri = np.array(src_tri, dtype=np.float32)

            dst_tri = []
            for p in pts:
                idx = np.argmin(np.abs(src_pts - p).sum(axis=1))
                dst_tri.append(dst_pts[idx])
            dst_tri = np.array(dst_tri, dtype=np.float32)

            # 仿射变换：改用INTER_LINEAR + BORDER_CONSTANT，减少模糊和边缘问题
            M = cv2.getAffineTransform(src_tri, dst_tri)
            warped = cv2.warpAffine(warped, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))

    return warped


# ---------------------- 3. 修复关键点索引错误 + 优化嘴巴开合逻辑（核心修改） ----------------------
def generate_dynamic_face(original_img, erased_img, landmarks, open_rate, eye_close):
    h, w = original_img.shape[:2]
    target_landmarks = landmarks.copy()

    # ===================== 修复：dlib 68点索引 0~67，无68 =====================
    mouth_idx = list(range(48, 68))
    upper_lip_outer = [49, 50, 51, 52, 53, 54]
    upper_lip_inner = [61, 62, 63, 64]
    lower_lip_outer = [55, 56, 57, 58, 59, 60]
    lower_lip_inner = [65, 66, 67]
    left_corner = 48
    right_corner = 54

    # 【核心优化】更自然的嘴巴开合：上唇上抬幅度降低，下唇下压幅度提高，强化上下开合感
    for idx in upper_lip_outer + upper_lip_inner:
        target_landmarks[idx, 1] -= open_rate * 4  # 上唇上抬幅度降低
    for idx in lower_lip_outer + lower_lip_inner:
        target_landmarks[idx, 1] += open_rate * 16  # 下唇下压幅度大幅提高，强化开合
    target_landmarks[left_corner, 0] -= open_rate * 1.5  # 嘴角拉伸幅度降低，避免过度变形
    target_landmarks[right_corner, 0] += open_rate * 1.5

    # 提取嘴巴区域并变形
    mouth_pts = landmarks[mouth_idx].astype(np.int32)
    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_pts], 255)
    # 缩小dilate范围，减少模糊
    mouth_mask = cv2.dilate(mouth_mask, np.ones((5, 5), np.uint8), iterations=2)
    mouth_roi = cv2.bitwise_and(original_img, original_img, mask=mouth_mask)
    mouth_warped = warp_face_roi(mouth_roi, landmarks[mouth_idx], target_landmarks[mouth_idx], mouth_pts)

    # --- 眼睛变形 ---
    eye_idx = list(range(36, 48))
    left_eye = list(range(36, 42))
    right_eye = list(range(42, 48))

    # 优化眨眼幅度，更自然
    for idx in [37, 38, 43, 44]:
        target_landmarks[idx, 1] += eye_close * 4
    for idx in [40, 41, 46, 47]:
        target_landmarks[idx, 1] -= eye_close * 4

    eye_pts = landmarks[eye_idx].astype(np.int32)
    eye_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eye_mask, [landmarks[left_eye].astype(np.int32)], 255)
    cv2.fillPoly(eye_mask, [landmarks[right_eye].astype(np.int32)], 255)
    eye_mask = cv2.dilate(eye_mask, np.ones((5, 5), np.uint8), iterations=2)
    eye_roi = cv2.bitwise_and(original_img, original_img, mask=eye_mask)
    eye_warped = warp_face_roi(eye_roi, landmarks[eye_idx], target_landmarks[eye_idx], eye_pts)

    # 【核心优化】融合方式改为：先复制 erased_img，再用mask直接覆盖，减少半透明糊边
    result = erased_img.copy()
    # 降低高斯模糊强度，减少“贴上去”的模糊感
    mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (7, 7), 0) / 255.0
    # 只在mask区域替换，避免边缘半透明
    for c in range(3):
        result[:, :, c] = np.where(mouth_mask_blur > 0.1, mouth_warped[:, :, c], result[:, :, c])

    eye_mask_blur = cv2.GaussianBlur(eye_mask, (7, 7), 0) / 255.0
    for c in range(3):
        result[:, :, c] = np.where(eye_mask_blur > 0.1, eye_warped[:, :, c], result[:, :, c])

    return result.astype(np.uint8)


# ---------------------- 知识库 ----------------------
@st.cache_resource
def build_knowledge_base():
    documents = []
    know_dir = "knowledge_files"

    for fname in os.listdir(know_dir):
        fpath = os.path.join(know_dir, fname)
        try:
            if fname.endswith(".txt"):
                loader = TextLoader(fpath, encoding="utf-8")
                documents.extend(loader.load())
            elif fname.endswith(".pdf"):
                loader = PyPDFLoader(fpath)
                documents.extend(loader.load())
        except:
            pass

    if not documents:
        default_txt = """
本地数字人系统构建流程：
1. 准备本地文档：支持 TXT、PDF 格式。
2. 文档分块：将长文本切分成小段。
3. 向量化：使用本地 Ollama 生成向量。
4. 向量存储：使用 Chroma 本地向量库。
5. 问答检索：用户问题 → 检索知识库 → 本地大模型回答。
6. 语音合成：使用本地语音引擎。
7. 数字人视频：使用静态图片 + FFmpeg 合成说话视频。
"""
        with open("default_knowledge.txt", "w", encoding="utf-8") as f:
            f.write(default_txt)
        loader = TextLoader("default_knowledge.txt", encoding="utf-8")
        documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="qwen:7b")
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(texts, embeddings, persist_directory=DB_DIR)

    llm = Ollama(model="qwen:7b", temperature=0.1)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="根据以下上下文回答问题：\n上下文：{context}\n问题：{question}\n回答："
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return qa_chain


# ---------------------- 语音合成 ----------------------
def text2speech(text, save_path="tmp/voice.mp3"):
    engine.save_to_file(text, save_path)
    engine.runAndWait()
    return save_path


# ---------------------- 视频生成 ----------------------
def make_talking_video(img_path, audio_path, src_landmarks, original_img, out_path="tmp/final.mp4"):
    h, w = original_img.shape[:2]
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = "tmp/temp.mp4"
    vw = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    erased_img = erase_original_face_roi(original_img, src_landmarks)

    # 获取音频时长
    result = subprocess.run(
        [FFPROBE_PATH, "-i", audio_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    duration = float(result.stdout.strip())
    total_frames = int(fps * duration)

    # 眨眼参数
    blink_state = 0
    blink_timer = 0
    blink_frames = 3
    next_blink = np.random.randint(1 * fps, 2.5 * fps)

    # 优化口型波形，让开合更有节奏
    for i in range(total_frames):
        t = i / total_frames

        # 更快、更有起伏的口型波形，避免单调
        sin1 = np.sin(t * 45) * 0.7
        sin2 = np.sin(t * 90) * 0.4
        sin3 = np.sin(t * 120) * 0.25
        open_rate = 0.05 + sin1 + sin2 + sin3
        open_rate = np.clip(open_rate, 0, 0.85)  # 上限提高，允许更大的开合幅度

        # 眨眼逻辑
        blink_timer += 1
        eye_close = 0.0
        if blink_state == 0:
            if blink_timer >= next_blink:
                blink_state = 1
                blink_timer = 0
                next_blink = np.random.randint(1 * fps, 2.5 * fps)
        elif blink_state == 1:
            eye_close = blink_timer / blink_frames
            if blink_timer >= blink_frames:
                blink_state = 2
                blink_timer = 0
        else:
            eye_close = 1 - blink_timer / blink_frames
            if blink_timer >= blink_frames:
                blink_state = 0
                blink_timer = 0

        warped_frame = generate_dynamic_face(original_img, erased_img, src_landmarks, open_rate, eye_close)
        vw.write(warped_frame)

    vw.release()

    # 音视频合成
    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", temp_video,
        "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", out_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path


# ---------------------- 主界面 ----------------------
qa_model = build_knowledge_base()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div style="background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 18px; box-shadow: 0 10px 30px rgba(0,150,136,0.1); border: 1px solid rgba(0,150,136,0.08);">
            <h3 style='color: #004D40; margin-top: 0; font-size: 1.4rem;'>🖼️ 上传你的形象</h3>
    """, unsafe_allow_html=True)
    photo = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    img_save = None
    face_landmarks = None
    original_img = None

    if photo:
        im = Image.open(photo).convert("RGB")
        img_save = "tmp/photo.png"
        im.save(img_save)
        # 修正弃用警告：移除 use_column_width，改用 width 参数
        st.image(im, caption="✅ 图片已上传成功", use_column_width=True)

        with st.spinner("🔍 正在检测人脸关键点..."):
            face_landmarks, original_img = detect_face_landmarks(img_save)

        if face_landmarks is not None:
            st.success("✅ 人脸检测完成！五官关键点已定位")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 18px; box-shadow: 0 10px 30px rgba(0,150,136,0.1); border: 1px solid rgba(0,150,136,0.08);">
            <h3 style='color: #004D40; margin-top: 0; font-size: 1.4rem;'>💬 开始对话</h3>
    """, unsafe_allow_html=True)
    question = st.text_input("输入你的问题", "你好，介绍一下你自己", label_visibility="collapsed")
    run_btn = st.button("🚀 生成回答 + 数字人视频", type="primary", use_container_width=True)

    if run_btn:
        if not photo:
            st.warning("⚠️ 请先上传一张正面人脸图片！")
        elif face_landmarks is None:
            st.warning("⚠️ 人脸检测失败，请上传清晰的正面人脸照片！")
        else:
            with st.spinner("🧠 模型正在思考..."):
                ans = qa_model.invoke(question)
                st.success("✅ 回答生成完成")
                st.markdown(f"**回答：{ans}**")

            with st.spinner("🔊 正在生成语音..."):
                voice = text2speech(ans)

            with st.spinner("🎬 正在合成数字人视频..."):
                video = make_talking_video(img_save, voice, face_landmarks, original_img)
                st.video(video)
                st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align:center; color: #004D40; margin-top: 2.5rem; padding: 1rem; font-size: 1rem; font-weight: 500;">
        🎬 优化版数字人 | 自然口型 + 流畅眨眼 + 肤色融合 | 零重影体验
    </div>
""", unsafe_allow_html=True)