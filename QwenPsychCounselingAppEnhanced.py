import os
import time
import logging
import threading
import queue
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import gradio as gr
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info, generate_speech
import librosa
import torch
import numpy as np
import cv2
import sounddevice as sd

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载模型和处理器
try:
    logger.info("加载 Qwen2.5-Omni-7B 模型...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True
    )
    logger.info("模型和处理器加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise

# 系统提示
sys_prompt = (
    "你是一名专业的心理咨询师助手，擅长通过视频和文本交互分析用户情绪和性格特征，提供共情支持和建议。 "
    "你的目标是倾听用户，理解情绪，生成温暖且专业的回应，同时避免给出医疗建议。"
)

# 对话历史和情绪统计
history: List[Dict] = []
emotion_history = defaultdict(int)

# 存储路径
TEMP_DIR = "temp_files"
OUTPUT_DIR = "recorded_videos"
for dir_path in [TEMP_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 全局变量
video_capture = None
audio_input_stream = None
audio_queue = queue.Queue()
video_frames = []
stop_thread = False
is_recording = False
out_video = None

def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: Any):
    """音频流回调函数"""
    global is_recording, out_video
    if status:
        logger.warning(f"音频捕获状态: {status}")
    audio_queue.put(indata.copy())
    if is_recording and out_video:
        out_video.write(indata.tobytes())

def start_stream():
    """启动视频和音频流捕获"""
    global video_capture, stop_thread, video_frames
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("无法打开摄像头")
        raise Exception("无法打开摄像头")

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30

    stream = sd.InputStream(
        samplerate=processor.feature_extractor.sampling_rate,
        channels=1,
        callback=audio_callback
    )
    stream.start()

    def capture_loop():
        global video_frames, is_recording
        while not stop_thread:
            ret, frame = video_capture.read()
            if not ret:
                logger.warning("无法读取视频帧")
                break
            video_frames.append(frame)
            if is_recording and out_video:
                out_video.write(frame)
            time.sleep(1 / fps)
            # 内存管理：保留最近 10 秒的帧
            if len(video_frames) > fps * 10:
                video_frames = video_frames[-fps * 10:]

    thread = threading.Thread(target=capture_loop)
    thread.start()
    return stream, thread

def stop_stream(stream, thread):
    """停止视频和音频流捕获"""
    global stop_thread, video_capture, out_video
    stop_thread = True
    thread.join()
    stream.stop()
    stream.close()
    if video_capture:
        video_capture.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

def start_recording(filename: str, save_option: bool):
    """开始录制视频和音频"""
    global is_recording, out_video
    if is_recording:
        return "已处于录制状态"

    if not save_option:
        logger.info("用户选择不保存视频")
        return "已选择不保存视频，无需录制"

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    is_recording = True
    logger.info(f"开始录制视频: {output_path}")
    return f"录制已开始，保存至 {output_path}"

def stop_recording():
    """停止录制视频和音频"""
    global is_recording, out_video
    if not is_recording:
        return "未处于录制状态"
    
    is_recording = False
    if out_video:
        out_video.release()
        out_video = None
    logger.info("录制已停止")
    return "录制已停止"

def extract_segment(frames: List[np.ndarray], audio_queue: queue.Queue, duration: int = 5) -> Tuple[List[np.ndarray], np.ndarray]:
    """从捕获的视频和音频中提取一段数据"""
    frame_interval = int(30 * duration / len(frames)) if frames else 1
    selected_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames[-150::frame_interval]] if len(frames) > 150 else [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    audio_segment = np.array([])
    while not audio_queue.empty():
        audio_segment = np.append(audio_segment, audio_queue.get().flatten())
    if len(audio_segment) == 0:
        audio_segment = np.zeros(processor.feature_extractor.sampling_rate * duration)

    try:
        audio_segment = librosa.resample(audio_segment, orig_sr=processor.feature_extractor.sampling_rate, target_sr=processor.feature_extractor.sampling_rate)
    except Exception as e:
        logger.error(f"音频重采样失败: {str(e)}")
        audio_segment = np.zeros(processor.feature_extractor.sampling_rate * duration)

    return selected_frames, audio_segment

def analyze_emotions(input_type: str, frames: Optional[List[np.ndarray]] = None, audio_segment: Optional[np.ndarray] = None, text: Optional[str] = None) -> str:
    """分析情绪和性格特征"""
    global emotion_history

    emotions = ["焦虑", "疲惫", "平静", "开心"]
    detected_emotion = "未知"

    if input_type == "video":
        response = counseling_response(input_type, frames=frames, audio_segment=audio_segment, text=None)
    else:  # text
        response = counseling_response(input_type, text=text)

    for emotion in emotions:
        if emotion in response:
            emotion_history[emotion] += 1
            detected_emotion = emotion
            break

    total_emotions = sum(emotion_history.values())
    personality_insight = "需要更多交互数据以分析性格。"
    if total_emotions > 3:
        anxiety_ratio = emotion_history["焦虑"] / total_emotions
        fatigue_ratio = emotion_history["疲惫"] / total_emotions
        if anxiety_ratio > 0.5:
            personality_insight = "你可能倾向于敏感和内向，建议尝试放松技巧。"
        elif fatigue_ratio > 0.5:
            personality_insight = "你可能需要更多休息，性格可能偏向勤奋但易疲劳。"
        else:
            personality_insight = "你的情绪较为平衡，性格可能偏向稳定。"

    return f"检测到情绪: {detected_emotion}\n性格分析: {personality_insight}"

def counseling_response(input_type: str, frames: Optional[List[np.ndarray]] = None, audio_segment: Optional[np.ndarray] = None, text: Optional[str] = None) -> str:
    """处理输入，生成心理咨询回应"""
    global history

    try:
        if input_type == "video":
            if not frames or len(frames) == 0:
                return "未捕获到视频数据"
            content = "请根据我的视频和语音分析情绪并回应。"
            mm_info = {"videos": frames, "audios": [audio_segment]}
            logger.info(f"处理视频片段，帧数: {len(frames)}，音频长度: {len(audio_segment)} 样本")
        elif input_type == "text":
            if not text:
                return "请输入文本内容"
            content = text
            mm_info = None
            logger.info(f"处理文本输入: {content}")
        else:
            return "无效的输入类型"

    except Exception as e:
        logger.error(f"输入处理失败: {str(e)}")
        return f"输入处理错误: {str(e)}"

    history.append({"role": "user", "content": content, "mm_info": mm_info})

    messages = [{"role": "system", "content": sys_prompt}] + history
    try:
        text, mm_inputs = process_mm_info(messages, processor)
    except Exception as e:
        logger.error(f"多模态信息处理失败: {str(e)}")
        return f"多模态处理错误: {str(e)}"

    try:
        inputs = processor(text=text, **mm_inputs if mm_inputs else {}, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if torch.cuda.is_available() else v for k, v in inputs.items()}
    except Exception as e:
        logger.error(f"输入准备失败: {str(e)}")
        return f"输入准备错误: {str(e)}"

    try:
        generate_ids = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2
        )
        response = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )[0]
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        return f"生成错误: {str(e)}"

    history.append({"role": "assistant", "content": response})

    if input_type == "video":
        try:
            speech_output = generate_speech(response, voice_type="Chelsie")
            output_file = os.path.join(TEMP_DIR, f"response_{int(time.time())}.wav")
            with open(output_file, "wb") as f:
                f.write(speech_output)
            logger.info(f"语音输出已保存至 {output_file}")
            return f"{response}\n[语音文件已生成: {output_file}]"
        except Exception as e:
            logger.warning(f"语音生成失败: {str(e)}")
            return response
    return response

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# Qwen 心理辅导助手（支持视频与文本交互）")
    gr.Markdown("通过视频或文本进行心理辅导，支持视频保存和情绪/性格分析（仅限研究用途）。")

    # 对话历史显示
    chat_history = gr.Chatbot(label="对话历史")

    with gr.Row():
        # 视频预览
        video_preview = gr.Video(label="视频预览", source="webcam", interactive=False, visible=True)

        with gr.Column():
            # 输入类型选择
            input_type = gr.Radio(["video", "text"], label="输入类型", value="video")
            # 文本输入
            text_input = gr.Textbox(label="文本输入（仅在文本模式下使用）", placeholder="例如：我最近压力很大", visible=False)
            # 录制控制（仅视频模式）
            with gr.Group(visible=True) as video_controls:
                save_option = gr.Checkbox(label="是否保存视频", value=False)
                filename_input = gr.Textbox(label="录制文件名", value=f"session_{int(time.time())}", placeholder="输入文件名，例如 session1")
                record_btn = gr.Button("开始录制")
                stop_record_btn = gr.Button("停止录制")
            # 开始/停止交互
            start_btn = gr.Button("开始交互")
            stop_btn = gr.Button("停止交互")
            # 文本提交按钮（仅文本模式）
            text_submit_btn = gr.Button("提交文本", visible=False)
            # 输出组件
            output_text = gr.Textbox(label="助手回应", lines=7)
            # 状态显示
            status = gr.Textbox(label="状态", value="未开始")

    # 动态更新界面
    def update_visibility(input_type):
        return (
            gr.update(visible=input_type == "video"),
            gr.update(visible=input_type == "text"),
            gr.update(visible=input_type == "video"),
            gr.update(visible=input_type == "text")
        )

    input_type.change(
        fn=update_visibility,
        inputs=input_type,
        outputs=[video_preview, text_input, video_controls, text_submit_btn]
    )

    # 交互逻辑
    stream = None
    thread = None

    def start_interaction():
        """开始视频和音频流捕获"""
        global stream, thread
        if stream and thread:
            return "交互已开始，无需重复启动", []
        stream, thread = start_stream()
        return "交互已开始，每 5 秒分析一次（视频模式）", []

    def stop_interaction():
        """停止视频和音频流捕获"""
        global stream, thread
        if stream and thread:
            stop_stream(stream, thread)
            stream = None
            thread = None
        return "交互已停止", []

    def start_recording(filename, save_option):
        """开始录制视频"""
        result = start_recording(filename, save_option)
        return result

    def stop_recording():
        """停止录制视频"""
        result = stop_recording()
        return result

    def process_video_interaction(text: Optional[str], chat_history: List[Tuple[str, str]]):
        """处理实时视频和音频流，生成回应并分析心理"""
        global video_frames
        if not video_frames:
            return "未捕获到视频数据，请确保摄像头已开启", chat_history, ""

        frames, audio_segment = extract_segment(video_frames, audio_queue, duration=5)
        response = counseling_response("video", frames=frames, audio_segment=audio_segment, text=text)
        emotion_analysis = analyze_emotions("video", frames=frames, audio_segment=audio_segment)
        combined_response = f"{response}\n\n{emotion_analysis}"
        chat_history.append(("用户 (视频)", combined_response))
        return "处理完成", chat_history, combined_response

    def process_text_interaction(text: str, chat_history: List[Tuple[str, str]]):
        """处理文本输入，生成回应并分析心理"""
        if not text:
            return "请输入文本内容", chat_history, ""
        
        response = counseling_response("text", text=text)
        emotion_analysis = analyze_emotions("text", text=text)
        combined_response = f"{response}\n\n{emotion_analysis}"
        chat_history.append(("用户 (文本)", combined_response))
        return "处理完成", chat_history, combined_response

    # 绑定按钮事件
    start_btn.click(
        fn=start_interaction,
        outputs=[status, chat_history]
    )
    stop_btn.click(
        fn=stop_interaction,
        outputs=[status, chat_history]
    )
    record_btn.click(
        fn=start_recording,
        inputs=[filename_input, save_option],
        outputs=[status]
    )
    stop_record_btn.click(
        fn=stop_recording,
        outputs=[status]
    )
    text_submit_btn.click(
        fn=process_text_interaction,
        inputs=[text_input, chat_history],
        outputs=[status, chat_history, output_text]
    )

    # 定时处理（视频模式，每 5 秒）
    demo.load(
        fn=process_video_interaction,
        inputs=[text_input, chat_history],
        outputs=[status, chat_history, output_text],
        every=5,
        _js="() => document.querySelector('input[name=\"input_type\"]:checked').value === 'video'"
    )

# 启动应用
if __name__ == "__main__":
    logger.info("启动 Gradio 界面...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        logger.info("Gradio 界面已成功启动")
    except Exception as e:
        logger.error(f"Gradio 启动失败: {str(e)}")
        raise
    finally:
        if stream and thread:
            stop_stream(stream, thread)