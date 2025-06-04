import gradio as gr
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info, generate_speech
import librosa
import torch
import logging
import numpy as np
import cv2
import sounddevice as sd
from typing import Optional, Dict, List, Tuple
import time
import os
import threading
import queue
from collections import defaultdict

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载模型和处理器
try:
    logger.info("开始加载 Qwen2.5-Omni-7B 模型...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        load_in_8bit=False
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True
    )
    logger.info("模型和处理器加载完成。")
except Exception as e:
    logger.error(f"模型或处理器加载失败: {str(e)}")
    raise

# 系统提示
sys_prompt = (
    "你是一个富有同理心的虚拟心理咨询助手，由 Qwen 团队开发，能够处理文本、语音和视频输入，提供支持性和有益的回应。 "
    "你的目标是倾听用户的问题，理解他们的情绪，分析性格特征，并提供共情回应和个性化建议。保持回应温暖且专业，避免给出医疗建议。"
)

# 对话历史和情绪统计
history: List[Dict] = []
emotion_history = defaultdict(int)  # 统计情绪分布：{emotion: count}

# 存储路径
TEMP_DIR = "temp_files"
OUTPUT_DIR = "recorded_videos"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 全局变量
video_capture = None
audio_data = np.array([])
audio_queue = queue.Queue()
video_frames = []
stop_thread = False
is_recording = False
out_video = None

def audio_callback(indata, frames, time, status):
    """音频流回调函数，捕获麦克风数据"""
    global audio_data, is_recording, out_video
    if status:
        logger.warning("音频捕获状态: %s", status)
    audio_queue.put(indata.copy())
    if is_recording and out_video:
        out_video.write(indata.tobytes())

def start_stream():
    """启动视频和音频流捕获线程"""
    global video_capture, stop_thread, video_frames, audio_data, out_video
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("无法打开摄像头")
        raise Exception("无法打开摄像头")

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None

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
        logger.info("用户选择不保存视频，跳过录制初始化")
        return "已选择不保存视频，无需录制"

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
    output_path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    is_recording = True
    logger.info("开始录制视频: %s", output_path)
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

    audio_segment = librosa.resample(audio_segment, orig_sr=processor.feature_extractor.sampling_rate, target_sr=processor.feature_extractor.sampling_rate)
    return selected_frames, audio_segment

def analyze_psychology(frames: List[np.ndarray], audio_segment: np.ndarray) -> str:
    """简单分析情绪和性格特征，基于历史数据"""
    global emotion_history

    # 模拟情绪分析（基于模型输出或特征提取）
    # 这里假设模型返回的回应中包含情绪关键词
    emotions = ["焦虑", "疲惫", "平静", "开心"]
    sample_response = counseling_response(frames, audio_segment)  # 获取回应
    detected_emotion = "焦虑"  # 简化：假设检测到主要情绪（实际需模型细化）
    for emotion in emotions:
        if emotion in sample_response:
            emotion_history[emotion] += 1
            detected_emotion = emotion
            break

    # 简单性格分析（基于情绪分布）
    total_emotions = sum(emotion_history.values())
    if total_emotions > 5:  # 至少 5 次交互才分析性格
        anxiety_ratio = emotion_history["焦虑"] / total_emotions
        fatigue_ratio = emotion_history["疲惫"] / total_emotions
        if anxiety_ratio > 0.5:
            personality_insight = "你可能倾向于敏感和内向，建议尝试放松技巧。"
        elif fatigue_ratio > 0.5:
            personality_insight = "你可能需要更多休息，性格可能偏向勤奋但易疲劳。"
        else:
            personality_insight = "你的情绪较为平衡，性格可能偏向稳定。"
    else:
        personality_insight = "需要更多交互数据以分析性格。"

    return f"检测到情绪: {detected_emotion}\n性格分析: {personality_insight}"

def counseling_response(frames: List[np.ndarray], audio_segment: np.ndarray, text: Optional[str] = None) -> str:
    """处理视频和音频片段，生成心理咨询回应"""
    global history

    try:
        content = text if text else "请根据我的视频和语音分析情绪并回应。"
        mm_info = {"videos": frames, "audios": [audio_segment]}
        logger.info("处理视频片段，帧数: %d，音频长度: %d 样本", len(frames), len(audio_segment))
    except Exception as e:
        logger.error("输入处理失败: %s", str(e))
        return f"输入处理错误: {str(e)}"

    history.append({"role": "user", "content": content, "mm_info": mm_info})

    messages = [{"role": "system", "content": sys_prompt}] + history
    try:
        text, mm_inputs = process_mm_info(messages, processor)
    except Exception as e:
        logger.error("多模态信息处理失败: %s", str(e))
        return f"多模态处理错误: {str(e)}"

    try:
        inputs = processor(text=text, **mm_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if torch.cuda.is_available() else v for k, v in inputs.items()}
    except Exception as e:
        logger.error("输入准备失败: %s", str(e))
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
        logger.error("生成失败: %s", str(e))
        return f"生成错误: {str(e)}"

    history.append({"role": "assistant", "content": response})

    try:
        speech_output = generate_speech(response, voice_type="Chelsie")
        output_file = os.path.join(TEMP_DIR, f"response_{int(time.time())}.wav")
        with open(output_file, "wb") as f:
            f.write(speech_output)
        logger.info("语音输出已保存至 %s", output_file)
        return f"{response}\n[语音文件已生成: {output_file}]"
    except Exception as e:
        logger.warning("语音生成失败: %s, 仅返回文本", str(e))
        return response

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# CareLLM 心理咨询助手（实时视频交互与分析）")
    gr.Markdown("通过摄像头进行实时心理辅导，模型每 5 秒分析一次，支持视频保存和性格分析（仅限研究用途）。")

    # 对话历史显示
    chat_history = gr.Chatbot(label="对话历史")

    with gr.Row():
        # 视频预览
        video_preview = gr.Video(label="视频预览", source="webcam", interactive=False)

        with gr.Column():
            # 文本输入（可选补充）
            text_input = gr.Textbox(label="补充文本（可选）", placeholder="例如：我最近压力很大")
            # 录制控制
            save_option = gr.Checkbox(label="是否保存视频", value=False)
            filename_input = gr.Textbox(label="录制文件名", value="session", placeholder="输入文件名，例如 session1")
            record_btn = gr.Button("开始录制")
            stop_record_btn = gr.Button("停止录制")
            # 开始/停止交互
            start_btn = gr.Button("开始交互")
            stop_btn = gr.Button("停止交互")
            # 输出组件
            output_text = gr.Textbox(label="助手回应", lines=6)
            # 状态显示
            status = gr.Textbox(label="状态", value="未开始")

    # 交互逻辑
    stream = None
    thread = None

    def start_interaction():
        """开始视频和音频流捕获"""
        global stream, thread
        stream, thread = start_stream()
        return "交互已开始，每 5 秒分析一次", []

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

    def process_interaction(text: Optional[str], chat_history: List[Tuple[str, str]]):
        """处理实时视频和音频流，生成回应并分析心理"""
        global video_frames, audio_data
        if not video_frames:
            return "未捕获到视频数据，请确保摄像头已开启", chat_history, ""

        frames, audio_segment = extract_segment(video_frames, audio_queue, duration=5)
        response = counseling_response(frames, audio_segment, text)
        psychology_analysis = analyze_psychology(frames, audio_segment)
        combined_response = f"{response}\n\n{psychology_analysis}"
        chat_history.append(("用户 (视频)", combined_response))
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

    # 定时处理（每 5 秒）
    demo.load(
        fn=process_interaction,
        inputs=[text_input, chat_history],
        outputs=[status, chat_history, output_text],
        every=5
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
        logger.error("Gradio 启动失败: %s", str(e))
        raise
    finally:
        if stream and thread:
            stop_stream(stream, thread)