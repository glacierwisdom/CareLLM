import gradio as gr
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info, generate_speech
import librosa
from io import BytesIO
import urllib.request
import torch
import logging
import os
from typing import Optional, Dict, List

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载模型和处理器
try:
    logger.info("开始加载 Qwen2.5-Omni-7B 模型...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float16,  # 使用 FP16 优化内存使用
        device_map="auto",  # 自动分配到可用 GPU/CPU
        offload_folder="offload",  # 离线存储优化
        load_in_8bit=False  # 可选：启用 8-bit 量化以进一步减少内存
    )
    logger.info("模型加载完成。")
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True  # 允许加载远程代码
    )
    logger.info("处理器加载完成。")
except Exception as e:
    logger.error(f"模型或处理器加载失败: {str(e)}")
    raise

# 系统提示，定义助手角色和目标
sys_prompt = (
    "你是一个富有同理心的虚拟心理咨询助手，由 Qwen 团队开发，能够处理文本、语音和视频输入，提供支持性和有益的回应。 "
    "你的目标是倾听用户的问题，理解他们的情绪，并提供共情回应和个性化建议。保持回应温暖且专业，避免给出医疗建议。"
)

# 对话历史，用于多轮对话上下文
history: List[Dict] = []

def counseling_response(input_type: str, text: Optional[str] = None, audio: Optional[str] = None, video: Optional[str] = None) -> str:
    """
    处理用户输入（文本、语音或视频），生成心理咨询回应。
    
    Args:
        input_type (str): 输入类型 ("text", "voice", "video")
        text (str, optional): 文本输入
        audio (str, optional): 音频文件路径
        video (str, optional): 视频文件路径
    
    Returns:
        str: 生成的心理咨询回应
    
    Raises:
        ValueError: 如果输入类型无效或数据缺失
    """
    global history
    
    # 输入验证
    if not input_type or input_type not in ["text", "voice", "video"]:
        logger.error("无效的输入类型: %s", input_type)
        return "无效的输入类型，请选择 'text'、'voice' 或 'video'。"
    
    if input_type == "text" and not text:
        logger.error("文本输入为空")
        return "请输入您的文本问题。"
    elif input_type == "voice" and not audio:
        logger.error("未提供音频输入")
        return "请上传或录制音频。"
    elif input_type == "video" and not video:
        logger.error("未提供视频输入")
        return "请上传视频文件。"
    
    # 根据输入类型准备内容和多模态数据
    try:
        if input_type == "text":
            content = text.strip() if text else ""
            mm_info = None
            logger.info("处理文本输入: %s", content)
        elif input_type == "voice":
            content = "请根据我的语音消息回应，并分析我的情绪。"
            audio_data, sample_rate = librosa.load(audio, sr=processor.feature_extractor.sampling_rate)
            if sample_rate != processor.feature_extractor.sampling_rate:
                logger.warning("音频采样率不匹配，已调整为 %d Hz", processor.feature_extractor.sampling_rate)
            mm_info = {"audios": [audio_data]}
            logger.info("处理音频输入，长度: %d 样本", len(audio_data))
        elif input_type == "video":
            content = "请根据我的视频分析情绪并回应。"
            mm_info = {"videos": [video]}
            logger.info("处理视频输入，文件: %s", video)
        else:
            raise ValueError("未知输入类型")
    except Exception as e:
        logger.error("输入处理失败: %s", str(e))
        return f"输入处理错误: {str(e)}"

    # 添加用户输入到对话历史
    history.append({"role": "user", "content": content, "mm_info": mm_info})
    logger.debug("对话历史更新: %s", history)

    # 准备消息列表，包含系统提示和历史
    messages = [{"role": "system", "content": sys_prompt}] + history
    try:
        text, mm_inputs = process_mm_info(messages, processor)
        logger.info("多模态信息处理完成")
    except Exception as e:
        logger.error("多模态信息处理失败: %s", str(e))
        return f"多模态处理错误: {str(e)}"

    # 生成模型输入
    try:
        inputs = processor(text=text, **mm_inputs, return_tensors="pt", padding=True)
        # 将输入移动到 GPU（如果可用）
        inputs = {k: v.to("cuda") if torch.cuda.is_available() else v for k, v in inputs.items()}
        logger.info("模型输入准备完成，输入张量形状: %s", {k: v.shape for k, v in inputs.items()})
    except Exception as e:
        logger.error("输入准备失败: %s", str(e))
        return f"输入准备错误: {str(e)}"

    # 生成回应
    try:
        generate_ids = model.generate(
            **inputs,
            max_length=200,  # 限制输出长度
            do_sample=True,  # 启用采样以增加多样性
            temperature=0.7,  # 控制回应随机性
            top_p=0.9,  # 核采样阈值
            top_k=50,  # 限制最高概率的 token 数量
            repetition_penalty=1.2  # 减少重复
        )
        response = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )[0]
        logger.info("生成回应: %s", response)
    except Exception as e:
        logger.error("生成失败: %s", str(e))
        return f"生成错误: {str(e)}"

    # 添加助手回应到历史
    history.append({"role": "assistant", "content": response})
    logger.debug("助手回应添加到历史: %s", history)

    # 可选：生成语音输出
    if input_type == "voice":
        try:
            speech_output = generate_speech(response, voice_type="Chelsie")
            output_file = "response.wav"
            with open(output_file, "wb") as f:
                f.write(speech_output)
            logger.info("语音输出已保存至 %s", output_file)
            return f"{response}\n[语音文件已生成: {output_file}]"
        except Exception as e:
            logger.warning("语音生成失败: %s, 仅返回文本", str(e))
            return response
    return response

# Gradio 界面
demo = gr.Interface(
    fn=counseling_response,
    inputs=[
        gr.Radio(["text", "voice", "video"], label="输入类型", value="text"),  # 默认选择文本
        gr.Textbox(label="文本输入", placeholder="例如：我最近压力很大，睡不好觉", visible=True),
        gr.Audio(label="语音输入", type="filepath", visible=False),
        gr.Video(label="视频输入", visible=False)
    ],
    outputs=gr.Textbox(label="助手回应", lines=5),  # 增加文本框高度
    title="CareLLM 心理咨询助手",
    description="选择输入类型（文本、语音或视频），表达你的情感或问题，获取共情支持和建议（仅限研究用途）。",
    theme="huggingface",  # 使用 Hugging Face 主题
    allow_flagging="never"  # 禁用标记功能
)

# 根据输入类型更新输入框可见性
def update_visibility(input_type: str):
    """根据选择更新输入框可见性"""
    return (
        gr.update(visible=input_type == "text"),
        gr.update(visible=input_type == "voice"),
        gr.update(visible=input_type == "video")
    )

# 绑定事件，动态更新界面
demo.inputs[0].change(
    fn=update_visibility,
    inputs=demo.inputs[0],
    outputs=[demo.inputs[1], demo.inputs[2], demo.inputs[3]]
)

# 启动 Gradio 应用
if __name__ == "__main__":
    try:
        logger.info("启动 Gradio 界面...")
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7860,  # 默认端口
            share=True,  # 生成公共链接
            debug=True  # 启用调试模式
        )
        logger.info("Gradio 界面已成功启动")
    except Exception as e:
        logger.error("Gradio 启动失败: %s", str(e))
        raise