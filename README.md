#CareLLM: 基于 Qwen2.5-Omni 的多模态心理辅导探索# 


简介
CareLLM 是一个开源项目，旨在探索 Qwen2.5-Omni 在心理辅导场景中的多模态应用潜力。Qwen2.5-Omni 是阿里云 Qwen 团队开发的多模态大语言模型，基于 Thinker-Talker 架构，支持文本、音频、图像和视频的理解与生成，具备实时语音交互和视频问答能力。本项目通过整合心理辅导数据集、交互式 Web 界面和未来微调计划，展示 Qwen2.5-Omni 在情感支持、情绪分析和心理健康对话中的潜力。

⚠️ 重要声明：本项目基于未经微调的 Qwen2.5-Omni 模型，仅用于研究目的，不建议直接用于实际心理辅导场景。任何使用需遵守伦理规范和相关法律法规。

模型版本
Qwen2.5-Omni 提供多个版本，满足不同硬件和应用需求：



版本
参数规模
硬件需求
特点
适用场景



Qwen2.5-Omni-7B
70 亿
16GB VRAM（文本推理，BF16）32GB VRAM（音频输出）~60GB VRAM（60秒视频输入）
全功能，支持文本、语音、图像、视频
研究、开发心理辅导应用


Qwen2.5-Omni-3B
30 亿
8GB VRAM（文本推理，BF16）16GB VRAM（音频输出）
轻量级，性能稍低
低资源设备、快速实验


Qwen2.5-Omni-7B-MNN
70 亿
优化后 ~10GB VRAM支持边缘设备（如 Snapdragon 8 Elite）
针对移动设备优化，MNN 框架
移动端心理辅导应用


Qwen2.5-Omni-3B-MNN
30 亿
优化后 ~5GB VRAM
超轻量，适合低端设备
嵌入式设备、实时交互



托管地址：
Qwen2.5-Omni-7B/3B：Hugging Face
Qwen2.5-Omni-7B-MNN：Hugging Face
Qwen2.5-Omni-3B-MNN：ModelScope


选择建议：
7B：适合高性能 GPU，研究心理辅导多模态功能。
3B：适合低资源环境，快速测试。
MNN 版：适合移动设备或边缘部署。



功能

多模态交互：支持文本、语音、图像和视频输入，适合多样化的心理辅导场景。
心理辅导对话：生成同理心回应，提供情感支持和心理健康建议。
情绪分析：初步分析用户情绪（如焦虑、抑郁），为个性化支持提供基础。
Web 界面：通过 Gradio 或 Hugging Face Spaces 提供交互式对话平台。
数据集支持：整合公开心理辅导数据集，为微调奠定基础。
开源社区：欢迎贡献数据集、微调脚本或功能建议。

心理辅导中的应用
Qwen2.5-Omni 的多模态能力使其在心理辅导中有广泛应用潜力，结合心理学 AI 研究，以下是具体场景和工作流程：
应用场景

情感支持与同理心回应：
功能：生成温暖、理解的回应，如“听起来你正经历一段艰难的时光，能否分享更多感受？”。
研究支持：AI 可通过同理心回应增强用户信任和参与度 [1].
工作流程：
用户输入文本或语音（如“我感到很焦虑”）。
模型分析情绪，生成支持性回应。
提供建议（如深呼吸练习）或引导进一步对话。




情绪分析与个性化支持：
功能：通过文本或语音分析用户情绪状态（如焦虑、抑郁），提供个性化建议。
研究支持：多模态 AI 可检测情绪线索（如语调、面部表情），提高支持准确性 [2].
工作流程：
用户通过 Web 界面输入多模态数据。
模型提取情绪特征，生成报告或建议。
建议微调以提升情绪分类精度。




虚拟现实自我对话：
功能：结合虚拟现实（VR），用户与虚拟角色对话，促进自我反思。
研究支持：VR 自我对话可增强认知灵活性和问题解决能力 [3].
工作流程：
部署模型到 VR 平台，生成虚拟咨询师。
用户通过语音或文本与虚拟角色互动。
模型引导用户探索情绪和解决方案。




心理健康服务可访问性：
功能：为资源匮乏地区提供初步心理支持。
研究支持：AI 咨询降低心理健康服务门槛 [4].
工作流程：
部署 Web-UI 到社区平台。
用户通过手机或电脑访问，获取即时支持。
强调研究用途，需专业监督。





实现心理辅导功能的工作流程

数据收集：
使用公开数据集（如 Amod/mental_health_counseling_conversations）。
收集用户交互数据（需遵守隐私法规）。


模型微调：
使用 LoRA 或全参数微调，优化同理心回应和情绪分析。
示例：基于 SNAP 数据集训练模型生成支持性对话。


部署 Web-UI：
通过 Gradio 或 Hugging Face Spaces 提供交互界面。
支持文本、语音输入，未来集成视频。


用户交互：
用户通过 Web 界面提问，模型生成回应。
提供情绪反馈或建议（如放松技巧）。


评估与迭代：
收集用户反馈，评估回应质量。
持续微调模型，改进准确性和安全性。




注意：未经微调的 Qwen2.5-Omni 可能生成不准确或不恰当的回应，需进一步优化以确保安全性和专业性。

硬件要求
运行 Qwen2.5-Omni 需要高性能硬件，尤其是多模态任务。以下是详细要求：



任务
精度
VRAM 需求
推荐硬件



文本推理
BF16
16GB (7B)8GB (3B)
NVIDIA A100 16GB, RTX 3090


音频输出
BF16
32GB (7B)16GB (3B)
NVIDIA A100 40GB


视频输入 (60秒)
BF16
~60GB (7B)
4x NVIDIA A100 80GB


边缘设备 (MNN)
INT8
10GB (7B-MNN)5GB (3B-MNN)
Snapdragon 8 Elite



CPU-only 推理：极慢，仅适合测试。
存储：模型文件约 22GB (7B) 或 10GB (3B)，需足够磁盘空间。
多 GPU：推荐 4x GPU 集群以处理视频任务。
注意：实际 VRAM 可能高出理论值 1.2 倍，建议预留余量。

详见 Hugging Face 模型页面。
安装
环境要求

操作系统：Linux、Windows（WSL2 推荐）、macOS。
Python：3.8+。
PyTorch：2.0+。
硬件：至少 16GB GPU 内存。

安装步骤

创建虚拟环境：
python -m venv carellm_env
source carellm_env/bin/activate  # Linux/macOS
carellm_env\Scripts\activate  # Windows


安装依赖：
pip install torch transformers==4.45.2 accelerate gradio==4.36.1


下载模型权重：
git clone https://huggingface.co/Qwen/Qwen2.5-Omni-7B

注意：模型约 22GB，建议高速网络。

验证环境：
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("模型加载成功！")



部署
本地部署（vLLM）

安装 vLLM：
git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
cd vllm
git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0
pip install setuptools_scm torchdiffeq resampy x_transformers qwen-omni-utils accelerate
pip install -r requirements/cuda.txt
pip install --upgrade setuptools wheel
pip install .
pip install transformers==4.52.3


文本推理：
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only


语音输出：
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --output-dir output_wav


API 服务：
vllm serve Qwen/Qwen2.5-Omni-7B --port 8000 --host 127.0.0.1 --dtype bfloat16



Docker 部署
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5-omni -it qwenllm/qwen-omni:2.5-cu121 bash

边缘设备部署
使用 MNN 框架：

下载：Qwen2.5-Omni-7B-MNN
参考：MNN 部署文档

Web-UI 调用
本地 Web-UI（Gradio）

准备环境：

确保安装 Gradio 和 Transformers（见安装步骤）。
克隆仓库：git clone https://github.com/glacierwisdom/CareLLM.git
cd CareLLM




运行 Gradio 界面：
python demos/counseling_demo.py


打开浏览器，访问 http://127.0.0.1:7860。
输入心理健康问题（如“我感到压力很大，有什么建议？”），查看模型回应。
支持文本输入，语音交互需额外配置（见下文）。


自定义 Web-UI：

编辑 demos/counseling_demo.py：import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")

def generate_response(prompt):
    response = pipe(prompt, max_length=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="输入您的问题或感受"),
    outputs=gr.Textbox(label="模型回应"),
    title="CareLLM：Qwen2.5-Omni 心理辅导演示",
    description="输入心理健康相关问题，获取模型的回应（仅供研究，未微调）。",
    theme="soft"
)
demo.launch()




语音交互：

安装语音依赖：pip install sounddevice numpy


配置 vLLM 语音输出（见部署部分）。
修改 counseling_demo.py 添加语音输入/输出。



在线 Web-UI（Hugging Face Spaces）

创建 Space：

登录 Hugging Face。
点击“New Space”，命名如 glacierwisdom/CareLLM-Demo。
选择 Gradio 模板，设置公开。


上传文件：

上传 demos/counseling_demo.py。
创建 requirements.txt：torch
transformers==4.45.2
accelerate
gradio==4.36.1




部署：

Space 自动构建，完成后访问 URL（如 https://huggingface.co/spaces/glacierwisdom/CareLLM-Demo）。
输入问题，体验模型回应。


更新 README：
在线演示：
体验 CareLLM 的多模态心理辅导：[CareLLM Demo](https://huggingface.co/spaces/glacierwisdom/CareLLM-Demo)



数据集
计划使用以下数据集（需遵守许可证）：

SNAP: Counseling Conversation Analysis：1300 万条对话，需申请。
Amod/mental_health_counseling_conversations：Apache 2.0。
Mental Health Conversational Data：需检查许可证。

详见 datasets/README.md。
微调计划

LoRA 微调：优化同理心和情绪分析。
全参数微调：针对焦虑、抑郁等场景。
多模态微调：增强语音、图像处理能力。

详见 finetune/README.md。
伦理与法律

研究用途：仅限学术研究，模型输出未经验证。
数据隐私：勿输入敏感信息。
许可证：遵循 Apache 2.0。

贡献

CONTRIBUTING.md
Issues
社区：Qwen Discord

联系我们

GitHub：@glacierwisdom
Email：通过 Issues 联系。
社区：Qwen Discord

致谢

Qwen 团队：提供 Qwen2.5-Omni。
Hugging Face：模型托管。
EmoLLM：启发于 SmartFlowAI/EmoLLM.

参考文献：

AI-Driven Psychological Consultation
AI in Counseling
VR Self-Talk
AI in Psychotherapy

