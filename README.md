  
冰智科技心理健康大语言模型项目




  
  
  
  
  
  
  



项目简介

  CareLLM 是一个开源项目，基于 Qwen2.5-Omni 多模态大语言模型，探索心理健康领域的创新应用。Qwen2.5-Omni 支持文本、语音、图像和视频处理，具备实时交互能力。本项目通过交互式界面、心理辅导数据集和微调计划，提升模型在情感支持和情绪分析中的表现，为研究提供强大工具。



  
    ⚠️ 重要声明：本项目仅用于研究，未经微调的 Qwen2.5-Omni 模型不适合实际心理辅导。使用需遵守伦理规范和法律法规。
  




功能
- **多模态交互**：支持文本、语音、图像、视频输入，适用于心理辅导场景。
- **情感支持**：生成同理心回应，如“听起来你很焦虑，能否分享更多？”。
- **情绪分析**：分析用户情绪，提供个性化建议。
- **交互界面**：通过命令行、Python API、Gradio Web-UI 或 Hugging Face Spaces 与模型交互。
- **开源社区**：欢迎贡献代码、数据集或建议。



模型版本
版本对比

  
    版本
    参数规模
    硬件需求
    特点
    适用场景
  
  
    Qwen2.5-Omni-7B
    70 亿
    16GB VRAM (文本, BF16)32GB VRAM (音频)60GB VRAM (60s 视频)
    全功能多模态，最高性能
    心理辅导研究，复杂交互
  
  
    Qwen2.5-Omni-3B
    30 亿
    8GB VRAM (文本)16GB VRAM (音频)
    轻量级，快速推理
    低资源设备，快速测试
  
  
    Qwen2.5-Omni-7B-MNN
    70 亿
    10GB VRAM (优化后)
    MNN 优化，移动设备支持
    移动端心理辅导
  
  
    Qwen2.5-Omni-3B-MNN
    30 亿
    5GB VRAM (优化后)
    超轻量，低端设备
    嵌入式设备，实时交互
  



  选择建议：7B 适合高性能研究，3B/MNN 适合低资源或移动部署。




心理辅导应用
应用场景
Qwen2.5-Omni 的多模态能力在心理辅导中具有广泛潜力：


情感支持 [1]:
功能：生成温暖的同理心回应，增强用户信任。
流程：用户输入文本/语音（如“我感到压力很大”），模型分析情绪，生成回应（如“听起来很困难，有什么可以帮助你的？”），并提供建议（如正念练习）。
技术：基于 Transformer 的文本生成，结合上下文和情绪关键词分析。


情绪分析 [2]:
功能：检测焦虑、抑郁等情绪状态。
流程：用户输入多模态数据（文本、语音），模型提取语调、关键词特征，生成情绪报告或个性化建议。
技术：语音特征提取结合情感分类模型。


虚拟现实对话 [3]:
功能：通过 VR 平台与虚拟咨询师互动，促进自我反思。
流程：部署模型到 VR 环境，用户通过语音交互，模型实时生成引导性回应。
技术：实时语音生成与 VR 集成。




  注意：未经微调的模型可能生成不准确回应，需优化以确保安全性。




硬件要求
配置详情

  
    任务
    VRAM 需求
    推荐硬件
    优化建议
  
  
    文本推理
    16GB (7B), 8GB (3B)
    NVIDIA A100, RTX 3090
    BF16 量化，减小批大小
  
  
    音频输出
    32GB (7B), 16GB (3B)
    A100 40GB
    预训练语音模块
  
  
    视频 (60s)
    60GB (7B)
    4x A100 80GB
    多 GPU 分片
  
  
    边缘设备 (MNN)
    10GB (7B-MNN), 5GB (3B-MNN)
    Snapdragon 8 Elite
    INT8 量化
  



存储：7B 模型 22GB，3B 10GB。
CPU-only：需 64GB RAM，16 核，速度慢。
注意：预留 1.2 倍 VRAM 余量。



安装
环境准备
1. **系统要求**:
   - OS：Linux (Ubuntu 20.04+), Windows (WSL2), macOS (12.0+).
   - Python：3.8+ (推荐 3.10).
   - PyTorch：2.0+ (匹配 CUDA 12.1).
2. **安装步骤**:
   
     
       类型
       命令/代码
       说明
     
     
       Bash
       python -m venv carellm_env
       创建虚拟环境
     
     
       Bash
       source carellm_env/bin/activate
       激活环境 (Linux/macOS)
     
     
       Bash
       carellm_env\Scripts\activate
       激活环境 (Windows)
     
     
       Bash
       pip install torch transformers==4.45.2 accelerate gradio==4.36.1 sounddevice numpy
       安装依赖，包括语音支持
     
     
       Bash
       git clone https://huggingface.co/Qwen/Qwen2.5-Omni-7B
       下载 7B 模型 (22GB)
     
     
       Python
       from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Omni-7B'); print('模型加载成功！')
       验证模型加载
     
   



部署
部署方式
1. **vLLM 部署**:
   
     
       类型
       命令/代码
       说明
     
     
       Bash
       git clone https://github.com/QwenLM/vllm.git -b qwen2_omni_publiccd vllmpip install -r requirements-cuda.txtpip install .
       安装 vLLM
     
     
       Bash
       python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2
       文本推理
     
     
       Bash
       vllm serve Qwen/Qwen2.5-Omni-7B --port 8000 --dtype bfloat16 -tp 4
       API 服务 (4 GPU)
     
   
2. **Docker 部署**:
   
     
       类型
       命令/代码
       说明
     
     
       Bash
       docker run --gpus all -it qwenllm/qwen-omni:2.5-cu121 bash
       启动容器
     
   
3. **边缘设备 (MNN)**:
   
     
       类型
       命令/代码
       说明
     
     
       Bash
       pip install MNN
       安装 MNN 框架
     
   
   - 下载 7B-MNN，参考 MNN 文档.



与模型交互
交互方式
1. **命令行交互**：
   - 使用 Python 脚本直接交互，适合测试和调试。
   
     
       类型
       命令/代码
       说明
     
     
       Python
       from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")
prompt = "我最近感到很焦虑，怎样才能缓解？"
response = pipe(prompt, max_length=200, do_sample=True, temperature=0.7)
print(response[0]['generated_text'])

       使用 pipeline 快速生成对话
     
     
       Python
       from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
inputs = tokenizer("我感到压力很大", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))

       低级 API，精细控制生成
     
   


Web-UI 交互（Gradio）：
提供本地 Web 界面，支持文本和语音输入。


  
    类型
    命令/代码
    说明
  
  
    Bash
    git clone https://github.com/glacierwisdom/CareLLM.gitcd CareLLMpython demos/counseling_demo.py
    启动 Gradio，访问 http://127.0.0.1:7860
  
  
    Python
    import gradio as gr

from transformers import pipelinepipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")def generate_response(prompt):    response = pipe(prompt, max_length=200)    return response[0]['generated_text']demo = gr.Interface(    fn=generate_response,    inputs=[gr.Textbox(label="问题"), gr.Audio(label="语音", source="microphone")],    outputs=gr.Textbox(label="回应"),    title="CareLLM 心理辅导",    description="输入心理健康问题，获取回应（研究用途）")demo.launch()       自定义 Gradio，支持语音        

在线交互（Hugging Face Spaces）：
云端 Web 界面，适合分享。


  
    类型
    命令/代码
    说明
  
  
    步骤
    登录 Hugging Face，创建 Space（glacierwisdom/CareLLM-Demo）。
    新建 Gradio Space
  
  
    Bash
    torch

transformers==4.45.2accelerategradio==4.36.1sounddevicenumpy       创建 requirements.txt 配置依赖                 步骤       部署后访问 Space URL（如 https://huggingface.co/spaces/glacierwisdom/CareLLM-Demo）。       在线交互        

语音交互配置：
  
    类型
    命令/代码
    说明
  
  
    Bash
    pip install sounddevice numpy
    安装语音依赖
  
  
    Bash
    python end2end.py --model Qwen/Qwen2.5-Omni-7B --do-wave --voice-type Chelsie --output-dir output_wav
    生成语音输出
  




数据集
核心数据集
- **SoulChatCorpus**：
  - **概述**：230 万样本多轮共情对话，覆盖家庭、婚恋、职场等场景。
  - **特点**：注入倾听、安慰等共情要素，安全合规。
  - **资源**： ModelScope
- **GoEmotions**：
  - **概述**：58,009 条 Reddit 评论，27 种情感类别。
  - **特点**：高质量标注，适合情绪分析。
  - **资源**： Hugging Face
- **EmpatheticDialogues**：
  - **概述**：25,000+ 条共情对话，基于个人故事。
  - **特点**：多轮对话，模拟咨询。
  - **资源**： GitHub



微调计划
微调策略
- **LoRA**：优化共情回应，降低计算成本。
- **全参数**：针对焦虑、抑郁场景。
- **多模态**：增强语音、图像处理。
- 详情： finetune/README.md



伦理与法律
- **研究用途**：仅限学术研究。
- **隐私**：勿输入敏感信息。
- **许可证**： Apache 2.0



贡献
- 贡献指南
- Issues
- Qwen Discord



联系我们
- **GitHub**： @glacierwisdom
- **社区**： Qwen Discord



致谢
- Qwen 团队
- Hugging Face
- EmoLLM: SmartFlowAI/EmoLLM
