  
冰智科技心理健康大语言模型项目




  
    
  



  
  
  
  
  
  
  



项目简介

  
    CareLLM 是一个开源项目，基于 Qwen2.5-Omni 多模态大语言模型，探索心理健康领域的创新应用。Qwen2.5-Omni 由阿里云 Qwen 团队开发，采用 Thinker-Talker 架构，支持文本、语音、图像和视频的理解与生成，具备实时语音交互和视频问答能力。本项目通过整合心理辅导数据集、开发交互式 Web 界面和计划微调方案，提升模型在情感支持、情绪分析和心理健康对话中的表现，旨在为心理健康研究提供强大工具。
  



  
    ⚠️ 重要声明：本项目基于未经微调的 Qwen2.5-Omni 模型，仅用于研究目的，不建议直接用于实际心理辅导场景。任何使用需遵守伦理规范和相关法律法规。
  


功能

  
    多模态交互：支持文本、语音、图像和视频输入，适应多样化的心理辅导场景，如语音咨询和视频情绪分析。
    情感支持：生成温暖的同理心回应，例如“听起来你正经历一段艰难时光，能否分享更多感受？”。
    情绪分析：通过文本和语音分析用户情绪（如焦虑、抑郁），提供个性化建议。
    交互式 Web 界面：通过 Gradio 或 Hugging Face Spaces 提供用户友好的对话平台，支持实时交互。
    开源社区：欢迎贡献心理辅导数据集、微调脚本或功能建议，共同推进研究。
  


模型版本
版本概览

  Qwen2.5-Omni 提供多个版本，满足从高性能研究到边缘设备部署的需求。以下是详细对比：



  
    版本
    参数规模
    硬件需求
    特点
    适用场景
  
  
    Qwen2.5-Omni-7B
    70 亿
    ~16GB VRAM（文本推理，BF16）~32GB VRAM（音频输出）~60GB VRAM（60秒视频输入）
    全功能，支持文本、语音、图像、视频处理
    心理辅导研究、多模态交互开发
  
  
    Qwen2.5-Omni-3B
    30 亿
    ~8GB VRAM（文本推理）~16GB VRAM（音频输出）
    轻量级，性能稍低，推理速度快
    低资源设备、快速原型测试
  
  
    Qwen2.5-Omni-7B-MNN
    70 亿
    优化后 ~10GB VRAM支持边缘设备（如 Snapdragon 8 Elite）
    MNN 框架优化，适合移动设备
    移动端心理辅导应用
  
  
    Qwen2.5-Omni-3B-MNN
    30 亿
    优化后 ~5GB VRAM
    超轻量，适合低端设备
    嵌入式设备、实时交互
  



  选择建议：7B 版本适合高性能 GPU 研究，3B 适合资源受限环境，MNN 版本适合移动或边缘部署。


心理辅导中的应用
应用场景

  Qwen2.5-Omni 的多模态能力使其在心理辅导领域具有显著潜力，结合心理学 AI 研究，以下是详细场景：
  
    情感支持与同理心回应 [1]:
      
        功能：生成温暖的回应，如“听起来你正经历一段艰难时光，能否分享更多感受？”，增强用户信任。
        技术细节：利用模型的文本生成能力，通过上下文分析用户输入，生成包含倾听、安慰等共情要素的回应。
        工作流程:
          
            用户通过 Web-UI 输入文本或语音（如“我最近感到很焦虑”）。
            模型分析情绪关键词和语调，生成支持性回应。
            提供建议（如深呼吸练习）或引导进一步对话。
          
        
        研究支持：AI 同理心回应可提高用户参与度，适合初步情感支持。
      
    
    情绪分析与个性化支持 [2]:
      
        功能：通过文本、语音或潜在图像输入，检测用户情绪状态（如焦虑、抑郁）。
        技术细节：结合模型的语音特征提取和文本情感分类，分析语调、语速和关键词，生成情绪报告。
        工作流程:
          
            用户通过 Web-UI 输入多模态数据。
            模型提取情绪特征（如语调中的紧张感）。
            生成个性化建议（如放松技巧）或情绪分析报告。
          
        
        研究支持：多模态 AI 可提高情绪检测准确性，需微调以优化分类性能。
      
    
    虚拟现实自我对话 [3]:
      
        功能：结合虚拟现实（VR），用户与虚拟咨询师对话，促进自我反思和情绪调节。
        技术细节：模型生成实时语音回应，集成 VR 平台的 3D 角色动画，模拟真实咨询场景。
        工作流程:
          
            部署模型到 VR 平台（如 Unity）。
            用户通过语音或文本与虚拟角色互动。
            模型引导用户探索情绪和解决方案。
          
        
        研究支持：VR 自我对话可增强认知灵活性和问题解决能力。
      
    
    心理健康服务可访问性 [4]:
      
        功能：为资源匮乏地区提供初步心理支持，降低服务门槛。
        技术细节：通过 Web-UI 或移动端部署，提供多语言、多模态交互。
        工作流程:
          
            部署 Web-UI 到社区平台或移动应用。
            用户通过手机或电脑访问，获取即时支持。
            强调研究用途，需专业监督。
          
        
        研究支持：AI 驱动的咨询可提高心理健康服务的可及性。
      
    
  


实现心理辅导功能的完整流程

  
    数据收集:
      
        使用公开数据集（如 SoulChatCorpus、GoEmotions）。
        收集用户交互数据，需遵守 GDPR 等隐私法规。
      
    
    模型微调:
      
        使用 LoRA 优化同理心回应，减少计算成本。
        全参数微调针对特定场景（如焦虑缓解）。
        示例：基于 SoulChatCorpus 训练，支持 8-20 轮对话。
      
    
    部署与交互:
      
        通过 Gradio Web-UI 或 Hugging Face Spaces 提供服务。
        支持文本、语音输入，未来集成图像/视频。
      
    
    评估与优化:
      
        收集用户反馈，评估回应质量（BLEU、共情得分）。
        持续微调，提升准确性和安全性。
      
    
  



  注意：未经微调的 Qwen2.5-Omni 可能生成不准确或不恰当的回应，需进一步优化以确保专业性。


硬件要求
硬件配置详情

  Qwen2.5-Omni 的多模态任务需要高性能硬件支持。以下是详细要求：



  
    任务
    精度
    VRAM 需求
    推荐硬件
    优化建议
  
  
    文本推理
    BF16
    ~16GB (7B)~8GB (3B)
    NVIDIA A100 16GB, RTX 3090
    启用模型并行，减少批大小
  
  
    音频输出
    BF16
    ~32GB (7B)~16GB (3B)
    NVIDIA A100 40GB
    使用预训练语音模块
  
  
    视频输入 (60秒)
    BF16
    ~60GB (7B)
    4x NVIDIA A100 80GB
    分片推理，多 GPU 集群
  
  
    边缘设备 (MNN)
    INT8
    ~10GB (7B-MNN)~5GB (3B-MNN)
    Snapdragon 8 Elite
    量化模型，优化内存
  



  
    CPU-only 推理：速度极慢，仅适合测试，推荐至少 64GB RAM 和 16 核 CPU。
    存储需求：7B 模型约 22GB，3B 约 10GB，需预留足够磁盘空间。
    性能优化：
      
        使用 BF16 或 INT8 量化降低 VRAM 需求。
        多 GPU 配置可显著提升视频处理性能。
        边缘设备需 MNN 框架支持，优化内存分配。
      
    
    注意：实际 VRAM 消耗可能高出理论值 1.2 倍，建议预留余量。
  



  更多详情见 Hugging Face 模型页面。


安装
环境准备

  以下是安装 Qwen2.5-Omni 的详细步骤，确保环境兼容和模型加载顺畅：



  
    系统要求:
      
        操作系统：Linux（推荐 Ubuntu 20.04+）、Windows（建议 WSL2）、macOS（12.0+）。
        Python：3.8 或更高版本（推荐 3.10）。
        PyTorch：2.0 或更高版本，需匹配 CUDA 版本（如 CUDA 12.1）。
        硬件：至少 16GB GPU 内存（推荐 NVIDIA A100），或 64GB RAM（CPU-only）。
      
    
    创建虚拟环境:
      python -m venv carellm_env
source carellm_env/bin/activate  # Linux/macOS
carellm_env\Scripts\activate  # Windows

    
    安装依赖:
      pip install torch transformers==4.45.2 accelerate gradio==4.36.1 sounddevice numpy

      
        sounddevice 和 numpy：支持语音交互功能。
        版本说明：指定 transformers 4.45.2 以确保兼容性。
      
    
    下载模型权重:
      git clone https://huggingface.co/Qwen/Qwen2.5-Omni-7B

      
        存储：模型约 22GB，建议高速网络和至少 30GB 磁盘空间。
        替代：3B 版本（10GB）适合低资源环境。
      
    
    验证环境:
      python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Omni-7B'); print('模型加载成功！')"

      
        若报错，检查 GPU 驱动、CUDA 版本或 PyTorch 安装。
      
    
  



  参考 Qwen2.5-Omni 安装教程 和 官方文档。


部署
部署方式

  Qwen2.5-Omni 支持多种部署方式，满足本地、云端和边缘设备需求：



  1. 本地部署（vLLM）
  
    vLLM 是一个高效的推理框架，适合高性能 GPU 环境。
  
  
    安装 vLLM:
      git clone https://github.com/QwenLM/vllm.git -b qwen2_omni_public
cd vllm
git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0
pip install setuptools_scm torchdiffeq resampy x_transformers qwen-omni-utils accelerate
pip install -r requirements-cuda.txt
pip install --upgrade setuptools wheel
pip install .
pip install transformers==4.45.2

    
    文本推理（单 GPU）:
      python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only

    
    语音输出（单 GPU）:
      python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --output-dir output_wav

    
    API 服务（多 GPU）:
      vllm serve Qwen/Qwen2.5-Omni-7B --port 8000 --host 127.0.0.1 --dtype bfloat16 -tp 4

      
        -tp 4：启用 4 GPU 并行。
        访问 `http://127.0.0.1:8000` 测试 API。
      
    
    优化建议:
      
        启用 BF16 或 INT8 量化，降低 VRAM 需求。
        调整批大小以平衡速度和内存。
        使用 `--enforce-eager` 避免编译延迟。
      
    
  

  2. Docker 部署
  
    Docker 提供一致的环境，适合云端或本地部署。
  
  
    拉取镜像:
      docker pull qwenllm/qwen-omni:2.5-cu121

    
    运行容器:
      docker run --gpus all --ipc=host --network=host --rm --name qwen2.5-omni -it qwenllm/qwen-omni:2.5-cu121 bash

    
    推理:
      
        在容器内运行 `end2end.py` 或启动 vLLM 服务。
        映射端口（如 8000）以访问 API。
      
    
    优化:
      
        分配足够 GPU 资源（`--gpus all`）。
        使用 `--ipc=host` 提高内存共享效率。
      
    
  

  3. 边缘设备部署（MNN）
  
    MNN 框架优化模型，适合移动设备或低功耗设备。
  
  
    下载模型:
      
        Qwen2.5-Omni-7B-MNN
        Qwen2.5-Omni-3B-MNN
      
    
    安装 MNN:
      pip install MNN

    
    部署:
      
        参考 MNN 文档。
        支持 Snapdragon 8 Elite 等高性能芯片。
      
    
    优化:
      
        使用 INT8 量化，降低内存占用。
        优化推理线程数，适配设备性能。
      
    
  



  参考 vLLM 文档 和 MNN 文档。


Web-UI 调用
交互界面部署

  CareLLM 提供 Web 界面，通过 Gradio 或 Hugging Face Spaces 实现用户交互，支持文本和语音输入。



  1. 本地部署（Gradio）
  
    Gradio 提供本地 Web 界面，适合开发和测试。
  
  
    准备环境:
      
        确保安装 Gradio、Transformers 和语音依赖（见安装步骤）。
        克隆仓库：
          git clone https://github.com/glacierwisdom/CareLLM.git
cd CareLLM

        
      
    
    运行 Gradio 界面:
      python demos/counseling_demo.py

      
        打开浏览器，访问 http://127.0.0.1:7860。
        输入心理健康问题（如“我最近感到很焦虑，怎样才能缓解？”），查看模型回应。
      
    
    自定义 Web-UI:
      
        编辑 demos/counseling_demo.py：
          import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")
def generate_response(prompt):    response = pipe(prompt, max_length=200, do_sample=True, temperature=0.7)    return response[0]['generated_text']
demo = gr.Interface(    fn=generate_response,    inputs=gr.Textbox(label="输入您的问题或感受", placeholder="例如：我感到压力很大，有什么建议？"),    outputs=gr.Textbox(label="模型回应"),    title="CareLLM：Qwen2.5-Omni 心理辅导演示",    description="输入心理健康相关问题，获取模型的回应（仅供研究，未微调）。",    theme="soft")demo.launch()                添加语音交互:                      安装语音依赖：              pip install sounddevice numpy                        修改 counseling_demo.py：              inputs=[gr.Textbox(label="输入问题"), gr.Audio(source="microphone", label="语音输入")],                        配置 vLLM 语音输出（见部署部分）。                                优化:              使用 --share 生成公网 URL（需 Gradio 账户）：          python demos/counseling_demo.py --share                调整 max_length 和 temperature 控制回应长度和多样性。            
  2. 在线部署（Hugging Face Spaces）
  
    Hugging Face Spaces 提供云端 Web-UI，适合分享和演示。
  
  
    创建 Space:
      
        登录 Hugging Face。
        点击“New Space”，命名如 glacierwisdom/CareLLM-Demo。
        选择 Gradio 模板，设置公开访问。
      
    
    上传文件:
      
        上传 demos/counseling_demo.py。
        创建 requirements.txt：
          torch
transformers==4.45.2
accelerate
gradio==4.36.1
sounddevice
numpy

        
      
    
    部署:
      
        Space 自动构建，完成后访问 URL（如 CareLLM Demo）。
        输入问题，体验模型回应。
      
    
    更新 README:
      在线演示：
体验 CareLLM 的多模态心理辅导：CareLLM Demo

    
    优化:
      
        选择高性能 GPU（如 A100）加速推理。
        定期更新 Space 以同步最新代码。
      
    
  


数据集
💞 共情对话数据集


  1. SoulChatCorpus
  
    
  
  
    内容概述:
      
        包含 230 万清洗后样本 的多轮共情对话数据集，专为心理咨询场景设计：
      
      
        单轮长文本：215,813 个心理咨询问题，生成 619,725 个专业回答。
        多轮增强：使用 ChatGPT（99% gpt-3.5）生成对话，人工校对共情表达。
        主题覆盖：12 类心理咨询场景，包括家庭、婚恋、职场、焦虑、抑郁等，过滤敏感话题（如自杀、暴力）。
      
    
    核心创新:
      
        ✅ 共情要素注入：每轮响应包含倾听、安慰、认可、信任等 7 类共情表达。
        ✅ 对话动态控制：8-20 轮渐进式引导，模拟真实心理咨询节奏。
        ✅ 安全合规：人工清洗 105,134 条低质样本，过滤隐私和不当内容。
      
    
    学术验证:
      
        📜 EMNLP 2023 Findings
        性能：BLEU-4 提升约 72.8%，人工共情得分 1.84/2.0。
        首个百万级中文心理共情对话数据集，广泛应用于对话模型微调。
      
    
    使用示例:
      
        加载数据集：
          from datasets import load_dataset
dataset = load_dataset("YIRONGCHEN/SoulChatCorpus")
print(dataset['train'][0])

        
        微调：结合 LoRA，训练模型生成共情回应。
      
    
    资源获取:
      
        ModelScope 数据集
        GitHub 项目
      
    
  

  2. GoEmotions
  
    
  
  
