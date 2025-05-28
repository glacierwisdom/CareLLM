  
# 冰智科技心理健康大语言模型项目




  
    
  



  
  
  
  
  
  
  



项目简介

  本项目致力于基于 Qwen2.5-Omni 多模态大语言模型，为心理健康领域提供创新解决方案。通过整合心理辅导数据集、开发交互式 Web 界面和未来微调计划，提升模型在情感支持、情绪分析和心理健康对话中的应用效果。



⚠️ 重要声明：本项目基于未经微调的 Qwen2.5-Omni 模型，仅用于研究目的，不建议直接用于实际心理辅导场景。任何使用需遵守伦理规范和相关法律法规。

功能

多模态交互：支持文本、语音、图像和视频输入，适合多样化的心理辅导场景。
情感支持：生成同理心回应，如“听起来你正经历一段艰难时光，能否分享更多？”。
情绪分析：初步分析用户情绪（如焦虑、抑郁），提供个性化建议。
Web 界面：通过 Gradio 或 Hugging Face Spaces 提供交互式对话平台。
开源社区：欢迎贡献数据集、微调脚本或功能建议。

模型版本
Qwen2.5-Omni 提供多个版本，满足不同硬件和场景需求：



版本
参数规模
硬件需求
特点
适用场景



Qwen2.5-Omni-7B
70 亿
16GB VRAM（文本，BF16）32GB VRAM（音频）~60GB VRAM（60秒视频）
全功能，支持多模态
心理辅导研究


Qwen2.5-Omni-3B
30 亿
8GB VRAM（文本）16GB VRAM（音频）
轻量级
低资源设备


Qwen2.5-Omni-7B-MNN
70 亿
~10GB VRAM
移动设备优化
移动端应用


Qwen2.5-Omni-3B-MNN
30 亿
~5GB VRAM
超轻量
嵌入式设备



托管：
Qwen2.5-Omni-7B
Qwen2.5-Omni-7B-MNN


选择建议：7B 适合高性能研究，3B/MNN 适合低资源部署。

目录结构
CareLLM/
├── README.md               # 项目主文档
├── LICENSE                 # Apache 2.0 许可证
├── CONTRIBUTING.md         # 贡献指南
├── assets/                 # 静态资源
│   └── carellm-logo.png    # 项目 Logo
├── datasets/               # 数据集说明
│   └── README.md           # 心理辅导数据集信息
├── finetune/               # 微调计划
│   └── README.md           # LoRA 和全参数微调
├── doc/                    # 文档
│   └── counseling_guide.md # 心理辅导指南
├── demos/                  # 演示脚本
│   └── counseling_demo.py  # Gradio Web 界面

心理辅导中的应用
Qwen2.5-Omni 的多模态能力在心理辅导中有广泛潜力：
应用场景

情感支持：
功能：生成同理心回应 [1].
流程：用户输入问题 → 模型生成回应 → 提供建议（如正念练习）。


情绪分析：
功能：检测情绪状态 [2].
流程：分析文本/语音 → 生成情绪报告 → 个性化建议。


虚拟现实自我对话：
功能：VR 咨询师对话 [3].
流程：部署到 VR 平台 → 用户语音互动 → 引导反思。


服务可访问性：
功能：降低心理健康服务门槛 [4].
流程：部署 Web-UI → 社区访问 → 即时支持。



工作流程

数据收集：使用 SoulChatCorpus 等数据集。
微调：优化同理心和情绪分析。
部署：通过 Web-UI 提供服务。
评估：收集反馈，迭代优化。

硬件要求



任务
VRAM
推荐硬件



文本推理
16GB (7B)
NVIDIA A100


音频输出
32GB (7B)
A100 40GB


视频 (60秒)
60GB (7B)
4x A100 80GB



存储：7B 模型约 22GB。
注意：预留 1.2 倍 VRAM。

安装

创建环境：
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate  # Windows


安装依赖：
pip install torch transformers==4.45.2 accelerate gradio==4.36.1


下载模型：
git clone https://huggingface.co/Qwen/Qwen2.5-Omni-7B



部署
vLLM

安装：
git clone https://github.com/QwenLM/vllm.git -b qwen2_omni_public
cd vllm
pip install -r requirements-cuda.txt
pip install .


推理：
python end2end.py --model=Qwen/Qwen2.5-Omni-7B --prompt=audio-in-video-v2



Docker
docker run --gpus all -it qwenllm/qwen-omni:2.5-cu121 bash

Web-UI 调用
本地（Gradio）

克隆仓库：
git clone https://github.com/glacierwisdom/CareLLM.git
cd CareLLM


运行：
python demos/counseling_demo.py


访问 http://127.0.0.1:7860。



在线（Hugging Face Spaces）

创建 Space：

新建 glacierwisdom/CareLLM-Demo。
上传 demos/counseling_demo.py 和 requirements.txt.


部署：

访问 Space URL。



数据集
💞 i. 共情对话数据集
SoulChatCorpus

内容概述包含 230万清洗后样本 的多轮共情对话：
🔹 单轮长文本：215,813 个问题 → 619,725 个回答
🔹 多轮增强：ChatGPT 生成，人工校对
🔹 主题：家庭、婚恋、职场等 12 类场景


核心创新  
✅ 共情要素：包含倾听、安慰等 7 类表达
✅ 对话控制：8-20 轮引导
✅ 安全合规：过滤敏感内容


学术验证  
📜 EMNLP 2023
BLEU-4 提升 72.8%，共情得分 1.84/2.0


资源  
🔗 ModelScope
💻 GitHub



GoEmotions

内容概述包含 58,009 条 Reddit 评论，涵盖 27 种情感类别：
赞赏、愤怒、悲伤、喜悦等


核心价值  
✅ 情感多样性：27 类 + 中性
✅ 高质量标注：人类注释
✅ 序列长度：最大 30


学术支持  
📜 arXiv


资源  
🔗 GitHub
📂 Hugging Face



EmpatheticDialogues

内容概述包含 25,000+ 条对话，引发共情反应。
核心价值  
✅ 共情丰富：基于个人故事
✅ 多轮对话：模拟真实咨询


资源  
🔗 GitHub



微调计划

LoRA：优化共情回应。
全参数：针对焦虑场景。
多模态：增强语音、图像。

详见 finetune/README.md.
伦理与法律

研究用途：仅限学术研究。
隐私：勿输入敏感信息。
许可证：Apache 2.0.

贡献

CONTRIBUTING.md
Issues
Qwen Discord

联系我们

GitHub：@glacierwisdom
社区：Qwen Discord

致谢

Qwen 团队
Hugging Face
EmoLLM：SmartFlowAI/EmoLLM


