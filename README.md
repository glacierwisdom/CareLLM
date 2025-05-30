<div align="center">  
<h1>冰智科技心理健康大语言模型项目</h1>
</div>

<!-- 项目徽标 -->
<p align="center">
  <a href="./LICENSE"><img src="[invalid url, do not cite] alt="License"></a>
  <a href="#"><img src="[invalid url, do not cite] alt="Supported OS"></a>
  <a href="#"><img src="[invalid url, do not cite] alt="Python Version"></a>
  <a href="[invalid url, do not cite] src="[invalid url, do not cite] alt="Contributors"></a>
  <a href="[invalid url, do not cite] src="[invalid url, do not cite] alt="Commit Activity"></a>
  <a href="[invalid url, do not cite] src="[invalid url, do not cite] alt="Open Issues"></a>
  <a href="[invalid url, do not cite] src="[invalid url, do not cite] alt="GitHub Stars"></a>
</p>

<!-- 项目简介 -->
<div align="center">
  <h2>项目简介</h2>
  <p>
    CareLLM 是一个开源项目，基于 <a href="[invalid url, do not cite] 多模态大语言模型，探索心理健康领域的创新应用。Qwen2.5-Omni 支持文本、语音、图像和视频处理，具备实时交互能力。本项目通过交互式界面、心理辅导数据集和微调计划，提升模型在情感支持和情绪分析中的表现，为研究提供强大工具。
  </p>
  <p style="color: red; font-weight: bold;">
    ⚠️ <strong>重要声明</strong>：本项目仅用于研究，未经微调的 Qwen2.5-Omni 模型不适合实际心理辅导。使用需遵守伦理规范和法律法规。
  </p>
</div>

<hr>

<h2>功能</h2>
<ul>
  <li><strong>多模态交互</strong>：支持文本、语音、图像、视频输入，适用于心理辅导场景。</li>
  <li><strong>情感支持</strong>：生成同理心回应，如“听起来你很焦虑，能否分享更多？”。</li>
  <li><strong>情绪分析</strong>：分析用户情绪，提供个性化建议。</li>
  <li><strong>交互界面</strong>：通过命令行、Python API、Gradio Web-UI 或 Hugging Face Spaces 与模型交互。</li>
  <li><strong>开源社区</strong>：欢迎贡献代码、数据集或建议。</li>
</ul>

<hr>

<h2>模型版本</h2>
<h3>版本对比</h3>
<table align="center" border="1" style="width: 90%; border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th>版本</th>
    <th>参数规模</th>
    <th>硬件需求</th>
    <th>特点</th>
    <th>适用场景</th>
  </tr>
  <tr>
    <td><a href="[invalid url, do not cite]
    <td>70 亿</td>
    <td>16GB VRAM (文本, BF16)<br>32GB VRAM (音频)<br>60GB VRAM (60s 视频)</td>
    <td>全功能多模态，最高性能</td>
    <td>心理辅导研究，复杂交互</td>
  </tr>
  <tr>
    <td><a href="[invalid url, do not cite]
    <td>30 亿</td>
    <td>8GB VRAM (文本)<br>16GB VRAM (音频)</td>
    <td>轻量级，快速推理</td>
    <td>低资源设备，快速测试</td>
  </tr>
  <tr>
    <td><a href="[invalid url, do not cite]
    <td>70 亿</td>
    <td>10GB VRAM (优化后)</td>
    <td>MNN 优化，移动设备支持</td>
    <td>移动端心理辅导</td>
  </tr>
  <tr>
    <td><a href="[invalid url, do not cite]
    <td>30 亿</td>
    <td>5GB VRAM (优化后)</td>
    <td>超轻量，低端设备</td>
    <td>嵌入式设备，实时交互</td>
  </tr>
</table>
<p align="center">
  <strong>选择建议</strong>：7B 适合高性能研究，3B/MNN 适合低资源或移动部署。
</p>

<hr>

<h2>心理辅导应用</h2>
<h3>应用场景</h3>
<p>
  Qwen2.5-Omni 的多模态能力使其在心理辅导领域具有显著潜力，以下是具体应用方式：
</p>
<ol>
  <li><strong>文本生成</strong>:
    <ul>
      <li><strong>功能</strong>：生成同理心和支持性的文本回应，增强用户信任。</li>
      <li><strong>用途</strong>：用于文本聊天机器人，帮助用户表达情感并提供初步支持。</li>
      <li><strong>示例</strong>：用户输入“我最近感到很焦虑”，模型生成“听起来你正经历一段艰难的时光，能否分享更多你的感受？”。</li>
      <li><strong>技术</strong>：基于 Transformer 的文本生成，结合上下文和情绪关键词分析。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户通过 Web-UI 或命令行输入文本问题。</li>
          <li>模型分析输入，生成支持性回应。</li>
          <li>提供建议（如正念练习）或引导进一步对话。</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><strong>语音交互</strong>:
    <ul>
      <li><strong>功能</strong>：处理和生成语音输入和输出，增强交互自然性。</li>
      <li><strong>用途</strong>：创建语音助手，用户用语音描述问题，模型以自然语音回应。</li>
      <li><strong>示例</strong>：用户用语音说“我感到压力很大”，模型用语音回答“压力是很常见的，能告诉我是什么让你感到压力大吗？”。</li>
      <li><strong>技术</strong>：基于 Thinker-Talker 架构的语音生成，结合语音特征提取。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户通过 Gradio Web-UI 的麦克风输入语音。</li>
          <li>模型处理语音，生成语音或文本回应。</li>
          <li>提供情绪分析或建议（如放松技巧）。</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><strong>视频分析</strong>:
    <ul>
      <li><strong>功能</strong>：理解视频输入，包括面部表情和肢体语言。</li>
      <li><strong>用途</strong>：在视频咨询中，分析非语言线索以理解情感状态。</li>
      <li><strong>示例</strong>：用户在视频中表达焦虑，模型识别面部表情并回应“从你的表情来看，你似乎很焦虑”。</li>
      <li><strong>技术</strong>：基于 TMRoPE（Time-aligned Multimodal RoPE）同步视频和音频，分析情绪线索。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户通过 Web-UI 上传视频或实时视频流。</li>
          <li>模型提取视觉特征，结合文本/语音生成回应。</li>
          <li>提供情绪报告或个性化建议。</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><strong>多模态整合</strong>:
    <ul>
      <li><strong>功能</strong>：同时处理文本、音频和视频输入，提供综合分析。</li>
      <li><strong>用途</strong>：为用户提供更个性化的支持，结合多种线索理解需求。</li>
      <li><strong>示例</strong>：用户发送语音和视频，模型结合两者生成回应。</li>
      <li><strong>技术</strong>：Thinker-Talker 架构整合多模态输入，生成流式输出。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户通过 Web-UI 输入多模态数据。</li>
          <li>模型分析综合特征，生成个性化回应。</li>
          <li>提供情绪分析或建议，需微调以提升准确性。</li>
        </ul>
      </li>
    </ul>
  </li>
</ol>
<p align="center">
  <strong>注意事项</strong>：
  <ul>
    <li><strong>伦理考虑</strong>：AI 模型不应替代专业心理咨询师，仅作为辅助工具，最终决策需由专业人士做出。</li>
    <li><strong>隐私保护</strong>：处理用户数据时需遵守 GDPR 等隐私法规，确保信息安全。</li>
    <li><strong>模型限制</strong>：未经微调的 Qwen2.5-Omni 可能无法准确理解复杂情感，需结合人类判断。</li>
  </ul>
</p>

<hr>

<h2>硬件要求</h2>
<h3>配置详情</h3>
<table align="center" border="1" style="width: 90%; border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th>任务</th>
    <th>VRAM 需求</th>
    <th>推荐硬件</th>
    <th>优化建议</th>
  </tr>
  <tr>
    <td>文本推理</td>
    <td>16GB (7B), 8GB (3B)</td>
    <td>NVIDIA A100, RTX 3090</td>
    <td>BF16 量化，减小批大小</td>
  </tr>
  <tr>
    <td>音频输出</td>
    <td>32GB (7B), 16GB (3B)</td>
    <td>A100 40GB</td>
    <td>预训练语音模块</td>
  </tr>
  <tr>
    <td>视频 (60s)</td>
    <td>60GB (7B)</td>
    <td>4x A100 80GB</td>
    <td>多 GPU 分片</td>
  </tr>
  <tr>
    <td>边缘设备 (MNN)</td>
    <td>10GB (7B-MNN), 5GB (3B-MNN)</td>
    <td>Snapdragon 8 Elite</td>
    <td>INT8 量化</td>
  </tr>
</table>
<ul>
  <li><strong>存储</strong>：7B 模型 22GB，3B 10GB。</li>
  <li><strong>CPU-only</strong>：需 64GB RAM，16 核，速度慢。</li>
  <li><strong>注意</strong>：预留 1.2 倍 VRAM 余量。</li>
</ul>

<hr>

<h2>安装</h2>
<h3>环境准备</h3>
<ol>
  <li><strong>系统要求</strong>:
    <ul>
      <li>OS：Linux (Ubuntu 20.04+), Windows (WSL2), macOS (12.0+).</li>
      <li>Python：3.8+ (推荐 3.10).</li>
      <li>PyTorch：2.0+ (匹配 CUDA 12.1).</li>
    </ul>
  </li>
  <li><strong>安装步骤</strong>:
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>python -m venv carellm_env</code></pre></td>
        <td>创建虚拟环境</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>source carellm_env/bin/activate</code></pre></td>
        <td>激活环境 (Linux/macOS)</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>carellm_env\Scripts\activate</code></pre></td>
        <td>激活环境 (Windows)</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>pip install torch transformers==4.45.2 accelerate gradio==4.36.1 sounddevice numpy</code></pre></td>
        <td>安装依赖，包括语音支持</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>git clone [invalid url, do not cite]
        <td>下载 7B 模型 (22GB)</td>
      </tr>
      <tr>
        <td>Python</td>
        <td><pre><code>from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Omni-7B'); print('模型加载成功！')</code></pre></td>
        <td>验证模型加载</td>
      </tr>
    </table>
  </li>
</ol>

<hr>

<h2>部署</h2>
<h3>部署方式</h3>
<ol>
  <li><strong>vLLM 部署</strong>:
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>git clone [invalid url, do not cite] -b qwen2_omni_public<br>cd vllm<br>pip install -r requirements-cuda.txt<br>pip install .</code></pre></td>
        <td>安装 vLLM</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2</code></pre></td>
        <td>文本推理</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>vllm serve Qwen/Qwen2.5-Omni-7B --port 8000 --dtype bfloat16 -tp 4</code></pre></td>
        <td>API 服务 (4 GPU)</td>
      </tr>
    </table>
  </li>
  <li><strong>Docker 部署</strong>:
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>docker run --gpus all -it qwenllm/qwen-omni:2.5-cu121 bash</code></pre></td>
        <td>启动容器</td>
      </tr>
    </table>
  </li>
  <li><strong>边缘设备 (MNN)</strong>:
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>pip install MNN</code></pre></td>
        <td>安装 MNN 框架</td>
      </tr>
    </table>
    <p>下载 <a href="[invalid url, do not cite] <a href="[invalid url, do not cite] 文档</a>.</p>
  </li>
</ol>

<hr>

<h2>与模型交互</h2>
<h3>交互方式</h3>
<ol>
  <li><strong>命令行交互</strong>:
    <p>使用 Python 脚本直接交互，适合测试和调试。</p>
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Python</td>
        <td><pre><code>from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")
prompt = "我最近感到很焦虑，怎样才能缓解？"
response = pipe(prompt, max_length=200, do_sample=True, temperature=0.7)
print(response[0]['generated_text'])
</code></pre></td>
        <td>使用 pipeline 快速生成对话</td>
      </tr>
      <tr>
        <td>Python</td>
        <td><pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
inputs = tokenizer("我感到压力很大", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
</code></pre></td>
        <td>低级 API，精细控制生成</td>
      </tr>
    </table>
  </li>
  <li><strong>Web-UI 交互（Gradio）</strong>:
    <p>提供本地 Web 界面，支持文本和语音输入。</p>
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>git clone [invalid url, do not cite] CareLLM<br>python demos/counseling_demo.py</code></pre></td>
        <td>启动 Gradio，访问 [invalid url, do not cite]
      </tr>
      <tr>
        <td>Python</td>
        <td><pre><code>import gradio as gr
from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Omni-7B")
def generate_response(prompt):
    response = pipe(prompt, max_length=200)
    return response[0]['generated_text']
demo = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(label="问题"), gr.Audio(label="语音", source="microphone")],
    outputs=gr.Textbox(label="回应"),
    title="CareLLM 心理辅导",
    description="输入心理健康问题，获取回应（研究用途）"
)
demo.launch()
</code></pre></td>
        <td>自定义 Gradio，支持语音</td>
      </tr>
    </table>
  </li>
  <li><strong>在线交互（Hugging Face Spaces）</strong>:
    <p>云端 Web 界面，适合分享。</p>
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>步骤</td>
        <td>登录 <a href="[invalid url, do not cite] Face</a>，创建 Space（glacierwisdom/CareLLM-Demo）。</td>
        <td>新建 Gradio Space</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>torch
transformers==4.45.2
accelerate
gradio==4.36.1
sounddevice
numpy
</code></pre></td>
        <td>创建 requirements.txt 配置依赖</td>
      </tr>
      <tr>
        <td>步骤</td>
        <td>部署后访问 Space URL（如 [invalid url, do not cite]
        <td>在线交互</td>
      </tr>
    </table>
  </li>
  <li><strong>语音交互配置</strong>:
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>pip install sounddevice numpy</code></pre></td>
        <td>安装语音依赖</td>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>python end2end.py --model Qwen/Qwen2.5-Omni-7B --do-wave --voice-type Chelsie --output-dir output_wav</code></pre></td>
        <td>生成语音输出</td>
      </tr>
    </table>
  </li>
</ol>

<hr>

<h2>数据集</h2>
<h3>核心数据集</h3>
<ul>
  <li><strong>SoulChatCorpus</strong>:
    <ul>
      <li><strong>概述</strong>：230 万样本多轮共情对话，覆盖家庭、婚恋、职场等场景。</li>
      <li><strong>特点</strong>：注入倾听、安慰等共情要素，安全合规。</li>
      <li><strong>资源</strong>： <a href="[invalid url, do not cite]
    </ul>
  </li>
  <li><strong>GoEmotions</strong>:
    <ul>
      <li><strong>概述</strong>：58,009 条 Reddit 评论，27 种情感类别。</li>
      <li><strong>特点</strong>：高质量标注，适合情绪分析。</li>
      <li><strong>资源</strong>： <a href="[invalid url, do not cite] Face</a></li>
    </ul>
  </li>
  <li><strong>EmpatheticDialogues</strong>:
    <ul>
      <li><strong>概述</strong>：25,000+ 条共情对话，基于个人故事。</li>
      <li><strong>特点</strong>：多轮对话，模拟咨询。</li>
      <li><strong>资源</strong>： <a href="[invalid url, do not cite]
    </ul>
  </li>
</ul>

<hr>

<h2>微调计划</h2>
<h3>微调策略</h3>
<ul>
  <li><strong>LoRA</strong>：优化共情回应，降低计算成本。</li>
  <li><strong>全参数</strong>：针对焦虑、抑郁场景。</li>
  <li><strong>多模态</strong>：增强语音、图像处理。</li>
</ul>
<p>详情： <a href="finetune/README.md">finetune/README.md</a></p>

<hr>

<h2>伦理与法律</h2>
<ul>
  <li><strong>研究用途</strong>：仅限学术研究。</li>
  <li><strong>隐私</strong>：勿输入敏感信息。</li>
  <li><strong>许可证</strong>： <a href="./LICENSE">Apache 2.0</a></li>
</ul>

<hr>

<h2>贡献</h2>
<ul>
  <li><a href="./CONTRIBUTING.md">贡献指南</a></li>
  <li><a href="[invalid url, do not cite]
  <li><a href="[invalid url, do not cite] Discord</a></li>
</ul>

<hr>

<h2>联系我们</h2>
<ul>
  <li><strong>GitHub</strong>： <a href="[invalid url, do not cite]
  <li><strong>社区</strong>： <a href="[invalid url, do not cite] Discord</a></li>
</ul>
