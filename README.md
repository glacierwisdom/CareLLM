<div align="center">
  <h1>冰智科技心理健康大语言模型项目</h1>
</div>

<!-- 项目徽标 -->
<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg" alt="Supported OS"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.8+-aff.svg" alt="Python Version"></a>
  <a href="https://github.com/glacierwisdom/CareLLM/graphs/contributors"><img src="https://img.shields.io/github/contributors/glacierwisdom/CareLLM?color=9ea" alt="Contributors"></a>
  <a href="https://github.com/glacierwisdom/CareLLM/commits/main"><img src="https://img.shields.io/github/commit-activity/m/glacierwisdom/CareLLM?color=3af" alt="Commit Activity"></a>
  <a href="https://github.com/glacierwisdom/CareLLM/issues"><img src="https://img.shields.io/github/issues/glacierwisdom/CareLLM?color=9cc" alt="Open Issues"></a>
  <a href="https://github.com/glacierwisdom/CareLLM/stargazers"><img src="https://img.shields.io/github/stars/glacierwisdom/CareLLM?color=ccf" alt="GitHub Stars"></a>
</p>

<!-- 项目简介 -->
<div align="center">
  <h2>项目简介</h2>
  <p>
    CareLLM 是一个开源项目，基于 <a href="https://github.com/QwenLM/Qwen2.5-Omni">Qwen2.5-Omni</a> 多模态大语言模型，探索心理健康领域的创新应用。Qwen2.5-Omni 支持文本、语音、图像和视频处理，具备实时交互能力。本项目通过交互式界面、心理辅导数据集和微调计划，提升模型在情感支持和情绪分析中的表现，为研究提供强大工具。
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
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">Qwen2.5-Omni-7B</a></td>
    <td>70 亿</td>
    <td>16GB VRAM (文本, BF16)<br>32GB VRAM (音频)<br>60GB VRAM (60s 视频)</td>
    <td>全功能多模态，最高性能</td>
    <td>心理辅导研究，复杂交互</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-3B">Qwen2.5-Omni-3B</a></td>
    <td>30 亿</td>
    <td>8GB VRAM (文本)<br>16GB VRAM (音频)</td>
    <td>轻量级，快速推理</td>
    <td>低资源设备，快速测试</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/taobao-mnn/Qwen2.5-Omni-7B-MNN">Qwen2.5-Omni-7B-MNN</a></td>
    <td>70 亿</td>
    <td>10GB VRAM (优化后)</td>
    <td>MNN 优化，移动设备支持</td>
    <td>移动端心理辅导</td>
  </tr>
  <tr>
    <td><a href="https://modelscope.cn/models/MNN/Qwen2.5-Omni-3B-MNN">Qwen2.5-Omni-3B-MNN</a></td>
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
Qwen2.5-Omni 的多模态能力可用于心理咨询场景，提供情感支持和情绪分析，以下是具体用法设计：

<ol>
  <li><strong>情感支持</strong>:
    <ul>
      <li><strong>功能</strong>：通过文本、语音或视频输入，生成温暖的同理心回应，增强用户信任。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户输入：通过文本（如“我最近压力很大，睡不好觉”）、语音或视频表达情感。</li>
          <li>模型分析：结合语调、表情和关键词，识别情绪（如焦虑、压力）。</li>
          <li>生成回应：输出如“听起来你最近过得很辛苦，睡不好觉一定很难受。试试深呼吸或定时的放松时间吧？”，支持文本或语音输出。</li>
        </ul>
      </li>
      <li><strong>技术实现</strong>：
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
text_input = "我最近压力很大，睡不好觉"
inputs = processor(text=text_input, return_tensors="pt", padding=True)
sys_prompt = "你是一个富有同理心的虚拟心理咨询助手，由 Qwen 团队开发，能够处理文本、语音和视频输入，提供支持性和有益的回应。"
inputs["input_ids"] = processor.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": text_input}], add_generation_prompt=True, return_tensors="pt").to("cuda")
generate_ids = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)</code></pre></td>
            <td>处理文本输入并生成回应</td>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>import librosa
from io import BytesIO
import urllib.request
audio_url = "https://example.com/stress_audio.wav"
audio_data, _ = librosa.load(BytesIO(urllib.request.urlopen(audio_url).read()), sr=processor.feature_extractor.sampling_rate)
audio_inputs = processor(text="请分析我的情绪", audios=[audio_data], return_tensors="pt", padding=True)
audio_inputs["input_ids"] = processor.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": "请分析我的情绪"}], add_generation_prompt=True, return_tensors="pt").to("cuda")
generate_ids = model.generate(**audio_inputs, max_length=200)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)</code></pre></td>
            <td>处理语音输入并生成回应</td>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>video_url = "https://example.com/stress_video.mp4"
video_inputs = processor(text="请根据视频分析我的情绪", videos=[video_url], return_tensors="pt", padding=True)
video_inputs["input_ids"] = processor.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": "请根据视频分析我的情绪"}], add_generation_prompt=True, return_tensors="pt").to("cuda")
generate_ids = model.generate(**video_inputs, max_length=200)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)</code></pre></td>
            <td>处理视频输入并生成回应</td>
          </tr>
        </table>
      </li>
    </ul>
  </li>

  <li><strong>情绪分析</strong>:
    <ul>
      <li><strong>功能</strong>：检测用户情绪状态（如焦虑、抑郁），提供个性化建议。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户输入：通过多模态数据（文本、语音、视频）表达情绪。</li>
          <li>模型分析：提取语调、面部表情和关键词，生成情绪报告。</li>
          <li>输出建议：如“你的语调显示出焦虑，试试正念练习或找人倾诉。”</li>
        </ul>
      </li>
      <li><strong>技术实现</strong>：结合 `qwen_omni_utils` 处理多模态输入，参考 [视频信息提取](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/video_information_extracting.ipynb) 和 [音频理解](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb)。</li>
    </ul>
  </li>

  <li><strong>多轮对话支持</strong>:
    <ul>
      <li><strong>功能</strong>：通过多轮对话跟踪情绪变化，提供持续支持。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户输入：如“深呼吸试过了，还是觉得焦虑”。</li>
          <li>模型回应：结合历史上下文，输出“谢谢你告诉我，焦虑的感觉可能需要更多支持。也许可以找个朋友聊聊，或者试试写下你的想法？”</li>
        </ul>
      </li>
      <li><strong>技术实现</strong>：
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>conversation = [
    {"role": "user", "content": "我最近压力很大，睡不好觉"},
    {"role": "assistant", "content": "听起来很困难，能否分享更多？"},
    {"role": "user", "content": "深呼吸试过了，还是觉得焦虑"}
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to("cuda")
generate_ids = model.generate(**inputs, max_length=200)
response = processor.batch_decode(generate_ids[:, inputs.input_ids.size(1):], skip_special_tokens=True)[0]
print(response)</code></pre></td>
            <td>实现多轮对话，参考 [多轮对话](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/multi_round_omni_chatting.ipynb)</td>
          </tr>
        </table>
      </li>
    </ul>
  </li>

  <li><strong>语音交互</strong>:
    <ul>
      <li><strong>功能</strong>：提供实时语音回应，增强用户体验。</li>
      <li><strong>流程</strong>：
        <ul>
          <li>用户输入语音，模型分析并生成语音输出。</li>
          <li>示例：用户说“我很焦虑”，模型以温暖语气回应“别担心，我支持你，试试深呼吸。”</li>
        </ul>
      </li>
      <li><strong>技术实现</strong>：
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>from qwen_omni_utils import generate_speech
speech_output = generate_speech(response, voice_type="Chelsie")
with open("response.wav", "wb") as f:
    f.write(speech_output)</code></pre></td>
            <td>生成语音回应，参考 [语音聊天](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/voice_chatting.ipynb)</td>
          </tr>
        </table>
      </li>
    </ul>
  </li>
</ol>

<p align="center">
  <strong>注意</strong>：未经微调的模型可能生成不准确回应，需优化以确保安全性。建议使用 Gradio 界面部署，参考 [语音聊天](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/voice_chatting.ipynb) 和 [多轮对话](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/multi_round_omni_chatting.ipynb)。
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
        <td><pre><code>git clone https://huggingface.co/Qwen/Qwen2.5-Omni-7B</code></pre></td>
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
        <td><pre><code>git clone https://github.com/QwenLM/vllm.git -b qwen2_omni_public<br>cd vllm<br>pip install -r requirements-cuda.txt<br>pip install .</code></pre></td>
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
    <p>下载 <a href="https://huggingface.co/taobao-mnn/Qwen2.5-Omni-7B-MNN">7B-MNN</a>，参考 <a href="https://github.com/alibaba/MNN">MNN 文档</a>.</p>
  </li>
</ol>

<hr>

<h2>与模型交互</h2>
<h3>交互方式</h3>
<ol>
  <li><strong>命令行交互</strong>：
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
  <li><strong>Web-UI 交互（Gradio）</strong>：
    <p>提供本地 Web 界面，支持文本和语音输入。</p>
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>Bash</td>
        <td><pre><code>git clone https://github.com/glacierwisdom/CareLLM.git<br>cd CareLLM<br>python demos/counseling_demo.py</code></pre></td>
        <td>启动 Gradio，访问 http://127.0.0.1:7860</td>
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
  <li><strong>在线交互（Hugging Face Spaces）</strong>：
    <p>云端 Web 界面，适合分享。</p>
    <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
      <tr style="background-color: #f2f2f2;">
        <th>类型</th>
        <th>命令/代码</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>步骤</td>
        <td>登录 <a href="https://huggingface.co/">Hugging Face</a>，创建 Space（glacierwisdom/CareLLM-Demo）。</td>
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
        <td>部署后访问 Space URL（如 https://huggingface.co/spaces/glacierwisdom/CareLLM-Demo）。</td>
        <td>在线交互</td>
      </tr>
    </table>
  </li>
  <li><strong>语音交互配置</strong>：
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
      <li><strong>概述</strong>：SoulChatCorpus 是一个大规模多轮共情对话数据集，包含 230 万对话样本，覆盖心理健康相关的多轮对话场景（如家庭关系、职场压力、自我认知等），用于提升模型的共情和倾听能力。</li>
      <li><strong>特点</strong>：对话轮数为 8-20 轮，包含倾听、安慰、共情回应等要素，经过严格的安全审查，适合心理辅导模型训练。</li>
      <li><strong>数据集结构</strong>：
        <table border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>字段</th>
            <th>描述</th>
            <th>示例</th>
          </tr>
          <tr>
            <td>dialogue_id</td>
            <td>对话唯一标识</td>
            <td>dialogue_001</td>
          </tr>
          <tr>
            <td>turns</td>
            <td>对话轮次（JSON 格式）</td>
            <td>[{"user": "我最近压力很大", "assistant": "听起来很困难，能否分享更多？"}]</td>
          </tr>
          <tr>
            <td>scene</td>
            <td>对话场景</td>
            <td>职场压力</td>
          </tr>
          <tr>
            <td>emotion</td>
            <td>用户情绪标签</td>
            <td>焦虑</td>
          </tr>
        </table>
      </li>
      <li><strong>资源</strong>： <a href="https://github.com/yirongch/SoulChat">SoulChat on GitHub</a></li>
    </ul>
  </li>
  <li><strong>GoEmotions</strong>:
    <ul>
      <li><strong>概述</strong>：GoEmotions 包含 58,009 条 Reddit 评论，标注了 27 种细粒度情感类别（如“愤怒”、“悲伤”、“开心”等），由 Google Research 团队开发，用于情绪分析任务。</li>
      <li><strong>特点</strong>：高质量人工标注，涵盖多种情绪，支持多标签分类，适用于情感识别模型训练。</li>
      <li><strong>数据集结构</strong>：
        <table border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>字段</th>
            <th>描述</th>
            <th>示例</th>
          </tr>
          <tr>
            <td>text</td>
            <td>评论文本</td>
            <td>"I’m so disappointed with this product!"</td>
          </tr>
          <tr>
            <td>emotions</td>
            <td>情感标签（多标签）</td>
            <td>["disappointment", "sadness"]</td>
          </tr>
          <tr>
            <td>id</td>
            <td>样本唯一标识</td>
            <td>reddit_12345</td>
          </tr>
          <tr>
            <td>subreddit</td>
            <td>来源子版块</td>
            <td>r/tech</td>
          </tr>
        </table>
      </li>
      <li><strong>资源</strong>： <a href="https://hf-mirror.com/datasets/google-research-datasets/go_emotions">Hugging Face</a></li>
    </ul>
  </li>
  <li><strong>EmpatheticDialogues</strong>:
    <ul>
      <li><strong>概述</strong>：EmpatheticDialogues 包含 25,000+ 条共情对话，基于真实个人故事，由 Facebook AI 开发，旨在训练模型生成共情回应，适用于心理咨询场景。</li>
      <li><strong>特点</strong>：每段对话基于用户分享的故事，包含 4-8 轮对话，标注了 32 种情绪类别，适合多轮对话生成任务。</li>
      <li><strong>数据集结构</strong>：
        <table border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>字段</th>
            <th>描述</th>
            <th>示例</th>
          </tr>
          <tr>
            <td>conv_id</td>
            <td>对话唯一标识</td>
            <td>conv_789</td>
          </tr>
          <tr>
            <td>emotion</td>
            <td>对话主导情绪</td>
            <td>anxious</td>
          </tr>
          <tr>
            <td>context</td>
            <td>用户故事背景</td>
            <td>"I just lost my job."</td>
          </tr>
          <tr>
            <td>dialogue</td>
            <td>对话内容（多轮）</td>
            <td>["That must be really tough.", "Yes, I feel so lost."]</td>
          </tr>
        </table>
      </li>
      <li><strong>资源</strong>： <a href="https://github.com/facebookresearch/EmpatheticDialogues">GitHub</a></li>
    </ul>
  </li>
</ul>
<p><strong>使用建议</strong>：SoulChatCorpus 适合多轮共情对话训练，GoEmotions 适合情绪分类任务，EmpatheticDialogues 适合基于故事的共情回应生成。</p>

<hr>

<h2>微调计划</h2>
<h3>使用 SWIFT 进行 LoRA 微调</h3>
<p>以下是使用 SWIFT 框架对 Qwen2.5-Omni 模型进行 LoRA（低秩适配）微调的详细步骤，目标是将模型优化为心理助手，提升其在情感支持和情绪分析中的表现。SWIFT 是一个高效的微调框架，支持多模态模型，适合资源有限的环境。参考 <a href="https://github.com/modelscope/swift">SWIFT Documentation</a>。</p>

<ol>
  <li><strong>环境准备</strong>:
    <ul>
      <li><strong>系统要求</strong>: OS：Linux (Ubuntu 20.04+), Windows (WSL2), macOS (12.0+); Python：3.8+ (推荐 3.10); PyTorch：2.0+ (匹配 CUDA 12.1)。</li>
      <li><strong>安装 SWIFT</strong>:
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>pip install ms-swift</code></pre></td>
            <td>安装 SWIFT</td>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>swift --version</code></pre></td>
            <td>验证安装</td>
          </tr>
        </table>
      </li>
      <li><strong>安装依赖</strong>:
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>pip install torch transformers datasets accelerate peft bitsandbytes</code></pre></td>
            <td>安装 LoRA 微调依赖</td>
          </tr>
        </table>
      </li>
      <li><strong>硬件要求</strong>: 单张 NVIDIA A100 40GB GPU 或 RTX 3090（16GB VRAM），建议预留 1.2 倍 VRAM 余量。</li>
    </ul>
  </li>

  <li><strong>数据集准备</strong>:
    <ul>
      <li><strong>SoulChatCorpus</strong>: 从 <a href="https://github.com/yirongch/SoulChat">SoulChat on GitHub</a> 下载，转换为 JSONL 格式。</li>
      <li><strong>EmpatheticDialogues</strong>: 从 <a href="https://github.com/facebookresearch/EmpatheticDialogues">GitHub</a> 下载，转换为对话格式。</li>
      <li><strong>ConvCounsel</strong>（可选）: 从 <a href="https://huggingface.co/datasets/conv-counsel">Hugging Face</a> 下载，处理语音和文本数据，增强多模态能力。</li>
      <li><strong>下载命令</strong>:
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>git clone https://github.com/yirongch/SoulChat</code></pre></td>
            <td>下载 SoulChatCorpus</td>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>git clone https://github.com/facebookresearch/EmpatheticDialogues</code></pre></td>
            <td>下载 EmpatheticDialogues</td>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>from datasets import load_dataset; dataset = load_dataset("conv-counsel"); dataset.save_to_disk("./conv_counsel")</code></pre></td>
            <td>下载 ConvCounsel</td>
          </tr>
        </table>
      </li>
      <li><strong>格式要求</strong>: SWIFT 接受 JSONL 或 Hugging Face Dataset 格式，每条数据需包含 <code>text</code>（输入）和 <code>label</code>/<code>response</code>（输出）字段。</li>
    </ul>
  </li>

  <li><strong>配置微调参数</strong>:
    <ul>
      <li>使用命令行参数定义 LoRA 微调，或创建配置文件 <code>config.json</code>：</li>
      <li><strong>示例配置文件</strong>：
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>JSON</td>
            <td><pre><code>{
  "model_id": "Qwen/Qwen2.5-Omni-7B",
  "dataset": "path_to_your_dataset",
  "tuner": "lora",
  "lora_rank": 8,
  "lora_alpha": 16,
  "learning_rate": 1e-4,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_train_epochs": 3,
  "output_dir": "./output_lora",
  "fp16": true,
  "evaluation_strategy": "steps",
  "eval_steps": 500,
  "save_steps": 500
}</code></pre></td>
            <td>LoRA 微调配置</td>
          </tr>
        </table>
      </li>
      <li><strong>参数解释</strong>:
        <ul>
          <li><code>lora_rank</code>: 低秩矩阵的秩，推荐值为 8。</li>
          <li><code>lora_alpha</code>: LoRA 缩放因子，建议为 2 倍 rank，即 16。</li>
          <li><code>learning_rate</code>: 学习率，1e-4 适合 LoRA 微调。</li>
          <li><code>gradient_accumulation_steps</code>: 累积梯度步数，优化批大小。</li>
          <li><code>fp16</code>: 启用混合精度训练，减少内存使用。</li>
        </ul>
      </li>
    </ul>
  </li>

  <li><strong>运行微调</strong>:
    <ul>
      <li><strong>命令行运行</strong>:
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>swift sft \
    --model_id Qwen/Qwen2.5-Omni-7B \
    --dataset path_to_your_dataset \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --output_dir ./output_lora \
    --fp16</code></pre></td>
            <td>运行 LoRA 微调</td>
          </tr>
          <tr>
            <td>Bash</td>
            <td><pre><code>swift sft --config config.json</code></pre></td>
            <td>使用配置文件运行</td>
          </tr>
        </table>
      </li>
      <li><strong>预期效果</strong>: 模型在焦虑、抑郁等场景下生成更具共情性的回应，如“听起来你很焦虑，能否告诉我更多？”。</li>
    </ul>
  </li>

  <li><strong>监控与评估</strong>:
    <ul>
      <li>监控训练日志，检查 loss 下降情况。</li>
      <li><strong>评估模型</strong>:
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./output_lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
inputs = tokenizer("我最近感到很焦虑，怎样才能缓解？", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))</code></pre></td>
            <td>测试模型生成效果</td>
          </tr>
        </table>
      </li>
      <li><strong>预期输出</strong>: “听起来你很焦虑，能否告诉我更多？也许试试深呼吸？”</li>
    </ul>
  </li>

  <li><strong>部署与测试</strong>:
    <ul>
      <li>将微调模型部署到 Gradio Web-UI：
        <table align="center" border="1" style="width: 80%; border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>类型</th>
            <th>命令/代码</th>
            <th>说明</th>
          </tr>
          <tr>
            <td>Python</td>
            <td><pre><code>import gradio as gr
from transformers import pipeline
pipe = pipeline("text-generation", model="./output_lora", tokenizer="Qwen/Qwen2.5-Omni-7B")
def generate_response(prompt):
    response = pipe(prompt, max_length=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text']
demo = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="CareLLM 心理助手")
demo.launch()</code></pre></td>
            <td>部署到 Gradio Web-UI</td>
          </tr>
        </table>
      </li>
      <li>测试语音输入需额外集成语音模块（如 <code>sounddevice</code>）。</li>
    </ul>
  </li>

  <li><strong>优化与扩展</strong>:
    <ul>
      <li><strong>优化</strong>: 调整 <code>lora_rank</code>（4-16）或 <code>learning_rate</code>（5e-5 至 2e-4）以平衡性能和资源。</li>
      <li><strong>扩展</strong>: 加入 ConvCounsel 数据，启用多模态微调，参考 <a href="https://github.com/huggingface/trl/tree/main/examples/multimodal">Multimodal Fine-Tuning with TRL</a>。</li>
    </ul>
  </li>
</ol>

<p><strong>注意事项</strong>: 微调前备份模型权重，建议先用 LoRA 进行实验以降低成本。更多 LoRA 微调细节可参考 <a href="https://arxiv.org/abs/2106.09685">LoRA Fine-Tuning Guide</a>。</p>

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
  <li><a href="https://github.com/glacierwisdom/CareLLM/issues">Issues</a></li>
  <li><a href="https://discord.com/invite/qwen">Qwen Discord</a></li>
</ul>

<hr>

<h2>联系我们</h2>
<ul>
  <li><strong>GitHub</strong>： <a href="https://github.com/glacierwisdom">@glacierwisdom</a></li>
  <li><strong>社区</strong>： <a href="https://discord.com/invite/qwen">Qwen Discord</a></li>
</ul>

<hr>

<h2>致谢</h2>
<ul>
  <li>Qwen 团队</li>
  <li>Hugging Face</li>
  <li>EmoLLM: <a href="https://github.com/SmartFlowAI/EmoLLM">SmartFlowAI/EmoLLM</a></li>
</ul>
