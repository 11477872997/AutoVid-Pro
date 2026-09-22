# AutoVid Pro

中文 / English bilingual documentation

## 中文

AutoVid Pro 是一个面向本地运行的视频与音频翻译、字幕审核和多语言配音工具。它将语音识别、字幕翻译、字幕编辑、声音克隆和背景音合并组织成一个可恢复的项目流程。

### 主要功能

- 支持视频和音频导入，使用 faster-whisper 进行语音识别。
- 第一步直接选择目标语言、翻译风格和翻译后端，并生成首版目标字幕。
- 支持本地 OPUS/NLLB 翻译模型和第三方 OpenAI 兼容中转平台。
- 第二步逐句编辑原文和译文，支持重新翻译全部字幕。
- CosyVoice2 本地声音克隆与多角色配音。
- 保留背景音乐和环境音，支持音量、语速、音高、表达风格控制。
- 支持单句重新生成、项目文件保存、任务状态和断点续作基础能力。
- React + Ant Design 前端，FastAPI 后端，默认运行在 `http://127.0.0.1:7860`。

### 工作流程

1. **识别与首次翻译**：上传素材，选择源语言、目标语言和翻译后端。
2. **字幕审核与重新翻译**：审核识别结果和译文，修改后可重新翻译。
3. **音频克隆与合成**：确认字幕后进行声音克隆、混音和字幕烧录。
4. **导出结果**：导出配音视频、音频和字幕文件。

### Windows 一键启动

要求：Windows 10/11、Python 3.10+、Node.js 18+。使用 NVIDIA GPU 时建议安装兼容 CUDA 的 PyTorch。

首次使用，双击：

```text
一键启动.bat
```

脚本会创建 `.venv`、安装 Python 和前端依赖、下载模型、构建 React 前端并启动后端。之后可双击：

```text
启动.bat
```

每次运行 `启动.bat` 都会先释放 7860 端口并重新执行 `npm run build`，因此会使用最新的前端源码。后端终端必须保持打开，关闭终端即停止程序。

浏览器访问：<http://127.0.0.1:7860>

### 第三方翻译中转平台

选择“第三方中转平台（OpenAI 兼容）”后：

1. 填写中转平台的 `Base URL`。
2. 填写 `API Key`。
3. 点击“拉取模型”。
4. 从返回的模型列表中选择模型。

程序会请求兼容 OpenAI 的 `/models` 和 `/chat/completions` 接口。密钥只在当前进程请求中使用，不写入项目配置文件。

### 模型

默认或可选组件包括：

- faster-whisper：语音识别。
- `Helsinki-NLP/opus-mt-zh-en`：中文到英文的本地翻译示例。
- NLLB-200：可扩展的多语言本地翻译模型。
- `iic/CosyVoice2-0.5B`：声音克隆和跨语言 TTS。
- FFmpeg：音频分离、混音、视频合成和字幕烧录。

模型可通过 `download_models.py` 下载：

```powershell
python download_models.py --source modelscope
```

不同模型和 CosyVoice2 源码可能有单独的许可证和使用限制。发布或部署前请分别阅读上游项目许可证、模型卡和服务条款。

### 开发模式

后端：

```powershell
.venv\Scripts\activate
python server.py
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

生产构建：

```powershell
cd frontend
npm run build
```

### 项目目录

```text
app.py                 核心模型、音频处理和项目逻辑
server.py              FastAPI API 和静态文件服务
frontend/              React + TypeScript + Ant Design 前端
models/                本地模型目录
voices/                声音库目录
outputs/               上传文件、项目和处理结果
vendor/CosyVoice/      CosyVoice2 源码
download_models.py     模型下载脚本
一键启动.bat           首次安装和启动
启动.bat               构建前端并启动后端
```

### 责任与安全说明

- 只克隆本人或已获得明确授权的声音。
- 不要上传包含隐私、机密或未获授权的音频和视频。
- 第三方 API Key 不要提交到 Git 仓库；建议使用环境变量或本地密码管理工具。
- 生成内容应标注为 AI 配音或翻译内容，并遵守当地法律、平台规则和版权要求。
- 本项目按现状提供，不保证所有模型、GPU、驱动和第三方服务组合都能正常工作。

### 贡献

欢迎提交 Issue、改进文档、补充模型适配或提交 Pull Request。提交代码前请先确认没有包含模型权重、API Key、个人音频或生成结果。

### 许可证

本项目代码建议采用 Apache-2.0 或 MIT 许可证。当前仓库如尚未放置正式许可证文件，请在公开发布前选择一种许可证并添加对应的 `LICENSE` 文件。模型、CosyVoice2、FFmpeg 和其他依赖仍分别受其上游许可证约束。

---

## English

AutoVid Pro is a local-first video and audio translation, subtitle review, and multilingual dubbing tool. It combines speech recognition, subtitle translation, subtitle editing, voice cloning, background-audio preservation, and resumable project processing.

### Features

- Import video or audio and transcribe it with faster-whisper.
- Choose the target language, translation style, and translation backend during the first step.
- Use local OPUS/NLLB translation models or an OpenAI-compatible third-party relay.
- Review and edit source and translated subtitles line by line, then retranslate when needed.
- Local CosyVoice2 voice cloning and multi-speaker dubbing.
- Preserve background music and ambience with volume, speed, pitch, and style controls.
- Regenerate individual lines and save project state for resumable work.
- React + Ant Design frontend and FastAPI backend at `http://127.0.0.1:7860`.

### Workflow

1. **Recognition and first translation**: upload media and select languages and backend.
2. **Subtitle review and retranslation**: edit source or target text and retranslate.
3. **Voice cloning and synthesis**: generate voices, mix audio, and optionally burn subtitles.
4. **Export**: download the dubbed video, audio, and subtitle files.

### One-click Windows setup

Requirements: Windows 10/11, Python 3.10+, and Node.js 18+. An NVIDIA GPU with a compatible CUDA PyTorch build is recommended.

For the first run, double-click `一键启动.bat`. It creates `.venv`, installs dependencies, downloads models, builds the React frontend, and starts the backend.

For later runs, double-click `启动.bat`. It releases port 7860, rebuilds the frontend with `npm run build`, and starts the backend. Keep the backend terminal open; closing it stops the application.

Open <http://127.0.0.1:7860> in a browser.

### Third-party translation relays

Select **Third-party relay (OpenAI-compatible)**, enter the relay `Base URL` and `API Key`, click **Fetch models**, and select a returned model. AutoVid Pro uses the compatible `/models` and `/chat/completions` endpoints. The key is used only for requests in the current process and is not written to project configuration files.

### Models and licensing

The project can use faster-whisper, OPUS, NLLB-200, CosyVoice2, and FFmpeg. Download models with:

```powershell
python download_models.py --source modelscope
```

Each upstream model, CosyVoice2 source tree, and dependency may have separate licenses and usage restrictions. Read the upstream license, model card, and service terms before redistribution or deployment.

### Development

```powershell
.venv\Scripts\activate
python server.py
```

```powershell
cd frontend
npm install
npm run dev
```

Build the frontend with `npm run build` inside `frontend/`.

### Responsible use

Only clone voices you own or have explicit permission to use. Do not upload private or confidential media. Never commit API keys, model weights, personal audio, or generated outputs. Disclose AI-generated dubbing where appropriate and follow applicable law, copyright rules, platform policies, and third-party service terms.

### Contributing and license

Issues, documentation improvements, model adapters, and pull requests are welcome. Before publishing, add a formal `LICENSE` file for this repository, such as Apache-2.0 or MIT. Upstream models, CosyVoice2, FFmpeg, and other dependencies remain subject to their own licenses.

