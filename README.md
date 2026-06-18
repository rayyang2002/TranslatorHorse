# Video Translator

面向内容创作者的一站式视频翻译工具：转录、字幕翻译、IndexTTS2 配音、软字幕和硬字幕输出。

## 当前重点

| 能力 | 状态 |
| --- | --- |
| 本地视频/音频输入 | 已有 |
| URL 输入 | 已有 |
| Whisper/faster-whisper 转录 | 已有 |
| OpenAI-compatible API 字幕翻译 | 已有 |
| G20 语言能力矩阵 | 已加入 |
| IndexTTS2 接口配置入口 | 已加入 |
| IndexTTS2 实际服务 | 等用户确定版本后填写接口地址 |
| Edge-TTS 兜底配音 | 已有 |

## G20 语言策略

语言不再写死在界面里，而是由 `video_translator/config/languages.py` 维护能力矩阵。每种语言标记：

- `transcribe`: 转录能力
- `translate`: 翻译能力
- `tts`: IndexTTS2 配音能力
- `g20_tier`: `core` 或 `extended`

查看命令：

```bash
python video_translator/main.py --g20-list
```

## TTS 选择

当前建议先用 `edge-tts` 跑通完整流程。它不需要你先安装 IndexTTS2 服务。

IndexTTS2 先按“接口模式”预留。以后你确定使用哪个 IndexTTS2 版本后，只要那个版本能启动一个本地 HTTP 服务，就可以在软件界面里填写：

- 接口地址，例如 `http://127.0.0.1:7860`
- 接口路径，例如 `/tts`
- 接口密钥，没有就留空
- 参考音频
- 情绪风格
- 字幕时长对齐策略

软件当前支持两种 IndexTTS2 接口返回方式：

- 直接返回 `audio/wav` 音频内容
- 返回 JSON，里面包含 `audio_base64` 或 `audio_path`

界面和流水线已经预留这些参数：

- TTS 引擎：`indextts2`
- 音色模式：默认音色、参考音频克隆、音色库
- 参考音频：用于音色克隆
- 情绪风格：自然、讲解、严肃、温柔、激动、纪录片
- 字幕时长对齐：严格匹配字幕时长、自然轻微拉伸、按段落处理

实际接口适配位置：

```text
video_translator/core/tts_indextts2.py
```

等本机 IndexTTS2 版本确定后，通常只需要确认它的接口路径和返回格式。

## 桌面应用

```bash
python app.py
```

## 本地配置

真实运行配置保存在本机：

```text
video_translator/config/settings.json
```

该文件包含 API Key 等本地信息，已被 `.gitignore` 排除，不应提交到代码仓库。
首次克隆后可以复制模板：

```bash
copy video_translator\config\settings.example.json video_translator\config\settings.json
```

然后在软件界面或 `settings.json` 中填写自己的模型服务配置。

## 命令行示例

```bash
python video_translator/main.py test_data/test_video.mp4 --api-key sk-xxx
python video_translator/main.py test_data/test_video.mp4 --api-key sk-xxx --tts --tts-engine edge-tts
python video_translator/main.py test_data/test_video.mp4 --api-key sk-xxx --tts --tts-engine indextts2 --indextts2-url http://127.0.0.1:7860
python video_translator/main.py test_data/test_video.mp4 --api-key sk-xxx --subtitle-mode soft
```
