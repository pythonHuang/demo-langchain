# 基于langChain 陪聊机器人使用指南

欢迎使用陪聊机器人！这是一个简单易用的对话系统，旨在通过纯文本对话界面为您提供有趣和互动的聊天体验。下面是一些基本步骤，帮助您轻松开始使用。

## 核心功能：

1. **纯文本对话界面**：提供一个简单易用的文本界面，适合进行多轮对话。
2. **个性化交流**：在开始聊天前，机器人会先询问您的称呼和您想聊的话题，以便进行更个性化的对话。
3. **主动发起话题**：机器人会根据您选择的话题首先发言，并主动询问您的看法。
4. **持续互动**：在整个对话过程中，机器人会持续与您交流，并经常询问您的看法，确保对话的连贯性和互动性。
   当然，我可以帮您优化这个使用指南。以下是经过改进的版本：

---

## 使用指南

### 设置环境变量

set PYTHONIOENCODING=utf-8

set PYTHONIUTF8=1

为了顺利运行此程序，您需要设置环境变量。请按照以下步骤操作：

#### 第一步：创建环境变量文件

1. 复制 `.env.example` 文件，创建一个新的 `.env` 文件。
   在终端中，您可以使用以下命令来实现这一操作：

```bash
cp .env.example .env
```

2. 配置 OpenAI 密钥
   在 `.env` 文件中配置您的 OpenAI 密钥。

- 您可以在 [AGI 课堂手册](https://a.agiclass.ai) 找到 OpenAI API 密钥获取方式。
- 在文件中找到 `OPENAI_API_KEY`，并将其值替换为您的密钥。

例如：

```
OPENAI_API_KEY='您的OpenAI密钥'
```

### 第二步：安装必要的软件包

在开始使用前，您需要安装一些必要的软件包。这可以通过以下步骤实现：

1. 打开您的命令行界面（例如：Windows的命令提示符，macOS的终端）。
2. 输入以下命令，以安装所需的软件包：

   ```
   pip install -r requirements.txt
   ```

   这条命令将会自动下载并安装所有必需的依赖包。

### 第三步：运行项目

完成软件包的安装后，您可以开始运行陪聊机器人项目。请按照以下步骤操作：

1. 在命令行界面中，输入以下命令来启动项目：

   ```
   python main.py
   ```

   这条命令会启动服务器，并使陪聊机器人开始运行。

### 第四步：通过浏览器访问

1. 打开您的网络浏览器（例如：Chrome, Firefox, Safari）。
2. 在浏览器的地址栏中输入以下地址并回车：

   ```
   http://127.0.0.1:7801/
   ```
   通过这个地址，您可以访问陪聊机器人的网页界面，并开始与机器人聊天。
