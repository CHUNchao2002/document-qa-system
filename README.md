# 文档智能问答系统

基于 LangChain 和 DeepSeek 的智能文档问答系统，支持多文档上传、语义检索和智能问答。

## 功能特点

- 📚 支持多文档上传和处理
- 🔍 基于向量数据库的语义检索
- 💡 智能问答与上下文理解
- 💾 支持本地向量库持久化
- 🔄 支持加载多个向量库
- 🐛 内置调试模式
- 📝 聊天历史记录

## 安装说明

1. 克隆项目：

```bash
git clone https://github.com/CHUNchao2002/doc-qa-system.git
cd doc-qa-system
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置 API 密钥：

在代码中设置您的 DeepSeek API 密钥：

```python
deepseek_api_key = "your-api-key-here"
```

## 使用方法

1. 启动应用：

```bash
streamlit run doc_chat.py
```

2. 选择操作模式：
   - 上传新文档：上传并处理新的文档文件
   - 加载本地向量库：使用已存在的向量数据库

3. 开始问答：
   - 在输入框中输入您的问题
   - 系统会基于文档内容提供智能回答
   - 可以查看参考文档内容和聊天历史

## 支持的文件格式

- 文本文件 (.txt)
- Markdown 文件 (.md)
- PDF 文件 (.pdf)

## 系统要求

- Python 3.8+
- 足够的磁盘空间用于存储向量数据库
- 稳定的网络连接（用于API调用）

## 调试模式

系统提供调试模式，可以查看：
- 文档处理过程
- 检索详情
- API 调用状态
- 系统运行日志

## 注意事项

1. 请确保有足够的磁盘空间存储向量数据库
2. 大文件处理可能需要较长时间
3. 需要稳定的网络连接以访问 DeepSeek API

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 