# Examples

`ollama_flex` 示例脚本，按场景划分：

- `basic_chat.py`：基础聊天 + 流式事件
- `async_chat.py`：异步聊天 + 异步流式
- `tools_call.py`：同步工具调用循环
- `async_tools_call.py`：异步工具调用循环
- `structured_output.py`：结构化输出（自动重试）
- `session_chat.py`：会话管理与上下文
- `thinking_logprobs.py`：thinking / logprobs
- `stream_generate_events.py`：生成流事件
- `generate_completion.py`：代码补全（prompt + suffix）
- `embed_vectors.py`：嵌入向量
- `multimodal_chat.py`：多模态图片理解
- `model_admin.py`：模型列表/预热/运行状态
- `recipes_showcase.py`：recipes 统一演示
- `history_cli.py`：带历史的交互 CLI
- `web_search_fetch.py`：web_search / web_fetch

运行方式：

```bash
python examples/basic_chat.py
```

