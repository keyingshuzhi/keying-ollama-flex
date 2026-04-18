# keying-ollama-flex 架构说明

开发者：柯影数智团队
联系方式：1090461393@qq.com

## 设计目标

- 保持对官方 `ollama-python` 的兼容和跟进能力
- 对外 API 稳定，内部实现可持续迭代
- 支持同步/异步、流式、工具调用、结构化输出的统一抽象
- 便于在其他项目直接复用

## 分层结构

1. `ollama_flex.__init__`
- 对外稳定导出层（public API）
- 只暴露开发者常用类型和入口

2. `ollama_flex.client`
- 核心编排层（Facade）
- 负责调用官方 SDK、封装工具循环、结构化重试、模型管理

3. `ollama_flex.session`
- 会话状态层
- 负责历史上下文、窗口裁剪、多轮对话体验

4. `ollama_flex.tools`
- 工具注册与执行层
- 负责 schema 生成、参数解析/矫正、同步与异步执行

5. `ollama_flex.recipes`
- 脚本级工作流层
- 负责把常见演示脚本（下载进度、模型展示、CLI 历史对话）封装成可复用接口

6. `ollama_flex._internal`
- 内部实现细分层（private）
- `messages.py`：消息构建与工具消息序列化
- `extractors.py`：响应解析
- `streaming.py`：流式聚合
- `history.py`：历史裁剪与系统消息保留

## 关键架构策略

- “公开接口不动，内部模块可迭代”：避免业务项目频繁改造适配
- “职责单一”：`client/session/tools` 各司其职，减少重复代码
- “内部能力复用”：响应解析和消息处理只保留一套实现

## 扩展建议

1. 新增 provider 适配器时，优先新增 facade 方法，不破坏现有签名。
2. 需要高阶业务能力时，在 `session` 层添加能力组合，不要在 `_internal` 暴露业务语义。
3. 所有公共行为变更应补充 `tests/` 用例，再更新 README 示例。
