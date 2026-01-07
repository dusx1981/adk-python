根据代码分析，我梳理出请求大模型的完整调用链。这是一个相当复杂的异步处理流程：

## 主要调用链概述

### 1. **入口点：Agent的 `run_async`**
```
LlmAgent._run_async_impl()
    ↓
self._llm_flow.run_async(ctx)  # 根据配置选择 SingleFlow 或 AutoFlow
    ↓
BaseLlmFlow.run_async()
```

### 2. **主要流程：BaseLlmFlow.run_async()**
```
run_async()
    ↓
_run_one_step_async() [循环直到获得最终响应]
    │
    ├── _preprocess_async()      # 预处理：构建LLM请求
    │     ├── 运行所有 request_processors
    │     ├── 处理工具集合的LLM请求
    │     └── 处理单个工具的LLM请求
    │
    ├── 检查是否需要处理函数调用
    │     ↓
    │     _postprocess_handle_function_calls_async() [如果有未处理的函数调用]
    │
    ├── _call_llm_async()        # 核心：调用大模型
    │     │
    │     ├── _handle_before_model_callback()   # 前置回调
    │     │
    │     ├── 构建LLM请求配置
    │     │
    │     ├── 判断是否支持CFC(Conversational Function Calling)
    │     │     ├── 是: run_live() [实时连接模式]
    │     │     └── 否: 普通异步调用模式
    │     │           ├── increment_llm_call_count() [计数检查]
    │     │           ├── llm.generate_content_async() [关键！调用实际模型]
    │     │           ├── _run_and_handle_error() [错误处理包装器]
    │     │           └── _handle_after_model_callback() [后置回调]
    │     │
    │     └── 如果CFC且live连接，处理实时流
    │
    └── _postprocess_async()     # 后处理：处理模型响应
          ├── _postprocess_run_processors_async()
          ├── _finalize_model_response_event() [构建事件]
          └── _postprocess_handle_function_calls_async() [处理函数调用]
```

### 3. **核心模型调用路径**
```
llm.generate_content_async(request, stream=...)
    ↓
BaseLlm.generate_content_async() [基类方法]
    ↓
具体模型实现（如GeminiLlm、QwenLlm等）
    ↓
实际API调用（HTTP/RPC）
```

### 4. **实时模式 (Live API) 调用链**
```
run_live()
    ↓
llm.connect(llm_request) [建立连接]
    │
    ├── _send_to_model() [发送数据到模型]
    │     ├── 从 LiveRequestQueue 获取请求
    │     ├── 发送音频Blob数据
    │     ├── 发送实时事件（ActivityStart/End）
    │     └── 发送内容
    │
    └── _receive_from_model() [接收模型响应]
          ├── llm_connection.receive() [接收流数据]
          ├── _postprocess_live() [实时后处理]
          └── 处理转录、音频缓存等
```

## 关键类和方法详解

### 1. **请求预处理** (`_preprocess_async`)
```python
# 主要处理逻辑
1. 运行所有 request_processors
2. 处理工具集合：await toolset.process_llm_request()
3. 处理单个工具：await tool.process_llm_request()
4. 转换工具格式：_convert_tool_union_to_tools()
```

### 2. **模型调用** (`_call_llm_async`)
```python
# 关键逻辑
1. 前置回调：可以跳过模型调用直接返回结果
2. 添加标签：将agent名称添加到请求标签中
3. 获取LLM实例：invocation_context.agent.canonical_model
4. 判断模式：
   - CFC模式：使用 run_live() 建立实时连接
   - 非CFC模式：普通异步调用
5. 计数控制：increment_llm_call_count() 检查调用次数限制
6. 流式判断：根据 StreamingMode 选择是否使用流式
```

### 3. **错误处理** (`_run_and_handle_error`)
```python
# 包装器模式处理错误
try:
    async for response in response_generator:
        yield response
except Exception as model_error:
    # 1. 运行插件级别的错误回调
    # 2. 运行agent级别的错误回调 (canonical_on_model_error_callbacks)
    # 3. 如果回调返回响应则使用，否则重新抛出异常
```

### 4. **后处理函数调用**
```python
# 处理模型返回的函数调用
_postprocess_handle_function_calls_async()
    ↓
functions.handle_function_calls_async() [核心处理逻辑]
    ↓
# 执行工具调用
1. 解析函数调用参数
2. 运行 before_tool_callback
3. 执行工具函数
4. 运行 after_tool_callback
5. 处理错误（on_tool_error_callback）
6. 构建函数响应事件
```

## 时序流程示例

```
用户输入
    ↓
Agent.run_async()
    ↓
BaseLlmFlow.run_async()
    ↓
BaseLlmFlow._run_one_step_async()
    │
    ├── 预处理：构建包含历史、工具、配置的请求
    ├── 调用模型：llm.generate_content_async()
    ├── 模型返回：可能包含文本、函数调用等
    ├── 后处理：
    │     ├── 如果是函数调用：执行工具 → 获取结果 → 构建响应
    │     ├── 如果是普通文本：构建最终事件
    │     └── 如果是转移agent：启动新agent
    │
    └── 循环：直到获得最终响应或转移
```

## 关键配置点

1. **流式模式** (`StreamingMode`)
   - `SSE`: Server-Sent Events 流式
   - `NONE`: 非流式
   - `CFC`: 实时对话模式

2. **回调系统** (多层回调)
   ```
   插件回调 → Agent回调 → 默认处理
   ```

3. **工具处理** (复杂的工具链)
   ```
   工具集(Toolset) → 单个工具(Tool) → 函数工具(FunctionTool)
   ```

4. **错误处理链**
   ```
   模型错误 → 插件错误回调 → Agent错误回调 → 抛出异常
   工具错误 → 工具错误回调 → 返回错误结果
   ```

这个架构设计非常灵活，支持插件扩展、多层回调、多种工具类型，并且能够处理实时流式对话和普通异步调用两种模式。对于千问模型的集成，关键是需要实现 `BaseLlm` 接口的 `generate_content_async` 和 `stream_generate_content_async` 方法。