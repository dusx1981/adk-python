# ADK LLM Flow 流程详解 - SingleFlow & AutoFlow

## 概述

ADK (Agent Development Kit) 中，**Flow** 是 LlmAgent 执行 LLM 调用的核心机制。它负责：
- 构建 LLM 请求（Request）
- 调用 LLM 模型
- 处理 LLM 响应（Response）
- 处理函数调用（Function Calling）
- 支持 Agent 间转移（Agent Transfer）

本文档详细说明 `SingleFlow` 和 `AutoFlow` 的架构和调用逻辑。

## 类继承关系

```
BaseLlmFlow (抽象基类)
│
├── SingleFlow (单流程)
│   └── 处理工具调用，无子 Agent 转移
│
└── AutoFlow (自动流程)
    └── 继承 SingleFlow + 添加 Agent 转移能力
```

## 核心类详解

### 1. BaseLlmFlow (基类)

**文件位置**: `src/google/adk/flows/llm_flows/base_llm_flow.py`

**职责**: 所有 LLM Flow 的抽象基类，定义了完整的 LLM 调用生命周期

**核心属性**:
```python
class BaseLlmFlow:
    request_processors: list[BaseLlmRequestProcessor]   # 请求处理器链
    response_processors: list[BaseLlmResponseProcessor] # 响应处理器链
    audio_cache_manager: AudioCacheManager             # 音频缓存管理（用于实时模式）
```

**核心方法**:

#### run_async - 文本对话主入口
```python
async def run_async(self, invocation_context: InvocationContext) -> AsyncGenerator[Event, None]:
    """主循环：持续执行直到获得最终响应"""
    while True:
        last_event = None
        async for event in self._run_one_step_async(invocation_context):
            last_event = event
            yield event
        # 结束条件：最终响应 或 部分响应
        if not last_event or last_event.is_final_response() or last_event.partial:
            break
```

#### run_live - 实时音视频对话主入口
```python
async def run_live(self, invocation_context: InvocationContext) -> AsyncGenerator[Event, None]:
    """使用 Gemini Live API 进行实时音视频对话"""
    # 1. 预处理
    # 2. 建立 WebSocket 连接
    # 3. 并行执行发送和接收任务
    # 4. 处理 Agent 转移和任务完成信号
```

#### _run_one_step_async - 单步执行
```python
async def _run_one_step_async(self, invocation_context: InvocationContext) -> AsyncGenerator[Event, None]:
    """单步 = 一次 LLM 调用"""
    # 1. 创建 LLM 请求
    llm_request = LlmRequest()
    
    # 2. 预处理（请求处理器链）
    async for event in self._preprocess_async(invocation_context, llm_request):
        yield event
    
    # 3. 处理可恢复执行（长轮询工具恢复）
    if should_resume:
        async for event in self._postprocess_handle_function_calls_async(...):
            yield event
        return
    
    # 4. 调用 LLM
    async for llm_response in self._call_llm_async(invocation_context, llm_request, model_response_event):
        # 5. 后处理（响应处理器链）
        async for event in self._postprocess_async(...):
            yield event
```

#### _call_llm_async - 调用 LLM
```python
async def _call_llm_async(self, invocation_context, llm_request, model_response_event):
    # 1. 执行 before_model_callback
    if response := await self._handle_before_model_callback(...):
        yield response
        return
    
    # 2. 获取 LLM 实例
    llm = self.__get_llm(invocation_context)
    
    # 3. 根据配置选择调用方式
    if run_config.support_cfc:  # 支持 CFC（实时模式）
        async for response in self.run_live(invocation_context):
            yield response
    else:  # 普通模式
        async for response in llm.generate_content_async(llm_request, stream=...):
            # 4. 执行 after_model_callback
            if altered := await self._handle_after_model_callback(...):
                yield altered
            else:
                yield response
```

---

### 2. SingleFlow (单流程)

**文件位置**: `src/google/adk/flows/llm_flows/single_flow.py`

**继承**: `BaseLlmFlow`

**职责**: 仅处理当前 Agent 自身及其工具，**不支持子 Agent 转移**

**特点**: 
- 适用于单一 Agent 场景
- 可以调用工具，但不能转移给其他 Agent

**请求处理器链** (按顺序执行):
```python
request_processors = [
    basic.request_processor,              # 基础配置处理
    auth_preprocessor.request_processor,  # 认证预处理
    request_confirmation.request_processor,  # 请求确认处理
    instructions.request_processor,       # 指令处理
    identity.request_processor,           # Agent 身份处理
    contents.request_processor,           # 内容处理（事件历史）
    context_cache_processor.request_processor,  # 上下文缓存
    interactions_processor.request_processor,   # 交互状态
    _nl_planning.request_processor,       # 自然语言规划
    _code_execution.request_processor,    # 代码执行
    _output_schema_processor.request_processor,  # 输出模式
]
```

**响应处理器链**:
```python
response_processors = [
    _nl_planning.response_processor,      # NL 规划后处理
    _code_execution.response_processor,   # 代码执行后处理
]
```

---

### 3. AutoFlow (自动流程)

**文件位置**: `src/google/adk/flows/llm_flows/auto_flow.py`

**继承**: `SingleFlow`

**职责**: 在 SingleFlow 基础上增加 **Agent 转移（Agent Transfer）** 能力

**特点**:
- 支持父 Agent → 子 Agent 转移
- 支持子 Agent → 父 Agent 转移
- 支持子 Agent → 同级 Agent 转移（需满足条件）

**转移方向**:
1. **Parent → Sub-agent**: 父 Agent 可以将任务转移给子 Agent
2. **Sub-agent → Parent**: 子 Agent 可以将任务转移回父 Agent
3. **Sub-agent → Peer**: 同级 Agent 间转移（需父 Agent 也是 LlmAgent，且 `disallow_transfer_to_peers=False`）

**请求处理器链** (在 SingleFlow 基础上添加):
```python
request_processors = SingleFlow.request_processors + [
    agent_transfer.request_processor,  # 添加 Agent 转移处理器
]
```

---

## 处理器体系 (Processor)

### BaseLlmRequestProcessor (请求处理器基类)

```python
class BaseLlmRequestProcessor(ABC):
    @abstractmethod
    async def run_async(
        self, 
        invocation_context: InvocationContext, 
        llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        """处理 LLM 请求，可以生成事件"""
```

### BaseLlmResponseProcessor (响应处理器基类)

```python
class BaseLlmResponseProcessor(ABC):
    @abstractmethod
    async def run_async(
        self, 
        invocation_context: InvocationContext, 
        llm_response: LlmResponse
    ) -> AsyncGenerator[Event, None]:
        """处理 LLM 响应，可以生成事件"""
```

---

## 核心处理器详解

### 1. Agent Transfer Processor (AutoFlow 专属)

**文件位置**: `src/google/adk/flows/llm_flows/agent_transfer.py`

**职责**: 为 LLM 请求添加 Agent 转移能力

**处理逻辑**:
```python
async def run_async(self, invocation_context, llm_request):
    # 1. 获取可转移目标
    transfer_targets = _get_transfer_targets(invocation_context.agent)
    
    # 2. 创建转移工具
    transfer_tool = TransferToAgentTool(agent_names=[...])
    
    # 3. 添加转移指令到系统提示
    llm_request.append_instructions([_build_target_agents_instructions(...)])
    
    # 4. 将工具添加到请求
    await transfer_tool.process_llm_request(tool_context, llm_request)
```

**可转移目标计算**:
```python
def _get_transfer_targets(agent: LlmAgent) -> list[BaseAgent]:
    result = []
    # 1. 所有子 Agent
    result.extend(agent.sub_agents)
    
    # 2. 父 Agent（如果不禁止）
    if not agent.disallow_transfer_to_parent:
        result.append(agent.parent_agent)
    
    # 3. 同级 Agent（如果不禁止且父是 LlmAgent）
    if not agent.disallow_transfer_to_peers:
        result.extend([
            peer for peer in agent.parent_agent.sub_agents
            if peer.name != agent.name
        ])
    
    return result
```

---

### 2. Function Calling 处理器

**文件位置**: `src/google/adk/flows/llm_flows/functions.py`

**核心功能**:

#### handle_function_calls_async (异步处理函数调用)
```python
async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,           # 包含函数调用的事件
    tools_dict: dict[str, BaseTool],      # 可用工具字典
) -> Optional[Event]:
    """调用工具并返回函数响应事件"""
    # 1. 提取函数调用
    function_calls = function_call_event.get_function_calls()
    
    # 2. 并行执行所有工具调用
    tool_tasks = []
    for function_call in function_calls:
        tool = tools_dict[function_call.name]
        task = tool.run_async(args=function_call.args, tool_context=...)
        tool_tasks.append(task)
    
    # 3. 收集结果
    results = await asyncio.gather(*tool_tasks)
    
    # 4. 构建函数响应事件
    return Event(
        content=types.Content(parts=[types.Part(function_response=...)])
    )
```

#### handle_function_calls_live (实时模式处理)
```python
async def handle_function_calls_live(invocation_context, function_call_event, tools_dict):
    # 实时模式下直接调用工具
    # 不等待，直接返回函数响应事件
```

---

## 调用流程详解

### 标准执行流程 (run_async)

```
LlmAgent.run_async
    │
    ▼
BaseLlmFlow.run_async
    │
    ├──▶ While 循环（直到最终响应）
    │       │
    │       ▼
    │   _run_one_step_async
    │       │
    │       ├──▶ _preprocess_async (请求预处理)
    │       │       │
    │       │       ├──▶ basic.request_processor
    │       │       ├──▶ auth_preprocessor.request_processor
    │       │       ├──▶ instructions.request_processor
    │       │       ├──▶ contents.request_processor (历史事件)
    │       │       ├──▶ _nl_planning.request_processor (规划)
    │       │       ├──▶ _code_execution.request_processor (代码)
    │       │       └──▶ [AutoFlow] agent_transfer.request_processor (转移)
    │       │
    │       ├──▶ _call_llm_async (调用 LLM)
    │       │       │
    │       │       ├──▶ before_model_callback
    │       │       ├──▶ llm.generate_content_async
    │       │       └──▶ after_model_callback
    │       │
    │       └──▶ _postprocess_async (响应后处理)
    │               │
    │               ├──▶ 响应处理器链
    │               ├──▶ _finalize_model_response_event (构建事件)
    │               └──▶ _postprocess_handle_function_calls_async
    │                       │
    │                       ├──▶ functions.handle_function_calls_async
    │                       ├──▶ [如有] 生成认证事件
    │                       ├──▶ [如有] 生成确认请求事件
    │                       └──▶ [AutoFlow] 转移检查
    │                               └──▶ agent_to_run.run_async (递归)
    │
    └──▶ End While
```

### 实时执行流程 (run_live)

```
LlmAgent.run_live
    │
    ▼
BaseLlmFlow.run_live
    │
    ├──▶ _preprocess_async (请求预处理)
    │
    ├──▶ llm.connect (建立 WebSocket 连接)
    │
    ├──▶ While True (重连循环)
    │       │
    │       ├──▶ asyncio.create_task(_send_to_model)  # 发送任务
    │       │       │
    │       │       └──▶ 从 live_request_queue 获取请求并发送
    │       │
    │       └──▶ _receive_from_model  # 接收循环
    │               │
    │               ├──▶ llm_connection.receive()
    │               ├──▶ _postprocess_live (实时后处理)
    │               │       ├──▶ 响应处理器链
    │               │       └──▶ _finalize_model_response_event
    │               │
    │               ├──▶ [如有函数调用] handle_function_calls_live
    │               │
    │               ├──▶ [转移请求] 转移给子 Agent
    │               │       └──▶ agent_to_run.run_live (递归)
    │               │
    │               └──▶ [任务完成] 关闭连接并返回
    │
    └──▶ 清理资源
```

---

## 关键时序图

### 标准对话时序

```
User    Runner    LlmAgent    BaseLlmFlow    LLM    Tool
 │         │          │            │          │       │
 │──msg──▶│          │            │          │       │
 │         │──run──▶│            │          │       │
 │         │          │──step───▶│          │       │
 │         │          │            │──req──▶│       │
 │         │          │            │◀──res──│       │
 │         │          │            │          │       │
 │         │          │            │◀──FC────│       │
 │         │          │            │          │──call▶│
 │         │          │            │          │◀─res──│
 │         │          │            │          │       │
 │         │          │            │───FR────▶│       │
 │         │          │            │          │       │
 │         │          │◀──────────│          │       │
 │         │          │            │          │       │
 │◀────────│◀────────│            │          │       │
 │         │          │            │          │       │
```

FC = Function Call, FR = Function Response

---

## 代码示例

### 使用 SingleFlow

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# SingleFlow 是默认配置，无需显式指定
agent = LlmAgent(
    name="single_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
    tools=[google_search],  # 只能调用工具，不能转移给其他 Agent
)
```

### 使用 AutoFlow

```python
from google.adk.agents import LlmAgent

# 定义子 Agent
sub_agent_1 = LlmAgent(
    name="specialist_1",
    description="Expert in field A",
    model="gemini-2.0-flash",
    instruction="You handle tasks related to field A.",
)

sub_agent_2 = LlmAgent(
    name="specialist_2",
    description="Expert in field B",
    model="gemini-2.0-flash",
    instruction="You handle tasks related to field B.",
)

# AutoFlow 自动启用当 Agent 有子 Agent 时
parent_agent = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="You coordinate between specialists.",
    sub_agents=[sub_agent_1, sub_agent_2],  # 启用 AutoFlow
)
```

### 禁用 Peer 转移

```python
agent = LlmAgent(
    name="restricted_agent",
    model="gemini-2.0-flash",
    description="I can only talk to my parent",
    disallow_transfer_to_peers=True,  # 禁止同级转移
    disallow_transfer_to_parent=False,  # 允许转移到父 Agent
)
```

---

## 总结

| 特性 | SingleFlow | AutoFlow |
|------|-----------|----------|
| **继承** | BaseLlmFlow | SingleFlow |
| **工具调用** | ✅ | ✅ |
| **Agent 转移** | ❌ | ✅ |
| **转移方向** | N/A | Parent↔Sub, Peer↔Peer |
| **适用场景** | 单一 Agent | 多 Agent 协作 |
| **配置方式** | 默认 | 添加 sub_agents 自动启用 |

**核心设计理念**:
1. **单一职责**: SingleFlow 专注工具调用，AutoFlow 扩展 Agent 协作
2. **可扩展**: 通过处理器链模式支持灵活扩展
3. **实时支持**: 统一的 WebSocket 实时音视频处理
4. **可恢复**: 支持长轮询工具调用的断点续传
