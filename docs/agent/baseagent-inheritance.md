# ADK BaseAgent 类继承关系与功能设计

## 概述

ADK (Agent Development Kit) 的 Agent 系统采用分层设计，所有 Agent 都继承自 `BaseAgent` 基类。本文档详细梳理 Agent 类的继承关系和每个类的功能设计。

## 类继承关系图

```
BaseAgent (基类)
│
├── LlmAgent (LLM智能体)
│   └── 支持文本对话和实时音视频对话
│   └── 支持工具调用、代码执行、规划器
│
├── SequentialAgent (顺序智能体)
│   └── 按顺序执行子智能体
│
├── ParallelAgent (并行智能体)
│   └── 并行执行子智能体
│
├── LoopAgent (循环智能体)
│   └── 循环执行子智能体直到满足退出条件
│
├── LangGraphAgent (LangGraph智能体)
│   └── 集成 LangGraph 工作流
│
└── RemoteA2aAgent (远程A2A智能体)
    └── 通过A2A协议与远程智能体通信
```

## 详细类说明

### 1. BaseAgent (基类)

**文件位置**: `src/google/adk/agents/base_agent.py`

**职责**: 所有 Agent 的抽象基类，定义 Agent 的基本行为和生命周期

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | Agent 名称，必须是有效的 Python 标识符，不能为 "user" |
| `description` | str | Agent 能力描述，用于模型判断是否需要调用该 Agent |
| `parent_agent` | Optional[BaseAgent] | 父 Agent，一个 Agent 只能有一个父 Agent |
| `sub_agents` | list[BaseAgent] | 子 Agent 列表 |
| `before_agent_callback` | Optional[BeforeAgentCallback] | Agent 执行前的回调函数 |
| `after_agent_callback` | Optional[AfterAgentCallback] | Agent 执行后的回调函数 |

**核心方法**:

#### 入口方法 (final，不可重写)
```python
async def run_async(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
    """文本对话入口方法"""
    
async def run_live(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
    """实时音视频对话入口方法"""
```

#### 实现方法 (子类必须实现)
```python
async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    """文本对话的核心实现逻辑"""

async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    """实时音视频对话的核心实现逻辑"""
```

**执行流程**:
```
run_async/run_live
├── 创建 InvocationContext
├── 调用 before_agent_callback
│   ├── 如果返回 content，直接返回该内容并结束
│   └── 如果修改了 state，生成 state 变更事件
├── 调用 _run_async_impl/_run_live_impl (核心逻辑)
│   └── 子类在此实现具体行为
└── 调用 after_agent_callback
    ├── 如果返回 content，追加额外响应事件
    └── 如果修改了 state，生成 state 变更事件
```

**辅助方法**:
- `clone()`: 深拷贝 Agent，递归复制所有子 Agent
- `find_agent(name)`: 在当前 Agent 及其后代中查找指定名称的 Agent
- `find_sub_agent(name)`: 在后代中查找 Agent
- `root_agent`: 获取根 Agent
- `_load_agent_state()`: 从 InvocationContext 加载 Agent 状态
- `_create_agent_state_event()`: 创建包含当前 Agent 状态的事件

---

### 2. LlmAgent (LLM智能体)

**文件位置**: `src/google/adk/agents/llm_agent.py`

**继承**: `BaseAgent`

**职责**: 基于大语言模型的智能体，支持工具调用、代码执行、规划等高级功能

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `model` | Union[str, BaseLlm] | 使用的模型，可继承自父 Agent |
| `instruction` | Union[str, InstructionProvider] | 动态指令，支持占位符变量 |
| `global_instruction` | Union[str, InstructionProvider] | **已废弃**，全局指令 |
| `static_instruction` | Optional[types.ContentUnion] | 静态指令，用于上下文缓存优化 |
| `tools` | list[ToolUnion] | Agent 可调用的工具列表 |
| `code_executor` | Optional[BaseCodeExecutor] | 代码执行器 |
| `planner` | Optional[BasePlanner] | 规划器 |
| `output_schema` | Optional[Type[BaseModel]] | 输出数据结构约束 |
| `input_schema` | Optional[Type[BaseModel]] | 输入数据结构约束 |

**回调函数**:
| 回调 | 类型 | 说明 |
|------|------|------|
| `before_model_callback` | BeforeModelCallback | 调用模型前的回调 |
| `after_model_callback` | AfterModelCallback | 调用模型后的回调 |
| `on_model_error_callback` | OnModelErrorCallback | 模型调用出错时的回调 |
| `before_tool_callback` | BeforeToolCallback | 调用工具前的回调 |
| `after_tool_callback` | AfterToolCallback | 调用工具后的回调 |
| `on_tool_error_callback` | OnToolErrorCallback | 工具调用出错时的回调 |

**工作流程**:
```
_run_async_impl
├── 构建 LLM 请求
│   ├── 处理指令 (instruction)
│   ├── 处理工具声明
│   └── 构建消息历史
├── 调用模型 (通过 LLMFlow)
│   └── 根据响应类型处理
│       ├── 文本响应 → 直接返回
│       ├── 函数调用 → 调用工具并获取结果
│       └── 转移请求 → 触发 Agent 转移
└── 可能循环多次直到获得最终响应
```

**Flow 类型**:
- `AutoFlow`: 自动流程，根据模型响应决定下一步
- `SingleFlow`: 单次调用流程

---

### 3. SequentialAgent (顺序智能体)

**文件位置**: `src/google/adk/agents/sequential_agent.py`

**继承**: `BaseAgent`

**状态类**: `SequentialAgentState`

**职责**: 容器型 Agent，按顺序依次执行子 Agent

**核心机制**:
```python
class SequentialAgentState(BaseAgentState):
    current_sub_agent: str = ''  # 当前执行的子 Agent 名称
```

**执行逻辑**:
```python
async def _run_async_impl(self, ctx: InvocationContext):
    # 1. 从状态恢复执行位置
    agent_state = self._load_agent_state(ctx, SequentialAgentState)
    start_index = self._get_start_index(agent_state)
    
    # 2. 按顺序执行子 Agent
    for sub_agent in self.sub_agents[start_index:]:
        # 保存当前执行位置到状态
        if ctx.is_resumable:
            agent_state = SequentialAgentState(current_sub_agent=sub_agent.name)
            ctx.set_agent_state(self.name, agent_state=agent_state)
            yield self._create_agent_state_event(ctx)
        
        # 执行子 Agent
        async for event in sub_agent.run_async(ctx):
            yield event
            if ctx.should_pause_invocation(event):
                return  # 暂停，下次从当前位置恢复
```

**使用场景**:
- 多步骤工作流
- 流水线处理
- 依赖顺序的任务链

---

### 4. ParallelAgent (并行智能体)

**文件位置**: `src/google/adk/agents/parallel_agent.py`

**继承**: `BaseAgent`

**职责**: 容器型 Agent，并行执行所有子 Agent

**核心机制**:

#### 分支隔离
每个子 Agent 在独立的分支中执行，避免子 Agent 间的事件互相可见：
```python
def _create_branch_ctx_for_sub_agent(agent, sub_agent, invocation_context):
    # 创建隔离的 branch
    invocation_context = invocation_context.model_copy()
    branch_suffix = f'{agent.name}.{sub_agent.name}'
    invocation_context.branch = (
        f'{invocation_context.branch}.{branch_suffix}'
        if invocation_context.branch
        else branch_suffix
    )
    return invocation_context
```

#### 事件合并 (Python 3.11+)
使用 `asyncio.TaskGroup` 实现并行执行和事件合并：
```python
async def _merge_agent_run(agent_runs: list[AsyncGenerator[Event, None]]):
    queue = asyncio.Queue()
    
    async with asyncio.TaskGroup() as tg:
        for events_for_one_agent in agent_runs:
            tg.create_task(process_an_agent(events_for_one_agent))
        
        # 从队列获取事件并交错输出
        while not all_done:
            event, resume_signal = await queue.get()
            yield event
            resume_signal.set()  # 通知 Agent 生成下一个事件
```

#### 事件合并 (Python 3.10)
对于不支持 `TaskGroup` 的 Python 版本，提供自定义实现：
```python
async def _merge_agent_run_pre_3_11(agent_runs):
    # 使用 asyncio.gather 和手动取消逻辑实现类似 TaskGroup 的行为
```

**使用场景**:
- 多个独立的并行任务
- 需要同时从多个来源收集信息
- 并行验证或检查

---

### 5. LoopAgent (循环智能体)

**文件位置**: `src/google/adk/agents/loop_agent.py`

**继承**: `BaseAgent`

**状态类**: `LoopAgentState`

**职责**: 容器型 Agent，循环执行子 Agent 直到满足退出条件

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `max_iterations` | Optional[int] | 最大循环次数，None 表示无限循环 |

**状态类**:
```python
class LoopAgentState(BaseAgentState):
    current_sub_agent: str = ''  # 当前执行的子 Agent 名称
    times_looped: int = 0        # 已循环次数
```

**退出条件**:
1. 某个子 Agent 生成 `escalate` 事件
2. 达到 `max_iterations` 限制

**执行逻辑**:
```python
async def _run_async_impl(self, ctx: InvocationContext):
    while (
        not self.max_iterations or times_looped < self.max_iterations
    ) and not should_exit:
        for sub_agent in self.sub_agents:
            # 执行子 Agent
            async for event in sub_agent.run_async(ctx):
                yield event
                if event.actions.escalate:
                    should_exit = True
                if ctx.should_pause_invocation(event):
                    pause_invocation = True
            
            if should_exit or pause_invocation:
                break
        
        # 重置子 Agent 状态，准备下一轮循环
        start_index = 0
        times_looped += 1
        ctx.reset_sub_agent_states(self.name)
```

**使用场景**:
- 迭代优化任务
- 多轮协商对话
- 直到满足条件的重复处理

---

### 6. LangGraphAgent (LangGraph智能体)

**文件位置**: `src/google/adk/agents/langgraph_agent.py`

**继承**: `BaseAgent`

**职责**: 集成 LangGraph 框架的智能体，支持 LangGraph 定义的工作流

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `graph` | CompiledGraph | LangGraph 编译后的图结构 |
| `instruction` | str | 系统指令 |

**执行逻辑**:
```python
async def _run_async_impl(self, ctx: InvocationContext):
    # 1. 配置 checkpointer 用于多轮对话
    config = {'configurable': {'thread_id': ctx.session.id}}
    
    # 2. 构建消息
    messages = []
    if self.instruction and not graph_messages:
        messages.append(SystemMessage(content=self.instruction))
    messages += self._get_messages(ctx.session.events)
    
    # 3. 调用 LangGraph
    final_state = self.graph.invoke({'messages': messages}, config)
    result = final_state['messages'][-1].content
    
    # 4. 返回结果作为 Event
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.Content(role='model', parts=[types.Part.from_text(text=result)]),
    )
```

**消息处理**:
- 如果 graph 有 checkpointer: 只提取最近的用户消息
- 如果 graph 没有 checkpointer: 提取完整的用户-Agent 对话历史

**使用场景**:
- 复用现有的 LangGraph 工作流
- 需要复杂状态管理的对话
- 与 LangChain 生态集成

---

### 7. RemoteA2aAgent (远程A2A智能体)

**文件位置**: `src/google/adk/agents/remote_a2a_agent.py`

**继承**: `BaseAgent`

**职责**: 通过 A2A (Agent-to-Agent) 协议与远程 Agent 通信的代理

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `agent_card` | Union[AgentCard, str] | 远程 Agent 的卡片，可以是对象、URL 或文件路径 |
| `timeout` | float | HTTP 超时时间（秒） |
| `a2a_client_factory` | Optional[A2AClientFactory] | A2A 客户端工厂 |
| `a2a_request_meta_provider` | Callable | 请求元数据提供者 |

**核心能力**:

#### Agent Card 解析
支持多种方式指定远程 Agent:
1. **AgentCard 对象**: 直接传入 AgentCard 实例
2. **URL**: 自动从 `/.well-known/agent.json` 获取
3. **文件路径**: 从本地 JSON 文件加载

#### 消息转换
ADK Event ↔ A2A Message 双向转换:
- 支持文本、文件、表单等多种 Part 类型
- 支持函数调用和响应的映射

#### 会话状态管理
- 支持长轮询和多轮对话
- 处理任务状态更新和工件更新事件

**执行流程**:
```python
async def _run_async_impl(self, ctx: InvocationContext):
    # 1. 解析或加载 Agent Card
    # 2. 创建 A2A 客户端
    # 3. 转换 Event 为 A2A Message
    # 4. 发送任务到远程 Agent
    # 5. 处理响应流（状态更新、工件更新）
    # 6. 转换 A2A 响应为 ADK Event
    # 7. 返回结果
```

**使用场景**:
- 调用外部第三方 Agent
- 微服务架构中的 Agent 通信
- 跨平台 Agent 协作

---

## 配置类体系

每个 Agent 类都有对应的配置类，用于从配置文件创建 Agent：

```
BaseAgentConfig (基类配置)
│
├── LlmAgentConfig (LLM Agent 配置)
├── SequentialAgentConfig (顺序 Agent 配置)
├── ParallelAgentConfig (并行 Agent 配置)
└── LoopAgentConfig (循环 Agent 配置)
```

### 配置基类 (BaseAgentConfig)

**文件位置**: `src/google/adk/agents/base_agent_config.py`

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | Agent 名称 |
| `description` | str | Agent 描述 |
| `sub_agents` | Optional[list[AgentRefConfig]] | 子 Agent 引用配置 |
| `before_agent_callbacks` | Optional[list[CallbackConfig]] | 前置回调配置 |
| `after_agent_callbacks` | Optional[list[CallbackConfig]] | 后置回调配置 |

### 配置使用示例

```python
# 从配置文件创建 Agent
from google.adk.agents import LlmAgent
from google.adk.agents.llm_agent_config import LlmAgentConfig

config = LlmAgentConfig(
    name="my_agent",
    description="An agent that can search the web",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
    tools=["google_search"]
)

agent = LlmAgent.from_config(config, "/path/to/config.yaml")
```

---

## 状态管理

### BaseAgentState (基类状态)

**文件位置**: `src/google/adk/agents/base_agent.py`

```python
class BaseAgentState(BaseModel):
    """Base class for all agent states."""
    pass
```

### 各 Agent 状态类

| Agent | 状态类 | 状态属性 |
|-------|--------|----------|
| SequentialAgent | SequentialAgentState | `current_sub_agent: str` |
| LoopAgent | LoopAgentState | `current_sub_agent: str`, `times_looped: int` |

### 状态生命周期

```
1. Agent 开始执行
   └── _load_agent_state(ctx, StateType) 尝试恢复状态
   
2. Agent 执行中
   └── ctx.set_agent_state(agent_name, agent_state=state) 保存状态
   └── yield _create_agent_state_event(ctx) 生成状态事件
   
3. Agent 执行结束
   └── ctx.set_agent_state(agent_name, end_of_agent=True) 标记结束
   └── yield _create_agent_state_event(ctx) 生成结束事件
```

---

## 回调机制

### Agent 级别回调

**before_agent_callback**:
- 在 Agent 执行前调用
- 可以返回 `Content` 来跳过 Agent 执行
- 可以修改 state 来记录前置操作

**after_agent_callback**:
- 在 Agent 执行后调用
- 可以返回额外 `Content` 追加到响应
- 可以修改 state 来记录后置操作

### LlmAgent 专用回调

**before_model_callback**:
```python
def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 可以修改 llm_request
    # 可以返回 LlmResponse 跳过模型调用
```

**after_model_callback**:
```python
def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    # 可以修改 llm_response
    # 可以返回新的 LlmResponse 替换原响应
```

**on_model_error_callback**:
```python
def on_model_error_callback(callback_context: CallbackContext, llm_request: LlmRequest, error: Exception) -> Optional[LlmResponse]:
    # 处理模型调用错误
    # 可以返回 LlmResponse 作为错误恢复
```

**before_tool_callback**:
```python
def before_tool_callback(tool: BaseTool, tool_args: dict, tool_context: ToolContext) -> Optional[dict]:
    # 可以修改 tool_args
    # 可以返回 dict 跳过工具调用
```

**after_tool_callback**:
```python
def after_tool_callback(tool: BaseTool, tool_args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
    # 可以修改 tool_response
    # 可以返回新的 dict 替换原响应
```

**on_tool_error_callback**:
```python
def on_tool_error_callback(tool: BaseTool, tool_args: dict, tool_context: ToolContext, error: Exception) -> Optional[dict]:
    # 处理工具调用错误
    # 可以返回 dict 作为错误恢复
```

---

## 使用示例

### 基础 LlmAgent

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

agent = LlmAgent(
    name="search_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant that can search the web.",
    description="An assistant that can search the web",
    tools=[google_search]
)
```

### 顺序工作流

```python
from google.adk.agents import SequentialAgent, LlmAgent

# 定义子 Agent
extractor = LlmAgent(name="extractor", instruction="Extract key information...")
summarizer = LlmAgent(name="summarizer", instruction="Summarize the extracted info...")
validator = LlmAgent(name="validator", instruction="Validate the summary...")

# 创建顺序 Agent
pipeline = SequentialAgent(
    name="processing_pipeline",
    description="A pipeline that extracts, summarizes, and validates",
    sub_agents=[extractor, summarizer, validator]
)
```

### 并行执行

```python
from google.adk.agents import ParallelAgent, LlmAgent

# 定义多个独立 Agent
researcher_a = LlmAgent(name="researcher_a", instruction="Research topic A...")
researcher_b = LlmAgent(name="researcher_b", instruction="Research topic B...")
researcher_c = LlmAgent(name="researcher_c", instruction="Research topic C...")

# 并行执行所有研究
research_team = ParallelAgent(
    name="research_team",
    description="A team that researches multiple topics in parallel",
    sub_agents=[researcher_a, researcher_b, researcher_c]
)
```

### 循环处理

```python
from google.adk.agents import LoopAgent, LlmAgent

# 定义需要迭代的 Agent
optimizer = LlmAgent(
    name="optimizer",
    instruction="Improve the content based on feedback..."
)

critic = LlmAgent(
    name="critic",
    instruction="Review and provide feedback...",
    # 当内容足够好时触发 escalate
)

# 循环优化直到满意或达到最大迭代次数
optimizer_loop = LoopAgent(
    name="optimizer_loop",
    description="Iteratively optimize content",
    sub_agents=[optimizer, critic],
    max_iterations=5
)
```

---

## 总结

ADK 的 Agent 架构设计遵循以下原则：

1. **单一职责**: 每个 Agent 类专注于特定的执行模式
2. **可组合性**: 容器型 Agent (Sequential, Parallel, Loop) 可以嵌套任意类型的子 Agent
3. **状态管理**: 支持可恢复的执行，通过状态保存和恢复实现断点续传
4. **扩展性**: 通过回调机制允许在不修改核心逻辑的情况下定制行为
5. **互操作性**: LangGraphAgent 和 RemoteA2aAgent 支持与外部框架和远程服务的集成

选择 Agent 类型的决策树：

```
是否需要调用 LLM?
├── 是 → LlmAgent
└── 否 → 需要组合多个 Agent?
    ├── 按顺序执行 → SequentialAgent
    ├── 并行执行 → ParallelAgent
    ├── 循环执行 → LoopAgent
    ├── 使用 LangGraph 工作流 → LangGraphAgent
    └── 调用远程 Agent → RemoteA2aAgent
```
