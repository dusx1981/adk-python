# ADK Agent 之间通过 Session State 交互信息完整流程

## 概述

在 ADK (Agent Development Kit) 中，Agent 之间通过 **Session State** 进行信息共享和传递。这种机制允许多个 Agent 在同一个会话中协作，共享上下文信息。本文档详细梳理这一交互的完整流程。

## 核心概念

### 1. Session（会话）

Session 代表用户与 Agent 之间的一系列交互。每个 Session 包含：
- **id**: 唯一标识符
- **state**: 状态字典，存储共享信息
- **events**: 事件列表，记录所有交互历史
- **app_name**: 应用名称
- **user_id**: 用户ID

### 2. State（状态）

State 是一个特殊的字典类，维护当前值和待提交的变更（delta）：

```python
class State:
    APP_PREFIX = "app:"    # 应用级状态前缀
    USER_PREFIX = "user:"  # 用户级状态前缀  
    TEMP_PREFIX = "temp:"  # 临时状态前缀
    
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def has_delta(self) -> bool: ...
```

### 3. EventActions（事件动作）

每个事件可以附带动作，其中 `state_delta` 字段用于记录状态变更：

```python
class EventActions:
    state_delta: dict[str, object]  # 状态变更增量
    artifact_delta: dict[str, int]  # 工件版本变更
    transfer_to_agent: Optional[str] # 转移目标 Agent
    ...
```

## 交互流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Parent Agent                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ CallbackContext │───▶│    ToolContext  │───▶│  AgentTool   │ │
│  │  - state        │    │  - state        │    │  (sub-agent) │ │
│  └─────────────────┘    └─────────────────┘    └──────┬───────┘ │
└───────────────────────────────────────────────────────┼──────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Child Agent (Sub-agent)                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Runner.run_async│───▶│InvocationContext│───▶│ Agent.run_async││
│  │                 │    │  - session      │    │              │ │
│  └─────────────────┘    │  - session.state│    └──────────────┘ │
│                         └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
        │
        │ 返回 Event (携带 state_delta)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    State 同步回 Parent Session                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   提取 delta    │───▶│ tool_context.   │───▶│ Parent State │ │
│  │  event.actions  │    │ state.update()  │    │   更新       │ │
│  │  .state_delta   │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 详细流程步骤

### 步骤 1: 初始化 Session

当 Runner 启动时，会从 SessionService 获取或创建 Session：

```python
# runners.py:438
session = await self.session_service.get_session(
    app_name=self.app_name, 
    user_id=user_id, 
    session_id=session_id
)
```

### 步骤 2: 创建 InvocationContext

InvocationContext 是单次调用的上下文，包含 session 引用：

```python
# invocation_context.py:98-165
class InvocationContext(BaseModel):
    session: Session           # 当前会话（包含 state）
    agent: BaseAgent           # 当前 Agent
    agent_states: dict[str, dict[str, Any]]  # Agent 状态
    ...
```

### 步骤 3: Agent 执行与 State 访问

当 Agent 运行时，通过 CallbackContext 访问和修改 state：

```python
# callback_context.py:36-64
class CallbackContext(ReadonlyContext):
    def __init__(self, invocation_context, event_actions=None):
        # 创建 State 对象，绑定到 session.state 和 event_actions.state_delta
        self._state = State(
            value=invocation_context.session.state,
            delta=self._event_actions.state_delta,
        )
    
    @property
    def state(self) -> State:
        return self._state
```

**关键机制**: State 使用双字典设计：
- `_value`: 当前 session 的实际状态值
- `_delta`: 本次调用中待提交的变更

当修改 `ctx.state['key'] = value` 时，同时更新 `_value` 和 `_delta`。

### 步骤 4: AgentTool 调用子 Agent

当一个 Agent 调用另一个 Agent 作为工具时（AgentTool）：

```python
# agent_tool.py:113-198
async def run_async(self, *, args, tool_context):
    # 1. 创建子 Runner
    runner = Runner(
        app_name=child_app_name,
        agent=self.agent,
        session_service=InMemorySessionService(),  # 子 Session 服务
        ...
    )
    
    # 2. 提取父 state（过滤内部键）
    state_dict = {
        k: v for k, v in tool_context.state.to_dict().items()
        if not k.startswith('_adk')
    }
    
    # 3. 创建子 Session，继承父 state
    session = await runner.session_service.create_session(
        app_name=child_app_name,
        user_id=tool_context._invocation_context.user_id,
        state=state_dict,  # 传递父 state
    )
    
    # 4. 运行子 Agent
    async for event in runner.run_async(...):
        # 5. 关键：将子 Agent 的 state_delta 同步回父 Session
        if event.actions.state_delta:
            tool_context.state.update(event.actions.state_delta)
        ...
```

**关键同步点**: 在 `agent_tool.py:179-181`
```python
# Forward state delta to parent session.
if event.actions.state_delta:
    tool_context.state.update(event.actions.state_delta)
```

### 步骤 5: State 变更持久化

当事件被添加到 Session 时，state_delta 会被处理：

```python
# base_session_service.py:105-133
async def append_event(self, session, event):
    # 1. 修剪临时状态（temp: 前缀的键不保存）
    event = self._trim_temp_delta_state(event)
    
    # 2. 更新 session state
    self._update_session_state(session, event)
    
    # 3. 添加事件到 session
    session.events.append(event)

def _update_session_state(self, session, event):
    if not event.actions or not event.actions.state_delta:
        return
    for key, value in event.actions.state_delta.items():
        if key.startswith(State.TEMP_PREFIX):
            continue  # 跳过临时状态
        session.state.update({key: value})
```

### 步骤 6: 存储层处理

不同的 SessionService 实现处理 state 存储：

#### InMemorySessionService（内存存储）
```python
# 直接在内存中维护 session.state
```

#### SqliteSessionService（SQLite 存储）
```python
# sqlite_session_service.py:143-179
async def create_session(self, *, app_name, user_id, state=None):
    # 1. 分离 state 为不同层级
    state_deltas = _session_util.extract_state_delta(state)
    session_state = state_deltas["session"]
    
    # 2. 保存到数据库
    await db.execute(
        "INSERT INTO sessions (app_name, user_id, id, state, ...)",
        (app_name, user_id, session_id, json.dumps(session_state), ...)
    )
    
    # 3. 合并 app、user、session 三层 state
    merged_state = _merge_state(app_state, user_state, session_state)
    return Session(..., state=merged_state)
```

## State 前缀层级

State 支持三种层级的前缀，用于不同范围的共享：

| 前缀 | 含义 | 使用场景 |
|------|------|----------|
| `app:` | 应用级状态 | 所有用户、所有会话共享 |
| `user:` | 用户级状态 | 同一用户的所有会话共享 |
| `temp:` | 临时状态 | 单次调用，不持久化 |
| (无前缀) | 会话级状态 | 仅当前会话 |

**示例**:
```python
# 在工具中设置不同层级的状态
tool_context.state['app:global_config'] = {'theme': 'dark'}
tool_context.state['user:profile'] = {'name': 'Alice'}
tool_context.state['temp:calculation_result'] = 42  # 不会被保存
tool_context.state['conversation_count'] = 5  # 仅当前会话
```

## 关键代码路径

### 1. State 读取路径
```
Tool/Callback -> CallbackContext.state -> State.__getitem__
-> 先查 delta，再查 value
```

### 2. State 写入路径
```
Tool/Callback -> ctx.state['key'] = value -> State.__setitem__
-> 更新 _value 和 _delta
-> 生成 Event -> EventActions.state_delta
-> BaseSessionService.append_event -> 更新 session.state
```

### 3. 跨 Agent State 同步路径
```
Parent Agent -> AgentTool.run_async -> Child Runner
-> Child Agent 执行产生 state_delta
-> 返回 Event -> tool_context.state.update(delta)
-> Parent Session state 同步更新
```

## 时序图

```
User    ParentRunner    ParentAgent    AgentTool    ChildRunner    ChildAgent    SessionService
 |           |               |             |             |              |              |
 |──调用──▶  │               │             │             │              │              |
 │           │──run_async──▶│             │             │              │              |
 │           │               │──调用工具──▶│             │              │              |
 │           │               │             │──创建──────▶│              │              |
 │           │               │             │  Runner     │              │              |
 │           │               │             │             │              │              │
 │           │               │             │──create─────│──────────────│────────────▶│
 │           │               │             │  _session   │              │              │
 │           │               │             │             │              │              │
 │           │               │             │             │──run_async──▶│              │
 │           │               │             │             │              │              │
 │           │               │             │             │              │──修改──────▶│
 │           │               │             │             │              │  state       │
 │           │               │             │             │              │              │
 │           │               │             │             │              │◀────────────│
 │           │               │             │             │              │  返回 Event  │
 │           │               │             │             │              │  (含 delta)  │
 │           │               │             │             │              │              │
 │           │               │             │◀────────────│──────────────│              │
 │           │               │             │  同步 delta │              │              │
 │           │               │             │  到 parent  │              │              │
 │           │               │             │  state      │              │              │
 │           │               │             │             │              │              │
 │           │               │◀────────────│             │              │              │
 │           │               │  返回结果   │             │              │              │
 │           │               │             │             │              │              │
 │           │◀──────────────│             │             │              │              │
 │           │  生成事件     │             │             │              │              │
 │◀──────────│  (含 state)   │             │             │              │              │
 │           │               │             │             │              │              │
 │           │──append_event────────────────────────────────────────────────────────▶│
 │           │               │             │             │              │              │
 │           │               │             │             │              │              │
```

## 最佳实践

### 1. 使用适当的前缀
```python
# 应用级配置 - 所有用户共享
ctx.state['app:config'] = {'api_version': 'v1'}

# 用户级数据 - 跨会话持久化
tool_context.state['user:preferences'] = {'language': 'zh'}

# 会话级数据 - 仅当前会话
tool_context.state['current_topic'] = 'weather'

# 临时数据 - 不保存
tool_context.state['temp:intermediate_result'] = some_calculation
```

### 2. 避免在 AgentTool 中直接修改父 state
```python
# 推荐：在子 Agent 中修改，通过 state_delta 自动同步
# 不推荐：直接操作 tool_context.state，可能产生竞态条件
```

### 3. 注意 State 合并策略
当使用 DatabaseSessionService 时，state 分层存储：
- App state: 应用级，所有会话共享
- User state: 用户级，该用户的所有会话共享  
- Session state: 会话级，仅当前会话

读取时合并三层，写入时根据键前缀决定保存位置。

## 常见问题

### Q: 为什么子 Agent 的 state 变更能同步到父 Agent？
A: 通过 `AgentTool.run_async` 中的事件循环监听，提取子 Agent 事件的 `state_delta` 并调用 `tool_context.state.update()` 实现同步。

### Q: temp: 前缀的状态为什么消失了？
A: 临时状态在 `BaseSessionService._trim_temp_delta_state()` 中被过滤，不会保存到存储层。

### Q: 如何实现 Agent 间的私有状态？
A: 使用分支(branch)机制，或在 state 键名中包含 Agent 名称作为命名空间。

## 相关源码文件

| 文件 | 职责 |
|------|------|
| `src/google/adk/sessions/session.py` | Session 模型定义 |
| `src/google/adk/sessions/state.py` | State 双字典实现 |
| `src/google/adk/sessions/base_session_service.py` | State 变更持久化 |
| `src/google/adk/agents/invocation_context.py` | 调用上下文 |
| `src/google/adk/agents/callback_context.py` | State 访问接口 |
| `src/google/adk/tools/agent_tool.py` | 跨 Agent State 同步 |
| `src/google/adk/tools/tool_context.py` | 工具上下文 |
| `src/google/adk/runners.py` | Runner 执行流程 |
| `src/google/adk/events/event_actions.py` | State delta 定义 |

## 总结

ADK 的 Session State 机制通过以下设计实现 Agent 间信息共享：

1. **双字典 State**: 同时维护当前值和待提交变更
2. **Event 驱动**: 状态变更通过 Event 的 `state_delta` 传递
3. **层级前缀**: 支持应用、用户、会话、临时四级状态
4. **自动同步**: AgentTool 自动将子 Agent 的 state 同步回父会话
5. **持久化抽象**: SessionService 提供统一的存储接口

这种设计既保证了状态的一致性，又支持灵活的 Agent 协作模式。
