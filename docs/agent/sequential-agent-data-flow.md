# SequentialAgent Agent 间信息传递流程详解

## 概述

本文档以博客创作流水线为例，详细讲解 ADK 中 `SequentialAgent` 如何通过 **Session State** 在多个 Agent 之间传递信息。

## 示例场景

博客创作流水线包含 4 个 Agent 按顺序执行：

1. **researcher** - 研究主题，收集信息
2. **writer** - 根据研究结果撰写草稿
3. **editor** - 审阅草稿并提供反馈
4. **formatter** - 应用反馈并格式化最终文章

```python
blog_creation_pipeline = SequentialAgent(
    name="BlogCreationPipeline",
    sub_agents=[research_agent, writer_agent, editor_agent, formatter_agent]
)
```

## 核心机制

### 1. SequentialAgent 执行模型

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SequentialAgent (容器)                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  researcher  │───▶│   writer     │───▶│    editor    │───▶...   │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│    state['findings']   state['draft']      state['feedback']       │
└─────────────────────────────────────────────────────────────────────┘

关键特点：
- 所有子 Agent 共享同一个 Session 和 State
- 按列表顺序依次执行
- 前一个 Agent 的 state 变更对后一个 Agent 立即可见
```

### 2. 信息传递的两大支柱

#### A. output_key - 输出保存机制

**定义**: 在 `LlmAgent` 中设置 `output_key`，Agent 的最终输出会自动保存到 Session State。

**代码示例**:
```python
research_agent = Agent(
    name="researcher",
    model=qwen_model,
    instruction="...",
    output_key="research_findings"  # 输出保存到 state['research_findings']
)
```

**工作原理**:

```python
# llm_agent.py:792-813
if (
    self.output_key
    and event.is_final_response()
    and event.content
    and event.content.parts
):
    # 提取文本内容
    result = ''.join(
        part.text for part in event.content.parts
        if part.text and not part.thought
    )
    
    # 保存到 state_delta
    event.actions.state_delta[self.output_key] = result
```

**持久化流程**:
```
Agent 生成响应
    │
    ▼
event.actions.state_delta['research_findings'] = '研究结果内容'
    │
    ▼
SessionService.append_event() 
    │
    ▼
session.state.update({'research_findings': '研究结果内容'})
    │
    ▼
下一个 Agent 可通过 state['research_findings'] 读取
```

#### B. instruction 变量注入 - 输入读取机制

**定义**: 在 `instruction` 中使用 `{variable_name}` 占位符，系统会自动从 Session State 注入值。

**代码示例**:
```python
writer_agent = Agent(
    name="writer",
    model=qwen_model,
    instruction=(
        "根据以下研究发现撰写博客:\n"
        "{research_findings}\n"  # 自动注入 state['research_findings']
        "要求:..."
    ),
    output_key="draft_post"
)
```

**工作原理**:

```python
# instructions_utils.py:30-124
async def inject_session_state(template: str, readonly_context: ReadonlyContext) -> str:
    # 1. 使用正则匹配 {variable_name}
    pattern = r'{+[^{}]*}+'
    
    # 2. 对每个匹配项进行替换
    for match in re.finditer(pattern, template):
        var_name = match.group().strip('{}')
        
        # 3. 从 session.state 获取值
        if var_name in invocation_context.session.state:
            value = invocation_context.session.state[var_name]
            template = template.replace(match.group(), str(value))
    
    return template
```

**支持特性**:
- **标准变量**: `{variable_name}` - 从 state 读取
- **可选变量**: `{variable_name?}` - 不存在时替换为空字符串
- **Artifact**: `{artifact.filename}` - 从 artifact service 加载

## 完整数据流分析

### 阶段 1: Research Agent 执行

```python
# Agent 定义
research_agent = Agent(
    name="researcher",
    instruction="研究主题并提供5-7个要点...",
    output_key="research_findings"
)
```

**执行流程**:
```
1. SequentialAgent 调用 researcher.run_async()
   │
2. LlmFlow 处理 instruction（无变量需要注入）
   │
3. 调用 LLM 生成研究结果
   │
4. LlmAgent 检测到 output_key="research_findings"
   │
5. 将结果保存到 event.actions.state_delta['research_findings']
   │
6. 事件被追加到 Session
   │
7. Session.state 更新: {'research_findings': '• 要点1\n• 要点2...'}
```

### 阶段 2: Writer Agent 执行

```python
# Agent 定义
writer_agent = Agent(
    name="writer",
    instruction=(
        "根据研究发现撰写博客:\n"
        "{research_findings}\n"  # 等待注入
        "要求:..."
    ),
    output_key="draft_post"
)
```

**执行流程**:
```
1. SequentialAgent 调用 writer.run_async()
   │
2. LlmFlow 预处理 instruction
   │
3. _InstructionsLlmRequestProcessor 检测到 {research_findings}
   │
4. 调用 inject_session_state(template, context)
   │
5. 从 invocation_context.session.state['research_findings'] 读取值
   │
6. 替换后的 instruction:
   "根据研究发现撰写博客:\n• 要点1\n• 要点2...\n要求:..."
   │
7. 调用 LLM 生成博客草稿
   │
8. 将草稿保存到 event.actions.state_delta['draft_post']
   │
9. Session.state 更新: {'research_findings': '...', 'draft_post': '博客内容...'}
```

### 阶段 3: Editor Agent 执行

```python
# Agent 定义
editor_agent = Agent(
    name="editor",
    instruction=(
        "审阅以下博客草稿:\n"
        "{draft_post}\n"  # 等待注入
        "从以下方面分析..."
    ),
    output_key="editorial_feedback"
)
```

**执行流程**:
```
1. SequentialAgent 调用 editor.run_async()
   │
2. 预处理 instruction，注入 {draft_post}
   │
3. 从 state['draft_post'] 读取博客内容
   │
4. 替换后的 instruction 包含完整博客文本
   │
5. 调用 LLM 生成审阅反馈
   │
6. 将反馈保存到 state['editorial_feedback']
```

### 阶段 4: Formatter Agent 执行

```python
# Agent 定义
formatter_agent = Agent(
    name="formatter",
    instruction=(
        "根据反馈完善博客:\n"
        "原始草稿:\n{draft_post}\n"           # 注入草稿
        "\n反馈意见:\n{editorial_feedback}\n"  # 注入反馈
        "要求:应用改进建议并格式化..."
    ),
    output_key="final_post"
)
```

**执行流程**:
```
1. SequentialAgent 调用 formatter.run_async()
   │
2. 预处理 instruction
   │
3. 同时注入两个变量:
   - {draft_post} → state['draft_post']
   - {editorial_feedback} → state['editorial_feedback']
   │
4. 替换后的 instruction 包含草稿和反馈
   │
5. 调用 LLM 生成最终文章
   │
6. 将最终结果保存到 state['final_post']
```

## 时序图

```
User    SequentialAgent   Session   Researcher  Writer   Editor   Formatter
 │            │              │           │          │        │         │
 │──query──▶│              │           │          │        │         │
 │            │──run_async──▶│           │          │        │         │
 │            │              │──run_async▶│          │        │         │
 │            │              │           │          │        │         │
 │            │              │           │─研究主题─▶│        │         │
 │            │              │           │          │        │         │
 │            │              │◀──结果───│          │        │         │
 │            │              │           │          │        │         │
 │            │              │  state['findings']='结果'
 │            │              │           │          │        │         │
 │            │              │◀──完成───│          │        │         │
 │            │              │           │          │        │         │
 │            │              │──run_async──▶│       │        │         │
 │            │              │           │ 注入{findings}    │         │
 │            │              │           │          │        │         │
 │            │              │           │─撰写博客─▶│        │         │
 │            │              │           │          │        │         │
 │            │              │           │◀──草稿───│        │         │
 │            │              │           │          │        │         │
 │            │              │  state['draft']='草稿'
 │            │              │           │          │        │         │
 │            │              │◀──完成────│          │        │         │
 │            │              │           │          │        │         │
 │            │              │──run_async───────────▶│      │         │
 │            │              │           │          │ 注入{draft}     │
 │            │              │           │          │        │         │
 │            │              │           │          │─审阅──▶│         │
 │            │              │           │          │        │         │
 │            │              │           │          │◀─反馈──│         │
 │            │              │           │          │        │         │
 │            │              │  state['feedback']='反馈'
 │            │              │           │          │        │         │
 │            │              │◀──完成────│          │        │         │
 │            │              │           │          │        │         │
 │            │              │──run_async─────────────────────▶│       │
 │            │              │           │          │        │ 注入{draft,feedback}
 │            │              │           │          │        │         │
 │            │              │           │          │        │─完善───▶│
 │            │              │           │          │        │         │
 │            │              │           │          │        │◀─最终──│
 │            │              │           │          │        │         │
 │            │              │  state['final']='最终文章'
 │            │              │           │          │        │         │
 │◀───────────│◀────────────│◀──────────│          │        │         │
 │            │              │           │          │        │         │
```

## 关键源码位置

| 功能 | 文件路径 | 关键代码 |
|------|----------|----------|
| **SequentialAgent 执行** | `src/google/adk/agents/sequential_agent.py:54-91` | 按顺序循环执行 sub_agents |
| **output_key 保存** | `src/google/adk/agents/llm_agent.py:792-813` | 将 Agent 输出保存到 state_delta |
| **instruction 注入** | `src/google/adk/utils/instructions_utils.py:30-124` | 模板变量替换实现 |
| **instruction 处理** | `src/google/adk/flows/llm_flows/instructions.py:37-57` | 请求处理器调用注入 |
| **state 持久化** | `src/google/adk/sessions/base_session_service.py:126-133` | 将 delta 合并到 session.state |

## 最佳实践

### 1. 命名规范

```python
# 好的命名：清晰表达内容含义
output_key="research_findings"
output_key="blog_draft"
output_key="editorial_feedback"
output_key="final_post"

# 避免的命名：过于模糊或过于具体
output_key="output"        # 太模糊
output_key="research_agent_output_key_2024"  # 没必要这么长
```

### 2. 变量注入格式

```python
# ✅ 标准格式
"{research_findings}"

# ✅ 可选变量（不存在时为空）
"{optional_context?}"

# ✅ Artifact 引用
"{artifact.data.json}"

# ❌ 错误格式
"{ research_findings }"  # 多余空格
"{research-findings}"    # 非法字符，应使用下划线
```

### 3. 数据大小考虑

```python
# 如果数据量很大，考虑只传递引用或摘要
instruction=(
    "根据以下研究摘要:\n"
    "{research_summary}\n"  # 只传递摘要，而非完整数据
    "..."
)

# 大数据通过 artifact 传递
instruction=(
    "分析附件中的数据:\n"
    "{artifact.large_dataset.csv}\n"
)
```

### 4. 错误处理

```python
# 使用可选变量避免 KeyError
instruction=(
    "根据以下信息:\n"
    "{required_data}\n"
    "可选上下文: {optional_data?}\n"  # 问号表示可选
)

# 或者在代码中检查
async def build_instruction(readonly_context):
    state = readonly_context._invocation_context.session.state
    if 'data' not in state:
        return "请提供数据"
    return f"分析以下数据: {state['data']}"
```

## 常见问题

### Q: 为什么需要 output_key？不能直接读取上一个 Agent 的事件吗？

**A**: 
- 可以读取事件，但需要遍历事件历史，效率低
- `output_key` 提供了结构化的数据访问方式
- 简化了 Agent 间数据传递的复杂度
- 便于调试和跟踪数据流

### Q: 如果 instruction 中的变量不存在会怎样？

**A**:
- 标准格式 `{var}`: 抛出 `KeyError`，执行失败
- 可选格式 `{var?}`: 替换为空字符串，继续执行

### Q: 可以传递复杂数据类型（如 dict/list）吗？

**A**:
- 可以，会自动转为 JSON 字符串
- 建议显式控制序列化：
```python
import json
state['data'] = json.dumps(complex_object)
```

### Q: SequentialAgent 执行失败会怎样？

**A**:
- 默认会抛出异常，整个流程中断
- 可以通过回调函数实现错误恢复
- 支持可恢复执行（resumable）模式，从断点继续

## 总结

SequentialAgent 通过 **Session State** 实现 Agent 间信息传递，核心机制包括：

1. **output_key**: Agent 输出自动保存到 State
2. **instruction 变量**: 从 State 读取数据并注入到指令
3. **共享 Session**: 所有子 Agent 访问同一个 State 对象

这种设计实现了：
- ✅ 松耦合的 Agent 协作
- ✅ 清晰的数据流
- ✅ 可追踪的执行过程
- ✅ 支持断点恢复
