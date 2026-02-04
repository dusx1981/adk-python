# _InstructionsLlmRequestProcessor 技术文档

## 一、概述

`_InstructionsLlmRequestProcessor` 是 ADK LLM Flow 中的指令处理器，负责在向 LLM 发送请求前，处理和注入动态内容到 Agent 指令中。

## 二、类定位

```
ADK 架构
├── flows/llm_flows/
│   ├── instructions.py           ← _InstructionsLlmRequestProcessor 所在
│   ├── base_llm_processor.py      ← 基类 BaseLlmRequestProcessor
│   └── ...
├── utils/
│   └── instructions_utils.py      ← inject_session_state 所在
├── agents/
│   └── ...
└── ...
```

## 三、核心职责

| 方法 | 职责 |
|------|------|
| `run_async()` | 主入口，处理 global_instruction、static_instruction、instruction |
| `_process_agent_instruction()` | 调用 `inject_session_state` 注入状态变量 |

## 四、处理流程图

```
LLM Request
    ↓
┌───────────────────────────────────────┐
│  1. 处理 global_instruction (已废弃)   │
│     → canonical_global_instruction()  │
│     → inject_session_state()           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  2. 处理 static_instruction           │
│     → 直接追加到 LLM Request          │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  3. 处理 dynamic instruction          │
│     → canonical_instruction()         │
│     → inject_session_state()           │
│     → 注入到 LLM Request               │
└───────────────────────────────────────┘
    ↓
LLM Request (含处理后的指令)
```

## 五、inject_session_state 详解

### 5.1 函数签名

```python
async def inject_session_state(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `template` | `str` | 包含 `{变量占位符}` 的指令模板 |
| `readonly_context` | `ReadonlyContext` | 只读上下文，包含 session.state |

### 5.2 核心逻辑

```python
async def inject_session_state(template: str, readonly_context) -> str:
    invocation_context = readonly_context._invocation_context

    # 定义异步替换函数
    async def _async_sub(pattern, repl_async_fn, string) -> str:
        result = []
        last_end = 0
        for match in re.finditer(pattern, string):
            result.append(string[last_end : match.start()])  # 追加匹配前的文本
            replacement = await repl_async_fn(match)         # 获取替换值
            result.append(replacement)                         # 追加替换值
            last_end = match.end()                           # 更新位置
        result.append(string[last_end:])                      # 追加剩余文本
        return ''.join(result)                              # 拼接结果

    # 调用替换
    return await _async_sub(r'{+[^{}]*}+', _replace_match, template)
```

### 5.3 变量类型支持

| 变量类型 | 语法 | 来源 |
|---------|------|------|
| 普通变量 | `{var_name}` | `session.state["var_name"]` |
| 应用变量 | `{app:key}` | `session.state["app:key"]` |
| 用户变量 | `{user:key}` | `session.state["user:key"]` |
| 临时变量 | `{temp:key}` | `session.state["temp:key"]` |
| Artifact | `{artifact.filename}` | `artifact_service.load_artifact()` |
| 可选变量 | `{var_name?}` | 缺失时不报错，返回空字符串 |

## 六、正则表达式详解

### 6.1 正则模式

```python
r'{+[^{}]*}+'
```

### 6.2 模式分解

| 部分 | 含义 |
|------|------|
| `{` | 匹配左花括号 |
| `+` | **一个或多个** `{`（允许 `{` 或 `{{` 或 `{{{`） |
| `[^{}]*` | **零个或多个**非花括号字符 |
| `}` | 匹配右花括号 |
| `+` | **一个或多个** `}`（允许 `}` 或 `}}` 或 `}}}`） |

### 6.3 匹配示例

| 模板文本 | 是否匹配 | 说明 |
|---------|---------|------|
| `{name}` | ✅ | 标准变量 |
| `{user:key}` | ✅ | 带冒号的变量 |
| `{artifact.data}` | ✅ | Artifact 引用 |
| `{{name}}` | ✅ | 双花括号（允许） |
| `{temp:counter?}` | ✅ | 可选变量 |
| `{invalid{key}` | ❌ | 内部有花括号 |
| `{nested {inner}}` | ❌ | 嵌套花括号 |

### 6.4 正则作用

1. **定位占位符**：找到所有 `{...}` 模式
2. **容错处理**：允许 `{{{{var}}}}` 这种写法
3. **边界清晰**：`[^{}]*` 确保不匹配嵌套结构

## 七、替换逻辑详解

### 7.1 替换流程图

```
模板字符串
    ↓
┌─────────────────────────────────────┐
│  正则匹配：找到所有 {var}            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  遍历每个匹配项                      │
│  ┌─────────────────────────────┐   │
│  │ 1. 提取变量名                 │   │
│  │ 2. 判断是否可选 (带 ?)        │   │
│  │ 3. 判断类型：artifact / state │   │
│  │ 4. 获取值                     │   │
│  │ 5. 返回替换文本               │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  重建字符串（跳过原占位符，插入新值）│
└─────────────────────────────────────┘
    ↓
处理后的指令字符串
```

### 7.2 字符串重建原理

```python
# 原始模板
template = "Hi {name}, count: {counter}"

# 期望结果
result = "Hi Alice, count: 10"

# 重建过程
result_list = []

# 匹配 {name}
result_list.append(template[0:4])      # "Hi {name" → "Hi "
result_list.append("Alice")          # 替换值
last_end = 9                          # 跳过 "{name}"

# 匹配 {counter}
result_list.append(template[9:18])   # "}, count: {" → "}, count: "
result_list.append("10")              # 替换值
last_end = 19                         # 跳过 "{counter}"

# 无更多匹配
result_list.append(template[19:])     # "}" → "}"

# 列表内容
# ["Hi ", "Alice", ", count: ", "10"]

# 最终拼接
''.join(result_list)  # "Hi Alice, count: 10"
```

### 7.3 _replace_match 逻辑

```python
async def _replace_match(match) -> str:
    # 1. 提取变量名
    var_name = match.group().lstrip('{').rstrip('}').strip()
    # "{  user_name  }" → "user_name"

    # 2. 检查可选标记
    optional = False
    if var_name.endswith('?'):
        optional = True
        var_name = var_name.removesuffix('?')

    # 3. 分类处理
    if var_name.startswith('artifact.'):
        # → Artifact 分支
        return await _replace_artifact(var_name)
    else:
        # → State 分支
        return _replace_state(var_name, optional, match)
```

### 7.4 State 变量获取详细流程

```python
def _replace_state(var_name, optional, original_match):
    # 1. 验证变量名合法性
    if not _is_valid_state_name(var_name):
        return original_match.group()  # 不合法：保持原样

    # 2. 从 session.state 获取值
    if var_name in invocation_context.session.state:
        value = invocation_context.session.state[var_name]

        # 3. 值转换
        if value is None:
            return ''           # None → 空字符串
        return str(value)       # 其他 → 字符串

    # 4. 处理变量不存在的情况
    else:
        if optional:
            logger.debug('...not found, replacing with empty string')
            return ''           # 可选变量不存在 → 空字符串
        else:
            raise KeyError(f'Context variable not found: `{var_name}`')
```

### 7.5 _is_valid_state_name 验证逻辑

```python
def _is_valid_state_name(var_name):
    """验证变量名是否合法"""
    parts = var_name.split(':')

    # 普通变量：必须是合法标识符
    if len(parts) == 1:
        return var_name.isidentifier()

    # 带前缀变量：检查前缀和标识符
    if len(parts) == 2:
        prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
        # prefixes = ["app:", "user:", "temp:"]
        if (parts[0] + ':') in prefixes:
            return parts[1].isidentifier()
    return False
```

| 变量名 | 验证结果 | 说明 |
|-------|---------|------|
| `name` | ✅ | 普通变量，合法标识符 |
| `user_name` | ✅ | 包含下划线 |
| `user-name` | ❌ | 包含连字符，不是标识符 |
| `app:theme` | ✅ | app: 前缀 + 合法标识符 |
| `user:preference` | ✅ | user: 前缀 + 合法标识符 |
| `temp:counter` | ✅ | temp: 前缀 + 合法标识符 |
| `invalid:key-name` | ❌ | key-name 包含连字符 |
| `:` | ❌ | 缺少前缀或标识符 |

## 八、完整示例

### 8.1 示例场景

```python
# Session State
session.state = {
    "name": "Alice",
    "app:theme": "dark",
    "user:preference": "quiet_mode",
    "temp:counter": 42,
}

# Artifact
# 文件 "config.json" 内容: {"version": "1.0"}
```

### 8.2 指令模板

```python
template = """
You are {name}.

System Settings:
- Theme: {app:theme}
- Audio: {user:preference}
- Debug: {temp:counter?}
- Config: {artifact.config.json}
- Optional: {missing?}
"""
```

### 8.3 替换过程

| 占位符 | 变量名 | 值来源 | 替换结果 |
|-------|-------|--------|---------|
| `{name}` | `name` | `state["name"]` | `Alice` |
| `{app:theme}` | `app:theme` | `state["app:theme"]` | `dark` |
| `{user:preference}` | `user:preference` | `state["user:preference"]` | `quiet_mode` |
| `{temp:counter?}` | `temp:counter` | `state["temp:counter"]` | `42` |
| `{artifact.config.json}` | `config.json` | Artifact | `{"version": "1.0"}` |
| `{missing?}` | `missing` | 不存在（可选） | `""` |

### 8.4 最终输出

```python
"""
You are Alice.

System Settings:
- Theme: dark
- Audio: quiet_mode
- Debug: 42
- Config: {"version": "1.0"}
- Optional:
"""
```

## 九、错误处理

| 场景 | 处理方式 | 示例 |
|------|---------|------|
| 必选变量不存在 | 抛 `KeyError` | `{missing}` → `KeyError` |
| 可选变量不存在 | 返回空字符串 | `{missing?}` → `""` |
| 值为 `None` | 返回空字符串 | `state["key"]=None` → `""` |
| 非法变量名 | 保持原样 | `{invalid-key}` → `{invalid-key}` |
| Artifact 不存在（非可选） | 抛 `KeyError` | `{artifact.missing}` → `KeyError` |
| Artifact 服务未初始化 | 抛 `ValueError` | `artifact_service is None` |

## 十、集成到 LLM Request

```python
# 在 _InstructionsLlmRequestProcessor.run_async() 中

async def run_async(self, invocation_context, llm_request):
    # 1. 处理 global_instruction (已废弃)
    if isinstance(root_agent, LlmAgent) and root_agent.global_instruction:
        raw_si, bypass_state_injection = (
            await root_agent.canonical_global_instruction(
                ReadonlyContext(invocation_context)
            )
        )
        si = raw_si
        if not bypass_state_injection:
            si = await instructions_utils.inject_session_state(
                raw_si, ReadonlyContext(invocation_context)
            )
        llm_request.append_instructions([si])

    # 2. 处理 static_instruction
    if agent.static_instruction:
        static_content = _transformers.t_content(agent.static_instruction)
        llm_request.append_instructions(static_content)

    # 3. 处理 dynamic instruction
    if agent.instruction and not agent.static_instruction:
        si = await self._process_agent_instruction(agent, invocation_context)
        llm_request.append_instructions([si])
    elif agent.instruction and agent.static_instruction:
        si = await self._process_agent_instruction(agent, invocation_context)
        dynamic_content = types.Content(role='user', parts=[types.Part(text=si)])
        llm_request.contents.append(dynamic_content)
```

## 十一、最佳实践

1. **使用可选变量**：对于可能不存在的变量使用 `?` 后缀，避免运行时错误

   ```python
   # 推荐
   instruction = "Your name is {user_name?}, "
   ```

2. **使用前缀区分作用域**：使用 `app:`、`user:`、`temp:` 避免命名冲突

   ```python
   session.state = {
       "app:theme": "dark",
       "user:theme": "light",  # 不会与 app:theme 冲突
   }
   ```

3. **Artifact 存放大数据**：大文件（如 JSON 配置）用 Artifact，避免膨胀 state

   ```python
   # 推荐：大型配置用 Artifact
   instruction = "Config: {artifact.config.json}"

   # 不推荐：大型配置存 state
   session.state["config"] = {"large": "object..."}
   ```

4. **验证变量名**：使用合法标识符（字母、数字、下划线）

   ```python
   # 推荐
   {user_name} ✓
   {app_config} ✓

   # 不推荐
   {user-name} ✗ (连字符非法)
   {123number} ✗ (数字开头非法)
   ```

## 十二、代码位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `src/google/adk/flows/llm_flows/instructions.py` | 34-112 | `_InstructionsLlmRequestProcessor` 类 |
| `src/google/adk/utils/instructions_utils.py` | 30-124 | `inject_session_state` 函数 |

## 十三、相关类

- `BaseLlmRequestProcessor`：基类
- `ReadonlyContext`：只读上下文包装器
- `InvocationContext`：调用上下文
- `State`：状态管理类，定义前缀常量
