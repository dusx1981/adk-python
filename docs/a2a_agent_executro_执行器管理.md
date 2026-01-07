## 场景1：部署模式差异

### 单实例部署（直接传入 Runner）
```python
# 单机/单进程部署，Runner 是单例
from google.adk.runners import Runner

# 应用启动时创建单例 Runner
runner_instance = Runner(
    app_name="customer-support",
    agent=agent,
    session_service=session_service,
    # 大量预初始化配置...
    model_config={"model": "gemini-2.0-pro", "temperature": 0.7},
    tools=[database_tool, api_tool, cache_tool],
    memory_service=redis_memory_service,
    plugins=[logging_plugin, metrics_plugin]
)

# 多个 Executor 共享同一个 Runner
executor1 = A2aAgentExecutor(runner=runner_instance)  # 分支1：直接实例
executor2 = A2aAgentExecutor(runner=runner_instance)  # 共享相同实例

# 优点：
# 1. 资源复用：所有执行器共享同一个 Runner，节省内存
# 2. 性能优化：避免了重复初始化开销
# 3. 状态共享：会话、缓存等可以共享
```

### 多实例/多租户部署（传入 Callable）
```python
# 多租户 SaaS 服务，每个租户需要独立的 Runner
def create_tenant_runner(tenant_id: str) -> Runner:
    """为每个租户创建独立的 Runner"""
    tenant_config = get_tenant_config(tenant_id)
    
    return Runner(
        app_name=f"customer-support-{tenant_id}",
        agent=create_tenant_agent(tenant_config),
        session_service=TenantSessionService(tenant_id),
        model_config=tenant_config["model_settings"],
        # 每个租户独立的工具配置
        tools=get_tenant_tools(tenant_id),
        # 隔离的存储
        memory_service=TenantMemoryService(tenant_id)
    )

# 根据请求动态创建 Runner
async def handle_request(tenant_id: str, request):
    # 传入工厂函数，而不是实例
    executor = A2aAgentExecutor(
        runner=lambda: create_tenant_runner(tenant_id)  # 分支2：callable
    )
    
    # 第一次调用时创建 Runner
    result = await executor.execute(request)
    return result

# 优点：
# 1. 租户隔离：每个租户有独立的 Runner，避免数据混用
# 2. 动态配置：根据请求参数决定配置
# 3. 懒加载：只有实际使用时才创建
```

## 场景2：资源管理需求

### 长期运行服务（直接实例）
```python
# 长期运行的微服务，Runner 生命周期与应用一致
class AgentService:
    def __init__(self):
        # 启动时初始化，长期持有
        self.runner = Runner(
            app_name="long-running-service",
            agent=agent,
            # 昂贵的初始化
            session_service=init_database_connection(),
            memory_service=init_redis_pool(),
            # 预加载模型
            model_config=preload_model()
        )
        self.executor = A2aAgentExecutor(runner=self.runner)
    
    async def handle_requests(self):
        # 处理大量请求，重复使用同一个 Runner
        for request in request_stream:
            await self.executor.execute(request)
```

### 按需创建服务（Callable 工厂）
```python
# 无服务器函数（如 Cloud Functions），每次调用独立
def http_agent_handler(request):
    """HTTP 处理函数，每次调用可能创建新的 Runner"""
    
    def create_ephemeral_runner():
        """创建临时 Runner，执行后立即销毁"""
        return Runner(
            app_name="ephemeral-runner",
            agent=create_agent(),
            # 使用轻量级、无状态服务
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
            # 不进行昂贵初始化
            model_config={"model": "gemini-flash"}  # 快速模型
        )
    
    # 每次调用都创建新的 Executor
    executor = A2aAgentExecutor(runner=create_ephemeral_runner)
    return executor.execute(request)

# 优点：
# 1. 冷启动友好：只在需要时初始化
# 2. 资源释放：执行后 Runner 可被垃圾回收
# 3. 无状态设计：适合无服务器架构
```

## 场景3：配置动态性

### 静态配置（直接实例）
```python
# 配置在启动时确定，运行时不变
import os

# 从环境变量读取一次配置
model_name = os.getenv("MODEL_NAME", "gemini-1.5-pro")
api_key = os.getenv("API_KEY")

# 创建固定配置的 Runner
static_runner = Runner(
    app_name="static-config-runner",
    agent=agent,
    model_config={"model": model_name, "api_key": api_key},
    # 其他固定配置...
)

executor = A2aAgentExecutor(runner=static_runner)  # 分支1
```

### 动态配置（Callable 工厂）
```python
# 运行时动态获取配置
async def create_dynamic_runner() -> Runner:
    """从配置中心动态获取最新配置"""
    
    # 每次调用都获取最新配置
    config = await config_client.get_latest_config("agent-config")
    
    # 可能根据负载调整
    current_load = get_system_load()
    if current_load > 80:
        config["timeout"] = min(config.get("timeout", 30), 10)
    
    return Runner(
        app_name=config["app_name"],
        agent=create_agent_with_config(config),
        model_config=config["model"],
        # 动态设置超时
        timeout=config["timeout"]
    )

# Executor 使用工厂函数
executor = A2aAgentExecutor(runner=create_dynamic_runner)  # 分支2

# 优点：
# 1. 热重载：配置更新无需重启
# 2. 环境感知：根据运行时条件调整
# 3. A/B 测试：可动态切换不同配置
```

## 场景4：测试策略差异

### 集成测试（直接实例）
```python
# 集成测试使用真实的、预配置的 Runner
import pytest

@pytest.fixture
def runner_instance():
    """测试 fixture：预创建 Runner"""
    return Runner(
        app_name="test-runner",
        agent=test_agent,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService()
    )

@pytest.fixture
def executor(runner_instance):
    """使用预创建的 Runner"""
    return A2aAgentExecutor(runner=runner_instance)  # 分支1

async def test_integration(executor):
    """集成测试：使用真实 Runner"""
    result = await executor.execute(test_request)
    assert result.status == "completed"
```

### 单元测试（Callable 工厂）
```python
# 单元测试使用 Mock Runner
from unittest.mock import Mock, AsyncMock

def test_executor_logic():
    """测试 Executor 逻辑，不依赖真实 Runner"""
    
    # Mock 的 Runner 工厂
    mock_runner = Mock(spec=Runner)
    mock_runner.run_async = AsyncMock(return_value=async_mock_generator())
    
    def mock_factory():
        return mock_runner
    
    # 传入工厂函数
    executor = A2aAgentExecutor(runner=mock_factory)  # 分支2
    
    # 测试 Executor 逻辑，而不是 Runner
    result = executor.execute(test_context, event_queue)
    
    # 验证交互
    mock_runner.run_async.assert_called_once()

# 优点：
# 1. 隔离测试：不依赖外部服务
# 2. 快速执行：避免真实初始化
# 3. 精确控制：Mock 特定行为
```

## 场景5：资源初始化复杂性

### 简单初始化（直接实例）
```python
# 初始化简单快速，可以直接创建
simple_runner = Runner(
    app_name="simple-runner",
    agent=agent,
    session_service=InMemorySessionService(),  # 内存存储，快速
    # 没有外部依赖
)

executor = A2aAgentExecutor(runner=simple_runner)  # 分支1
```

### 复杂初始化（Callable 异步工厂）
```python
# 初始化需要异步操作和外部依赖
async def create_complex_runner() -> Runner:
    """复杂初始化，需要异步操作"""
    
    # 1. 异步获取配置
    config = await config_service.get_async("agent-config")
    
    # 2. 初始化数据库连接
    db_pool = await create_database_pool(config["database"])
    
    # 3. 加载外部资源
    vector_store = await load_vector_store(config["embeddings"])
    
    # 4. 预热模型
    await warmup_model(config["model_name"])
    
    return Runner(
        app_name="complex-runner",
        agent=agent,
        session_service=DatabaseSessionService(db_pool),
        tools=[vector_store, database_tool],
        model_config=config["model"]
    )

# 必须使用工厂函数，因为初始化是异步的
executor = A2aAgentExecutor(runner=create_complex_runner)  # 分支2

# 优点：
# 1. 支持异步初始化
# 2. 处理复杂依赖关系
# 3. 错误处理更灵活
```

## 性能对比分析

### 内存使用对比
```python
# 测试两种方式的内存使用
import tracemalloc

# 方式1：直接实例（预创建）
tracemalloc.start()
runner = Runner(...)  # 立即占用内存
executor1 = A2aAgentExecutor(runner=runner)
snapshot1 = tracemalloc.take_snapshot()
print(f"直接实例内存: {snapshot1.statistics('lineno')[:3]}")

# 方式2：工厂函数（延迟创建）
def factory():
    return Runner(...)  # 尚未创建

executor2 = A2aAgentExecutor(runner=factory)
snapshot2 = tracemalloc.take_snapshot()
print(f"工厂函数内存: {snapshot2.statistics('lineno')[:3]}")

# 首次调用时才创建
await executor2._resolve_runner()  # 此时才创建 Runner
snapshot3 = tracemalloc.take_snapshot()
print(f"首次调用后内存: {snapshot3.statistics('lineno')[:3]}")
```

### 执行时间对比
```python
import time

# 场景：冷启动 vs 热启动
def test_performance():
    # 方式1：直接实例（热启动快）
    runner = Runner(...)  # 启动时初始化，耗时 T1
    executor1 = A2aAgentExecutor(runner=runner)
    
    start = time.time()
    await executor1.execute(request)  # 执行快，无需初始化
    time1 = time.time() - start
    
    # 方式2：工厂函数（冷启动慢）
    def factory():
        return Runner(...)  # 每次创建都初始化
    
    executor2 = A2aAgentExecutor(runner=factory)
    
    start = time.time()
    await executor2.execute(request)  # 包含初始化时间 T1
    time2 = time.time() - start
    
    print(f"直接实例执行时间: {time1:.3f}s")
    print(f"工厂函数执行时间: {time2:.3f}s")
    print(f"初始化开销: {time2 - time1:.3f}s")
```

## 架构设计总结

### 为什么需要区分？

1. **生命周期管理**：
   ```python
   # 直接实例：与应用同生命周期
   # 工厂函数：按需创建和销毁
   ```

2. **资源优化**：
   ```python
   # 高频调用：直接实例避免重复初始化
   # 低频调用：工厂函数节省空闲资源
   ```

3. **配置灵活性**：
   ```python
   # 固定环境：直接实例简化部署
   # 多变环境：工厂函数动态适应
   ```

4. **测试友好性**：
   ```python
   # 集成测试：直接实例确保真实性
   # 单元测试：工厂函数便于 Mock
   ```

5. **错误隔离**：
   ```python
   # 直接实例：初始化失败整个应用启动失败
   # 工厂函数：初始化失败只影响单个请求
   ```

### 决策树：如何选择？

```
是否需要在运行时动态配置？
├── 是 → 使用工厂函数（Callable）
│   ├── 是否需要异步初始化？
│   │   ├── 是 → 异步工厂函数
│   │   └── 否 → 同步工厂函数
│   └── 是否需要每次调用独立实例？
│       ├── 是 → 工厂函数返回新实例
│       └── 否 → 工厂函数+缓存
└── 否 → 直接使用 Runner 实例
    ├── 是否是长期运行服务？
    │   ├── 是 → 直接实例（性能优先）
    │   └── 否 → 考虑工厂函数（资源节省）
    └── 是否需要共享状态？
        ├── 是 → 直接实例（状态共享）
        └── 否 → 工厂函数（状态隔离）
```

这种设计的核心思想是**分离关注点**：
- `A2aAgentExecutor` 只关心如何**使用** Runner
- 不关心 Runner 如何**创建**或**配置**
- 将创建逻辑委托给调用者，提供最大灵活性

这正是**依赖注入**和**控制反转**原则的体现：
- 高层模块（A2aAgentExecutor）不依赖低层模块（Runner）的具体创建方式
- 抽象出接口（Callable），让调用者决定具体实现
- 提高了框架的可测试性、可扩展性和可维护性