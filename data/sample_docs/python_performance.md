# Python 性能优化实战

## 性能分析工具

在优化之前，必须先定位瓶颈。Python 常用的性能分析工具：

### cProfile（标准库）
```python
import cProfile
cProfile.run('main()', sort='cumulative')
```
输出函数调用次数、总耗时、每次调用耗时，按累计时间排序。

### line_profiler（逐行分析）
```python
@profile
def slow_function():
    result = [x**2 for x in range(1000000)]
    return sum(result)
```
精确到每一行的执行时间和调用次数。

### py-spy（采样式分析器）
无需修改代码，直接 attach 到运行中的进程：
```bash
py-spy top --pid 12345
py-spy record -o flamegraph.svg -- python app.py
```
生成火焰图，直观展示 CPU 时间分布。

## 常见优化策略

### 1. 数据结构选择
- 查找操作用 `set`/`dict`（O(1)），不要用 `list`（O(n)）
- 频繁插入删除用 `collections.deque`
- 计数用 `collections.Counter`
- 多键索引用 `collections.defaultdict`

### 2. 生成器替代列表
```python
# 差：一次性加载所有数据到内存
squares = [x**2 for x in range(10000000)]

# 好：惰性求值，按需生成
squares = (x**2 for x in range(10000000))
```
内存占用从 GB 级降到 KB 级。

### 3. 避免全局变量查找
局部变量查找比全局变量快 20-30%：
```python
def process(items):
    append = items.append  # 局部引用
    for i in range(1000):
        append(i)
```

### 4. 使用内置函数
内置函数用 C 实现，比纯 Python 循环快：
- `sum()` 比手动累加快
- `map()` 比 for 循环 + 函数调用快
- `str.join()` 比字符串拼接快

### 5. 缓存（Memoization）
```python
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## 并发与并行

### GIL 的影响
CPython 的 GIL（全局解释器锁）限制了多线程的 CPU 并行能力：
- **I/O 密集型**：用 `asyncio` 或 `threading`（GIL 在 I/O 等待时释放）
- **CPU 密集型**：用 `multiprocessing` 或 C 扩展

### asyncio 异步编程
```python
import asyncio

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

async def main():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
```
单线程处理数万并发连接，适合 Web 服务器、爬虫。

### multiprocessing 多进程
```python
from multiprocessing import Pool

def cpu_heavy(n):
    return sum(i*i for i in range(n))

with Pool(4) as p:
    results = p.map(cpu_heavy, [10**7] * 4)
```

## 第三方加速库

| 库 | 用途 | 加速比 |
|---|---|---|
| NumPy | 数值计算（向量化） | 10-100x |
| Numba | JIT 编译数值函数 | 10-1000x |
| Cython | C 扩展编译 | 5-100x |
| PyPy | 替代解释器 | 2-10x |
| polars | DataFrame（替代 pandas） | 5-30x |

## 数据库查询优化

ORM 常见性能陷阱：
1. **N+1 查询**：循环中逐条查询 → 用 `select_related` / `prefetch_related`
2. **未加索引**：对 WHERE/ORDER BY 字段建索引
3. **SELECT ***：只查需要的字段
4. **大事务**：拆分为小批次提交

## 内存优化

- 使用 `__slots__` 减少对象内存占用（节省 40-50%）
- 用 `array.array` 替代 `list` 存储同类型数值
- 大数据集用 `pandas` 的 `category` 类型
- 及时 `del` 不再使用的大对象，或依赖 GC
