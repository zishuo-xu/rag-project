# 数据库索引原理与优化

## B+ 树索引

大多数关系型数据库（MySQL InnoDB、PostgreSQL）使用 B+ 树作为默认索引结构。

### B+ 树的特点
- 所有数据存储在叶子节点，内部节点只存键值和指针
- 叶子节点通过双向链表连接，支持高效范围查询
- 树高通常为 3-4 层，意味着最多 3-4 次磁盘 I/O 即可定位任意记录
- 每个节点大小通常等于一个磁盘页（16KB for InnoDB）

### 聚簇索引 vs 二级索引
- **聚簇索引（Clustered Index）**：叶子节点存储完整行数据，每张表只能有一个（通常是主键）
- **二级索引（Secondary Index）**：叶子节点存储主键值，查询非索引列需要"回表"

### 覆盖索引
如果查询的列全部包含在索引中，无需回表：
```sql
-- 假设有联合索引 (name, age)
SELECT name, age FROM users WHERE name = 'Alice';
-- 直接从索引返回，无需访问数据页
```

## 联合索引与最左前缀

联合索引 (a, b, c) 的 B+ 树按 a → b → c 顺序排列：
- `WHERE a = 1` ✅ 使用索引
- `WHERE a = 1 AND b = 2` ✅ 使用索引
- `WHERE a = 1 AND b = 2 AND c = 3` ✅ 使用索引
- `WHERE b = 2` ❌ 无法使用（缺少最左列）
- `WHERE a = 1 AND c = 3` ⚠️ 只用到 a（b 缺失，c 无法利用）
- `WHERE a > 1 AND b = 2` ⚠️ a 用范围后，b 无法利用

## 索引失效的常见场景

1. **对索引列使用函数**：`WHERE YEAR(create_time) = 2024` → 改为范围查询
2. **隐式类型转换**：`WHERE varchar_col = 123`（数字 vs 字符串）
3. **LIKE 左模糊**：`WHERE name LIKE '%张'` 无法使用索引
4. **OR 条件**：`WHERE a = 1 OR b = 2`（除非 a、b 各有独立索引）
5. **NOT IN / NOT EXISTS**：通常导致全表扫描
6. **数据量小**：优化器判断全表扫描比走索引更快

## 哈希索引

- 等值查询 O(1)，但不支持范围查询和排序
- Memory 引擎默认使用哈希索引
- InnoDB 有自适应哈希索引（AHI），自动对热点页建立哈希

## 全文索引

用于文本搜索（替代 LIKE '%keyword%'）：
```sql
ALTER TABLE articles ADD FULLTEXT INDEX ft_content (content);
SELECT * FROM articles WHERE MATCH(content) AGAINST('数据库优化' IN NATURAL LANGUAGE MODE);
```
- MySQL 5.6+ 支持 InnoDB 全文索引
- 中文需要 ngram 分词插件
- 大规模场景推荐 Elasticsearch

## 向量索引（AI 时代）

随着大模型和 RAG 的兴起，向量数据库成为新趋势：

### 常见向量索引算法
- **Flat（暴力搜索）**：精确但慢，适合小数据集
- **IVF（倒排文件）**：聚类后只搜索相关簇，牺牲精度换速度
- **HNSW（分层可导航小世界图）**：查询速度快，内存占用大
- **PQ（乘积量化）**：压缩向量，减少内存，适合超大规模

### 主流向量数据库
| 数据库 | 特点 |
|--------|------|
| ChromaDB | 嵌入式，轻量，适合原型开发 |
| Milvus | 分布式，支持十亿级向量 |
| Pinecone | 全托管 SaaS |
| Weaviate | 内置混合搜索 |
| pgvector | PostgreSQL 扩展，无需新数据库 |
| FAISS | Meta 开源库，非独立数据库 |

### 相似度度量
- **余弦相似度**：衡量方向，忽略长度（最常用于文本）
- **欧氏距离（L2）**：衡量绝对距离
- **内积（IP）**：归一化后等价于余弦

## 查询优化实践

### EXPLAIN 执行计划
```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 100 AND status = 'paid';
```
关注字段：
- `type`：ALL（全表）→ index → range → ref → eq_ref → const（从差到好）
- `key`：实际使用的索引
- `rows`：预估扫描行数
- `Extra`：Using index（覆盖索引）、Using filesort（需优化排序）

### 慢查询日志
```ini
# my.cnf
slow_query_log = 1
long_query_time = 1  # 超过1秒记录
```
用 `mysqldumpslow` 或 `pt-query-digest` 分析 Top N 慢查询。

### 分页优化
```sql
-- 差：OFFSET 越大越慢
SELECT * FROM articles ORDER BY id LIMIT 10 OFFSET 100000;

-- 好：游标分页（记住上一页最后一条的 id）
SELECT * FROM articles WHERE id > 100000 ORDER BY id LIMIT 10;
```
