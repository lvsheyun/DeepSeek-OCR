# DeepSeek OCR HTTP Service

基于 DeepSeek-OCR 和 vLLM 的 HTTP OCR 服务，支持图片和 PDF 文档的 OCR 识别。

## 功能特性

- ✅ FastAPI 异步服务，支持高并发
- ✅ 任务队列系统，防止 GPU OOM  
- ✅ 智能批处理，提升 GPU 利用率
- ✅ 支持文件上传和 URL 两种方式
- ✅ 异步任务模式，通过 task_id 查询结果
- ✅ 自动任务清理（TTL 机制）
- ✅ 可自定义 prompt 和采样参数

## 架构说明

### OOM 防护机制

服务通过以下机制防止 GPU 内存溢出：

1. **队列大小限制**：设置 `MAX_QUEUE_SIZE`（默认 100），队列满时返回 503
2. **批处理控制**：通过 `BATCH_SIZE` 控制同时处理的图片数量
3. **智能批处理**：收集任务直到达到 `BATCH_SIZE` 或 `BATCH_TIMEOUT`（秒）

### 批处理策略

服务采用混合批处理策略以平衡延迟和吞吐量：

```
等待条件：收集到 BATCH_SIZE 个任务 OR 等待超过 BATCH_TIMEOUT 秒
```

- **BATCH_SIZE**：影响 GPU 利用率和内存使用
- **BATCH_TIMEOUT**：影响单个请求的最大延迟

## 安装

### 1. 安装依赖

```bash
pip install -r requirements_service.txt
```

### 2. 确定最优 BATCH_SIZE（重要！）

不同 GPU 显存支持的最大 batch size 不同。运行测试脚本找到最优配置：

```bash
python test_batch_size.py --test-image assets/show1.jpg
```

脚本会测试不同 batch size，输出类似：

```
RECOMMENDATIONS
============================================================
For your GPU (NVIDIA RTX 4090):
  Maximum safe batch size: 16
  Recommended batch size: 8
    (Using 50% of max for safety margin and concurrent requests)

Update your .env file:
  OCR_BATCH_SIZE=8
  OCR_MAX_QUEUE_SIZE=100
```

### 3. 配置环境变量（可选）

创建 `.env` 文件：

```bash
# Model Configuration
OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR

# GPU Settings
OCR_GPU_MEMORY_UTIL=0.8
OCR_MAX_MODEL_LEN=8192
OCR_TENSOR_PARALLEL_SIZE=1

# Service Configuration（根据 test_batch_size.py 结果调整）
OCR_BATCH_SIZE=8              # 从测试脚本获取
OCR_MAX_QUEUE_SIZE=100        # 可根据需求调整
OCR_BATCH_TIMEOUT=2.0         # 批处理最大等待时间（秒）
OCR_TASK_TTL=3600             # 任务结果保留时间（秒）
OCR_CLEANUP_INTERVAL=300      # 清理间隔（秒）

# Image Settings
OCR_MAX_IMAGE_SIZE_MB=20
OCR_IMAGE_DOWNLOAD_TIMEOUT=30

# Default OCR Settings
OCR_DEFAULT_PROMPT=<image>\n<|grounding|>Convert the document to markdown.
OCR_DEFAULT_TEMPERATURE=0.0
OCR_DEFAULT_MAX_TOKENS=8192
OCR_NGRAM_SIZE=30
OCR_WINDOW_SIZE=90

# Server Settings
OCR_HOST=0.0.0.0
OCR_PORT=8000
OCR_LOG_LEVEL=INFO
```

## 运行服务

```bash
python run_service.py
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 统计信息：http://localhost:8000/stats

## API 使用

### 1. 上传图片文件

```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=<image>\nFree OCR."
```

响应：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### 2. 提交图片 URL

```bash
curl -X POST "http://localhost:8000/ocr/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/image.jpg",
    "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
    "temperature": 0.0,
    "max_tokens": 8192
  }'
```

响应：同上

### 3. 批量提交图片 URL

```bash
curl -X POST "http://localhost:8000/ocr/url/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg",
      "https://example.com/image3.jpg"
    ],
    "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
    "temperature": 0.0,
    "max_tokens": 8192
  }'
```

响应：
```json
{
  "batch_task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "pending"
}
```

### 4. 查询批量任务结果

```bash
curl "http://localhost:8000/ocr/batch/550e8400-e29b-41d4-a716-446655440001"
```

响应（处理中）：
```json
{
  "batch_task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "processing",
  "total": 3,
  "completed": 1,
  "failed": 0,
  "pending": 1,
  "processing": 1,
  "results": [
    {
      "url": "https://example.com/image1.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440002",
      "status": "completed",
      "result": "# Document Title\n\nDocument content..."
    },
    {
      "url": "https://example.com/image2.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440003",
      "status": "processing"
    },
    {
      "url": "https://example.com/image3.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440004",
      "status": "pending"
    }
  ],
  "created_at": 1699000000.0,
  "updated_at": 1699000001.0
}
```

响应（完成）：
```json
{
  "batch_task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "total": 3,
  "completed": 3,
  "failed": 0,
  "pending": 0,
  "processing": 0,
  "results": [
    {
      "url": "https://example.com/image1.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440002",
      "status": "completed",
      "result": "# Document 1\n\nContent 1..."
    },
    {
      "url": "https://example.com/image2.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440003",
      "status": "completed",
      "result": "# Document 2\n\nContent 2..."
    },
    {
      "url": "https://example.com/image3.jpg",
      "task_id": "550e8400-e29b-41d4-a716-446655440004",
      "status": "completed",
      "result": "# Document 3\n\nContent 3..."
    }
  ],
  "created_at": 1699000000.0,
  "updated_at": 1699000005.0
}
```

**注意**：批量 API 允许超出配置的 `MAX_QUEUE_SIZE` 限制，以便将分页工作负载保持在一起。每个 URL 会独立验证，部分失败不会影响其他任务。

### 5. 查询单个任务结果

```bash
curl "http://localhost:8000/ocr/550e8400-e29b-41d4-a716-446655440000"
```

响应（处理中）：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "created_at": 1699000000.0,
  "updated_at": 1699000001.0
}
```

响应（完成）：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": 1699000000.0,
  "updated_at": 1699000005.0,
  "result": "# Document Title\n\nDocument content..."
}
```

### 6. 队列满时的响应

```json
{
  "detail": "Service is at full capacity. Please try again later."
}
```

HTTP 状态码：503 Service Unavailable

### 7. 查看服务统计

```bash
curl "http://localhost:8000/stats"
```

响应：
```json
{
  "total": 45,
  "pending": 3,
  "processing": 2,
  "completed": 38,
  "failed": 2,
  "queue_limit": 100
}
```

## 常用 Prompt

```python
# 文档转 Markdown（带布局信息）
"<image>\n<|grounding|>Convert the document to markdown."

# 纯文本 OCR（无布局）
"<image>\nFree OCR."

# 其他图片 OCR
"<image>\n<|grounding|>OCR this image."

# 解析图表
"<image>\nParse the figure."

# 详细描述图片
"<image>\nDescribe this image in detail."
```

## 图像预处理参数说明

**重要提示**：以下参数当前是**硬编码**在 vLLM (v0.6.0+) 内部的，无法通过配置修改：

- `BASE_SIZE = 1024`：全局视图大小
- `IMAGE_SIZE = 640`：局部裁剪块大小  
- `CROP_MODE = True`：启用动态裁剪
- `MIN_CROPS = 2`：最小裁剪数量
- `MAX_CROPS = 6`：最大裁剪数量（最大可设为 9，但显存有限建议 6）

当前默认使用 **Gundam 模式**（高精度 + 高效率平衡）。

vLLM 支持的其他模式（需要修改 vLLM 源码）：
- Tiny: `base_size=512, image_size=512, crop_mode=False`
- Small: `base_size=640, image_size=640, crop_mode=False`  
- Base: `base_size=1024, image_size=1024, crop_mode=False`
- Large: `base_size=1280, image_size=1280, crop_mode=False`

## 性能优化建议

### GPU 显存不足

1. 降低 `BATCH_SIZE`（使用 `test_batch_size.py` 确定最优值）
2. 降低 `OCR_GPU_MEMORY_UTIL`（如 0.7）
3. 降低 `MAX_CROPS`（需修改 vLLM 源码）

### 提升吞吐量

1. 增加 `BATCH_SIZE`（在显存允许的情况下）
2. 调整 `BATCH_TIMEOUT`（更长的等待收集更多任务）
3. 增加 `MAX_QUEUE_SIZE`

### 降低延迟

1. 降低 `BATCH_SIZE`（如设为 1，但会降低吞吐量）
2. 降低 `BATCH_TIMEOUT`（如 0.5 秒）

## PDF 处理建议

PDF 通常有多页，单个 PDF 可能包含几百页图片。**建议架构**：

1. **外部服务**负责将 PDF 拆分为单页图片
2. **本服务**只处理单张图片 OCR
3. 外部服务通过 HTTP 状态码判断：
   - `202 Accepted`：任务已创建
   - `503 Service Unavailable`：队列满，稍后重试

**示例处理流程**：

```python
import httpx
import asyncio
from pdf2image import convert_from_path

async def process_pdf(pdf_path, ocr_service_url):
    # 1. 拆分 PDF 为图片
    images = convert_from_path(pdf_path)
    
    # 2. 提交所有图片任务
    task_ids = []
    async with httpx.AsyncClient() as client:
        for i, image in enumerate(images):
            # 保存临时图片
            image_path = f"/tmp/page_{i}.jpg"
            image.save(image_path)
            
            # 提交任务，如果队列满则等待重试
            while True:
                with open(image_path, "rb") as f:
                    response = await client.post(
                        f"{ocr_service_url}/ocr",
                        files={"file": f}
                    )
                
                if response.status_code == 202:
                    task_ids.append(response.json()["task_id"])
                    break
                elif response.status_code == 503:
                    # 队列满，等待后重试
                    await asyncio.sleep(5)
                else:
                    raise Exception(f"Error: {response.text}")
    
    # 3. 轮询结果
    results = []
    async with httpx.AsyncClient() as client:
        for task_id in task_ids:
            while True:
                response = await client.get(f"{ocr_service_url}/ocr/{task_id}")
                data = response.json()
                
                if data["status"] == "completed":
                    results.append(data["result"])
                    break
                elif data["status"] == "failed":
                    results.append(f"Error: {data['error']}")
                    break
                
                await asyncio.sleep(1)
    
    return results
```

## 故障排查

### 服务启动失败

1. 检查 GPU 是否可用：`nvidia-smi`
2. 检查 vLLM 是否正确安装：`python -c "import vllm"`
3. 检查模型路径是否正确

### OOM 错误

1. 运行 `python test_batch_size.py` 重新确定最优配置
2. 降低 `BATCH_SIZE`
3. 降低 `GPU_MEMORY_UTILIZATION`

### 任务一直处于 pending 状态

1. 检查 worker 是否正常运行（查看日志）
2. 检查是否有其他任务阻塞
3. 重启服务

## 监控建议

定期检查 `/stats` 端点：

```bash
watch -n 5 'curl -s http://localhost:8000/stats | jq'
```

关注指标：
- `pending` 持续增加 → 处理速度跟不上，考虑增加 BATCH_SIZE
- `failed` 增加 → 检查日志排查错误原因
- `total` 接近 `queue_limit` → 队列快满，考虑增加 MAX_QUEUE_SIZE

## 许可证

本项目基于 DeepSeek-OCR 开发，遵循原项目许可证。

