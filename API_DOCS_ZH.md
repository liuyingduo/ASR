# 语音识别 (ASR) 接口文档

本文档描述了语音识别服务的 WebSocket 实时接口和 HTTP 批量处理接口。

## 1. 实时语音识别 WebSocket 接口

**接口地址**: `/ws/asr`
**协议**: WebSocket

### 功能说明
该接口用于实时语音流的识别。客户端通过 WebSocket 连接发送音频数据，服务端会实时返回识别出的文本。

### 请求流程
1. **建立连接**: 客户端连接到 `ws://<host>:<port>/ws/asr`。
2. **发送数据**: 客户端持续发送音频数据的二进制流。
   - **格式**: Int16 PCM
   - **采样率**: 16000 Hz
   - **声道**: 单声道
3. **接收结果**: 服务端每接收到一定量的音频数据（约1秒），会进行一次推理，并通过 WebSocket 发送 JSON 格式的识别结果。

### 返回数据格式
服务端返回的数据为 JSON 对象：

```json
{
  "text": "识别出的文本内容"
}
```

### 示例代码 (Python)
```python
import asyncio
import websockets
import json

async def hello():
    uri = "ws://localhost:50109/ws/asr"
    async with websockets.connect(uri) as websocket:
        # 假设 audio_chunk 是读取到的音频二进制数据
        await websocket.send(audio_chunk)
        
        result = await websocket.recv()
        print(json.loads(result))

asyncio.get_event_loop().run_until_complete(hello())
```

---

## 2. 音频文件转文本 HTTP 接口

**接口地址**: `/api/v1/asr`
**方法**: `POST`
**Content-Type**: `multipart/form-data`

### 功能说明
该接口用于上传音频文件进行批量识别。支持一次上传多个文件。

### 请求参数

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `files` | List[File] | 是 | 音频文件列表。支持 wav 或 mp3 格式，建议采样率 16KHz。 |
| `keys` | String | 否 | 每个音频对应的唯一标识符（ID），多个 ID 用逗号 `,` 分隔。如果不传，默认使用文件名。 |
| `lang` | String | 否 | 音频内容的语言。默认为 `auto`。 |

#### 支持的语言 (`lang`)
- `auto`: 自动检测 (默认)
- `zh`: 中文
- `en`: 英文
- `yue`: 粤语
- `ja`: 日语
- `ko`: 韩语
- `nospeech`: 无语音

### 响应数据格式
返回一个 JSON 对象，包含识别结果列表。

```json
{
  "result": [
    {
      "key": "audio_filename_or_id",
      "text": "最终处理后的识别文本",
      "raw_text": "原始识别文本",
      "clean_text": "去除特殊标记后的文本"
    },
    ...
  ]
}
```

### 示例请求 (curl)

```bash
curl -X 'POST' \
  'http://localhost:50109/api/v1/asr' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@test_audio_1.wav;type=audio/wav' \
  -F 'files=@test_audio_2.mp3;type=audio/mpeg' \
  -F 'lang=zh'
```
