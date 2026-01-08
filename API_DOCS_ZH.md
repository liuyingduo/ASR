# 语音识别 (ASR) 接口文档

本文档描述了语音识别服务的 WebSocket 实时接口和 HTTP 批量处理接口。

**服务地址**: `https://qingdao.zaiwenai.com:50109/`

## API 根路径
- **主页**: `https://qingdao.zaiwenai.com:50109/` - 提供API文档和演示页面的链接
- **API文档**: `https://qingdao.zaiwenai.com:50109/docs` - 自动生成的API文档页面
- **实时演示**: `https://qingdao.zaiwenai.com:50109/demo` - 实时语音识别演示页面

## 1. 实时语音识别 WebSocket 接口

**接口地址**: `/ws/asr`
**协议**: WebSocket
**完整地址**: `wss://qingdao.zaiwenai.com:50109/ws/asr`

### 功能说明
该接口用于实时语音流的识别。客户端通过 WebSocket 连接发送音频数据，服务端会实时返回识别出的文本。接口采用滑动窗口机制，能够智能锁定已确认的文本，提高实时识别的准确性。

### 请求流程
1. **建立连接**: 客户端连接到 `wss://qingdao.zaiwenai.com:50109/ws/asr`。
2. **发送数据**: 客户端持续发送音频数据的二进制流。
   - **格式**: Int16 PCM
   - **采样率**: 16000 Hz
   - **声道**: 单声道
3. **接收结果**: 服务端每接收到4个音频块（约1秒），会进行一次推理，并通过 WebSocket 发送 JSON 格式的识别结果。

### 滑动窗口机制
- **窗口大小**: 最大40个音频块（约10秒音频）
- **锁定阈值**: 当音频块超过24个（约6秒）时，前面的识别结果会被锁定
- **实时更新**: 当前窗口内的文本会持续更新，已锁定的文本不再变化

### 返回数据格式
服务端返回的数据为 JSON 对象：

```json
{
  "text": "完整的识别文本",
  "locked": "已确认不会再变的文本",
  "pending": "当前窗口内可能还会变化的文本"
}
```

### 示例代码 (Python)
```python
import asyncio
import websockets
import json

async def hello():
    uri = "wss://qingdao.zaiwenai.com:50109/ws/asr"
    async with websockets.connect(uri) as websocket:
        # 假设 audio_chunk 是读取到的音频二进制数据
        await websocket.send(audio_chunk)
        
        result = await websocket.recv()
        data = json.loads(result)
        print(f"完整文本: {data['text']}")
        print(f"已锁定文本: {data['locked']}")
        print(f"待定文本: {data['pending']}")

asyncio.get_event_loop().run_until_complete(hello())
```

---

## 2. 音频文件转文本 HTTP 接口

**接口地址**: `/api/v1/asr`
**方法**: `POST`
**Content-Type**: `multipart/form-data`
**完整地址**: `https://qingdao.zaiwenai.com:50109/api/v1/asr`

### 功能说明
该接口用于上传音频文件进行批量识别。支持一次上传多个文件，支持自动重采样到16KHz。

### 请求参数

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `files` | List[File] | 是 | 音频文件列表。支持 wav 或 mp3 格式，会自动重采样到 16KHz。 |
| `keys` | String | 否 | 每个音频对应的唯一标识符（ID），多个 ID 用逗号 `,` 分隔。如果不传，默认使用文件名。 |
| `lang` | String | 否 | 音频内容的语言。默认为 `auto`。支持 ITN（逆文本规范化）处理。 |

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
      "text": "最终处理后的识别文本（包含富文本处理）",
      "raw_text": "原始识别文本（包含特殊标记）",
      "clean_text": "去除特殊标记后的文本"
    },
    ...
  ]
}
```

### 示例请求 (curl)

```bash
curl -X 'POST' \
  'https://qingdao.zaiwenai.com:50109/api/v1/asr' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@test_audio_1.wav;type=audio/wav' \
  -F 'files=@test_audio_2.mp3;type=audio/mpeg' \
  -F 'lang=zh'
```
