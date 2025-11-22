# ASR WebSocket API 文档

## 简介
本项目提供了一个基于 WebSocket 的实时语音转录接口，使用 Faster Whisper 模型进行语音识别。

## 基础 URL
`ws://<host>:<port>/ws/transcribe`

例如本地运行：`ws://localhost:8000/ws/transcribe`

## 接口详情

### 实时转录 WebSocket

**URL**: `/ws/transcribe`
**协议**: WebSocket

#### 连接流程
1. 客户端建立 WebSocket 连接。
2. 连接成功后，客户端开始发送音频数据。
3. 服务端接收音频数据并进行缓冲。
4. 服务端定期（或基于数据量）进行转录，并将当前的完整转录结果发送回客户端。

#### 输入数据格式
客户端应发送二进制消息（Binary Message）。
- **格式**: 32-bit Float PCM (Little Endian)
- **采样率**: 16000 Hz
- **声道**: 单声道 (Mono)

#### 输出数据格式
服务端发送文本消息（Text Message）。
- **内容**: 当前识别到的完整文本字符串。

#### 示例代码 (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

ws.onopen = () => {
    console.log('Connected');
    // 发送音频数据 (Float32Array)
    // ws.send(audioData);
};

ws.onmessage = (event) => {
    console.log('Transcription:', event.data);
};
```

## 运行说明

1. 安装依赖:
   ```bash
   pip install fastapi uvicorn websockets faster-whisper numpy
   ```

2. 启动服务:
   ```bash
   python server.py
   ```
   或者使用 uvicorn 命令:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

3. 访问前端:
   打开浏览器访问 `http://localhost:8000/` 即可使用简单的测试页面。
