# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
import torch
import numpy as np
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO

TARGET_FS = 16000


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
            <br>
            <a href='./demo'>Real-time ASR Demo</a>
        </body>
    </html>
    """


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = []
    
    # Parameters for inference frequency
    chunk_count = 0
    inference_interval = 4  # Run inference every 4 chunks (approx 1 sec)
    
    # 滑动窗口参数
    max_window_chunks = 40  # 最大窗口大小（约10秒音频）
    lock_threshold = 24     # 当超过此数量时，锁定前面的识别结果（约6秒）
    
    # 已锁定的文本（不再修改）
    locked_text = ""
    # 当前窗口的起始位置
    window_start = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            # Convert bytes to numpy int16
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            # Convert to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            audio_buffer.append(audio_float)
            
            chunk_count += 1
            
            if chunk_count % inference_interval == 0:
                current_buffer_len = len(audio_buffer)
                
                # 如果buffer超过最大窗口大小，需要锁定前面的部分
                if current_buffer_len > max_window_chunks:
                    # 计算需要锁定的音频块数量
                    chunks_to_lock = current_buffer_len - lock_threshold
                    
                    # 对需要锁定的部分进行推理
                    lock_audio = np.concatenate(audio_buffer[:chunks_to_lock])
                    lock_tensor = torch.from_numpy(lock_audio)
                    
                    lock_res = m.inference(
                        data_in=[lock_tensor],
                        language="auto",
                        use_itn=True,
                        ban_emo_unk=False,
                        fs=TARGET_FS,
                        **kwargs,
                    )
                    
                    if len(lock_res) > 0 and len(lock_res[0]) > 0:
                        lock_text = lock_res[0][0]["text"]
                        lock_text = re.sub(regex, "", lock_text, 0, re.MULTILINE)
                        lock_text = rich_transcription_postprocess(lock_text)
                        # 将这部分文本锁定
                        locked_text += lock_text
                    
                    # 移除已锁定的音频块
                    audio_buffer = audio_buffer[chunks_to_lock:]
                    window_start += chunks_to_lock
                
                # 对当前窗口内的音频进行推理
                if len(audio_buffer) > 0:
                    window_audio = np.concatenate(audio_buffer)
                    audio_tensor = torch.from_numpy(window_audio)
                    
                    res = m.inference(
                        data_in=[audio_tensor],
                        language="auto",
                        use_itn=True,
                        ban_emo_unk=False,
                        fs=TARGET_FS,
                        **kwargs,
                    )
                    
                    current_text = ""
                    if len(res) > 0 and len(res[0]) > 0:
                        text = res[0][0]["text"]
                        current_text = re.sub(regex, "", text, 0, re.MULTILINE)
                        current_text = rich_transcription_postprocess(current_text)
                    
                    # 组合已锁定文本和当前窗口文本
                    full_text = locked_text + current_text
                    
                    await websocket.send_json({
                        "text": full_text,
                        "locked": locked_text,      # 已确认不会再变的文本
                        "pending": current_text     # 当前窗口内可能还会变化的文本
                    })
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    audios = []
    for file in files:
        file_io = BytesIO(await file.read())
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)

        # transform to target sample
        if audio_fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            data_or_path_or_list = resampler(data_or_path_or_list)

        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=TARGET_FS,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=60109)
