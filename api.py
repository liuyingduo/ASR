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

    # Double buffer strategy
    # All audio is concatenated, with a boundary index tracking confirmed vs pending
    full_audio_buffer = []  # List of audio chunks
    confirmed_samples = 0   # Number of confirmed audio samples (boundary)
    confirmed_text = ""     # Confirmed text result

    # Parameters for inference frequency
    chunk_count = 0
    inference_interval = 4  # Run inference every 4 chunks (approx 1 sec)

    # Window settings
    pending_window = 5  # Keep last 5 seconds as pending (can be modified)
    confirm_threshold = int(pending_window * TARGET_FS)  # Convert to samples

    try:
        while True:
            data = await websocket.receive_bytes()
            # Convert bytes to numpy int16
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            # Convert to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0

            # Add to buffer
            full_audio_buffer.append(audio_float)

            chunk_count += 1

            if chunk_count % inference_interval == 0:
                # Concatenate all audio
                full_audio = np.concatenate(full_audio_buffer)
                total_samples = len(full_audio)

                # If total audio exceeds threshold, update confirmed boundary
                if total_samples > confirm_threshold:
                    confirmed_samples = total_samples - confirm_threshold

                    # Extract confirmed portion and update text
                    confirmed_audio = full_audio[:confirmed_samples]
                    confirmed_tensor = torch.from_numpy(confirmed_audio)

                    res = m.inference(
                        data_in=[confirmed_tensor],
                        language="auto",
                        use_itn=True,
                        ban_emo_unk=False,
                        fs=TARGET_FS,
                        **kwargs,
                    )

                    if len(res) > 0 and len(res[0]) > 0:
                        confirmed_text = re.sub(regex, "", res[0][0]["text"], 0, re.MULTILINE)
                        confirmed_text = rich_transcription_postprocess(confirmed_text)

                # Extract pending portion (from confirmed_samples to end)
                if total_samples > confirmed_samples:
                    pending_audio = full_audio[confirmed_samples:]
                    pending_tensor = torch.from_numpy(pending_audio)

                    # Run inference on pending buffer only
                    res = m.inference(
                        data_in=[pending_tensor],
                        language="auto",
                        use_itn=True,
                        ban_emo_unk=False,
                        fs=TARGET_FS,
                        **kwargs,
                    )

                    if len(res) > 0 and len(res[0]) > 0:
                        pending_text = res[0][0]["text"]
                        # Clean text
                        pending_text = re.sub(regex, "", pending_text, 0, re.MULTILINE)
                        pending_text = rich_transcription_postprocess(pending_text)

                        await websocket.send_json({
                            "confirmed_text": confirmed_text,
                            "pending_text": pending_text,
                            "is_final": False
                        })

    except WebSocketDisconnect:
        # On disconnect, finalize all pending audio
        if full_audio_buffer:
            full_audio = np.concatenate(full_audio_buffer)
            full_tensor = torch.from_numpy(full_audio)
            res = m.inference(
                data_in=[full_tensor],
                language="auto",
                use_itn=True,
                ban_emo_unk=False,
                fs=TARGET_FS,
                **kwargs,
            )

            if len(res) > 0 and len(res[0]) > 0:
                final_text = re.sub(regex, "", res[0][0]["text"], 0, re.MULTILINE)
                final_text = rich_transcription_postprocess(final_text)
                await websocket.send_json({
                    "confirmed_text": final_text,
                    "pending_text": "",
                    "is_final": True
                })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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

    uvicorn.run(app, host="0.0.0.0", port=50000)
