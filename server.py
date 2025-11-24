from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import numpy as np
import uvicorn
import os

app = FastAPI(title="ASR WebSocket API", description="Real-time speech to text using Faster Whisper")

# Initialize model
# Using the same configuration as in test_recognition.py
print("Loading Whisper model...")
model_size = "large-v2"
# Check if CUDA is available, otherwise use CPU (though user script used cuda)
# We'll stick to the user's config
model = WhisperModel(
    model_size,
    device="cuda",
    compute_type="float16",
    download_root="./whisper_models"
)
print("Model loaded.")

@app.get("/")
async def get():
    # Serve the frontend directly for convenience, or user can open index.html
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    # Buffer to store audio data
    # We expect float32 audio at 16000Hz
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            # Assuming client sends float32 raw PCM
            chunk = np.frombuffer(data, dtype=np.float32)
            
            # Append to buffer
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Perform transcription on the accumulated audio
            # In a production system, you would use a sliding window or VAD to process segments.
            # For this simple demo, we transcribe the entire buffer to allow context to improve results.
            # To prevent it from getting too slow, you might want to clear buffer after silence or limit length.
            
            # Only transcribe if we have some data
            if len(audio_buffer) > 0:
                segments, info = model.transcribe(
                    audio_buffer,
                    language="zh",
                    beam_size=5,
                    best_of=3,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
                
                # Collect text
                text = "".join([segment.text for segment in segments])
                
                # Send back the current full transcription
                await websocket.send_text(text)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
