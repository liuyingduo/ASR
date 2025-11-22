# test_recognition.py
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import time

# 使用已经加载的模型配置
model = WhisperModel(
    "large-v2",
    device="cuda",
    compute_type="float16",
    download_root="./whisper_models"
)


def record_audio(duration=3):
    """录制音频"""
    p = pyaudio.PyAudio()

    try:
        print(f"🎤 录制 {duration} 秒音频...")

        # 音频参数
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        # 自动选择输入设备
        input_device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_device_index = i
                print(f"使用输入设备: {device_info['name']}")
                break

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK
        )

        frames = []
        total_chunks = int(RATE / CHUNK * duration)

        start_time = time.time()
        for i in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            progress = (i + 1) / total_chunks
            bar = "█" * int(30 * progress) + "░" * (30 - int(30 * progress))
            print(f"\r录制进度: [{bar}] {progress * 100:.1f}%", end="", flush=True)

        record_time = time.time() - start_time

        stream.stop_stream()
        stream.close()

        print(f"\n✅ 录制完成，耗时: {record_time:.2f}秒")

        # 转换为numpy数组
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0

        return audio_float, record_time

    except Exception as e:
        print(f"❌ 录制错误: {e}")
        return None, 0
    finally:
        p.terminate()


def transcribe_audio(audio_data):
    """转录音频"""
    print("🔊 Large-v2模型识别中...")

    start_time = time.time()

    try:
        segments, info = model.transcribe(
            audio_data,
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

        texts = []
        for segment in segments:
            texts.append(segment.text.strip())
            print(f"   - {segment.text}")

        text = " ".join(texts).strip()
        transcribe_time = time.time() - start_time

        return text, transcribe_time, info

    except Exception as e:
        print(f"❌ 识别错误: {e}")
        return "", 0, None


# 主程序
print("🎉 Large-v2模型测试")
print("=" * 50)

try:
    while True:
        # input("按回车开始录音（3秒）...")

        # 录制音频
        audio_data, record_time = record_audio(3)
        if audio_data is None:
            continue

        # 转录音频
        text, transcribe_time, info = transcribe_audio(audio_data)

        # 显示结果
        print("\n" + "=" * 40)
        if text and len(text) > 1:
            print(f"🗣️  识别结果: {text}")
        else:
            print("❌ 未识别到有效语音")

        total_time = record_time + transcribe_time
        print(f"⏱️  录制: {record_time:.2f}s")
        print(f"⏱️  识别: {transcribe_time:.2f}s")
        print(f"⏱️  总延迟: {total_time:.2f}s")
        print("=" * 40)

except KeyboardInterrupt:
    print("\n👋 测试结束")
