# speech_recognition_d_drive.py
import os
import sys

# 添加D盘Python的site-packages到路径
d_python_lib = r"D:\python\Lib\site-packages"
if d_python_lib not in sys.path:
    sys.path.insert(0, d_python_lib)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pyaudio
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")


class DDriveSpeechRecognition:
    def __init__(self, model_size="base", language="zh"):
        print("=" * 60)
        print("D盘Python语音识别系统")
        print("=" * 60)

        print(f"Python路径: {sys.executable}")
        print(f"Python版本: {sys.version}")

        # 检查并安装依赖
        self._install_dependencies()

        # 加载模型
        self._load_model(model_size, language)

        # 音频设置
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        print("🎉 D盘Python语音识别系统就绪！")

    def _install_dependencies(self):
        """安装必要的依赖"""
        # 使用D盘Python的pip
        pip_path = r"D:\python\Scripts\pip.exe"

        packages = [
            "openai-whisper",
            "pyaudio",
            "numpy"
        ]

        for package in packages:
            try:
                if package == "pyaudio":
                    import pyaudio
                elif package == "numpy":
                    import numpy
                elif package == "openai-whisper":
                    import whisper
                print(f"✅ {package} 已安装")
            except ImportError:
                print(f"📥 安装 {package}...")
                os.system(f'"{pip_path}" install {package}')

    def _load_model(self, model_size, language):
        """加载语音识别模型"""
        print("📥 加载Whisper模型...")

        import whisper

        try:
            # 检查CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    print("✅ CUDA可用，使用GPU加速")
                    self.model = whisper.load_model(model_size)
                    print(f"✅ 模型已加载到: {self.model.device}")
                else:
                    print("ℹ️ CUDA不可用，使用CPU模式")
                    self.model = whisper.load_model(model_size)
            except:
                print("ℹ️ 使用默认模式加载模型")
                self.model = whisper.load_model(model_size)

            self.language = language

            # 预热
            print("🔥 预热模型...")
            dummy_audio = np.random.random(16000).astype(np.float32)
            _ = self.model.transcribe(dummy_audio, language=language)

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def record_audio(self, duration=5):
        """录制音频"""
        print(f"\n🎤 开始录制 {duration} 秒...")

        try:
            p = pyaudio.PyAudio()

            # 显示音频设备信息
            try:
                default_input = p.get_default_input_device_info()
                print(f"📱 使用音频设备: {default_input['name']}")
            except:
                print("📱 使用默认音频设备")

            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            frames = []
            start_time = time.time()

            for i in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                elapsed = time.time() - start_time
                progress = (elapsed / duration) * 100
                if i % 10 == 0:
                    print(f"\r⏺️ 录音进度: {progress:.1f}%", end="", flush=True)

            stream.stop_stream()
            stream.close()
            p.terminate()

            print("\n✅ 录音完成")

            # 转换为numpy数组
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            return audio_float

        except Exception as e:
            print(f"❌ 录音错误: {e}")
            return None

    def transcribe_audio(self, audio_array):
        """转录音频"""
        if audio_array is None:
            return "❌ 录音失败"

        print("🎯 转录中...")
        start_time = time.time()

        try:
            result = self.model.transcribe(
                audio_array,
                language=self.language,
                temperature=0.0
            )

            transcription_time = time.time() - start_time
            text = result["text"].strip()

            print(f"\n📊 转录耗时: {transcription_time:.2f}秒")
            print(f"🚀 处理速度: {5.0 / transcription_time:.1f}x实时")

            return text

        except Exception as e:
            print(f"❌ 转录错误: {e}")
            return f"转录失败: {e}"

    def start_demo(self):
        """开始演示"""
        print("\n" + "=" * 50)
        print("🎯 D盘Python语音识别演示")
        print("=" * 50)
        print("按 Ctrl+C 退出程序")
        print("=" * 50)

        try:
            while True:
                input("\n🎯 按回车开始录音 (5秒)...")

                # 录制和转录
                audio_data = self.record_audio(5)
                result = self.transcribe_audio(audio_data)

                print(f"\n🗣️  识别结果: {result}")
                print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 程序结束")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    try:
        # 启动语音识别
        stt = DDriveSpeechRecognition(model_size="base", language="zh")
        stt.start_demo()

    except Exception as e:
        print(f"❌ 启动失败: {e}")