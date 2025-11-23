# faster_realtime_asr.py
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
import os


class FasterRealtimeASR:
    def __init__(self, model_size="large-v2", sample_rate=16000, chunk_duration=3.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        # 性能参数
        self.silence_threshold = 0.01
        self.min_audio_length = 1.0

        print(f"🚀 初始化faster-whisper实时转录系统...")
        print(f"   模型: {model_size}")
        print(f"   采样率: {sample_rate}Hz")
        print(f"   块时长: {chunk_duration}秒")

        # 加载faster-whisper模型
        self.model = self.load_faster_whisper_model(model_size)

        # 音频处理
        self.audio_queue = queue.Queue(maxsize=20)
        self.is_recording = False
        self.is_processing = False
        self.transcription_history = []

        # PyAudio实例
        self.p = pyaudio.PyAudio()

        # GUI相关
        self.root = None
        self.text_widget = None
        self.status_var = None

        print("✅ faster-whisper系统初始化完成！")

    def load_faster_whisper_model(self, model_size):
        """加载faster-whisper模型"""
        print(f"📥 加载faster-whisper模型: {model_size}")

        try:
            model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16",
                download_root="./whisper_models"
            )
            print("   ✅ 模型加载成功 (CUDA)")
            return model
        except Exception as e:
            print(f"❌ CUDA加载失败: {e}")
            print("🔄 尝试使用CPU...")
            try:
                model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root="./whisper_models"
                )
                print("   ✅ 模型加载成功 (CPU)")
                return model
            except Exception as e2:
                print(f"❌ 模型加载完全失败: {e2}")
                raise e2

    def find_audio_input_device(self):
        """查找音频输入设备"""
        print("🔍 扫描音频输入设备...")

        input_devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
                print(f"   📢 {i}: {device_info['name']} (通道: {device_info['maxInputChannels']})")

        if not input_devices:
            raise Exception("未找到可用的音频输入设备")

        # 选择第一个可用的输入设备
        selected_device = input_devices[0]
        print(f"✅ 选择设备: {selected_device['name']}")
        return selected_device['index']

    def audio_capture_thread(self):
        """音频捕获线程"""
        device_index = self.find_audio_input_device()

        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )

        print("🎤 开始音频捕获...")

        try:
            while self.is_recording:
                # 读取音频数据
                data = stream.read(1024, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # 放入队列（非阻塞）
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_chunk, block=False)

        except Exception as e:
            print(f"❌ 音频捕获错误: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    def calculate_audio_energy(self, audio_chunk):
        """计算音频能量"""
        return np.sqrt(np.mean(audio_chunk ** 2))

    def is_valid_audio(self, audio_chunk):
        """检测是否为有效音频"""
        energy = self.calculate_audio_energy(audio_chunk)
        return energy > self.silence_threshold

    def transcribe_audio(self, audio_data):
        """使用faster-whisper转录音频"""
        try:
            segments, info = self.model.transcribe(
                audio_data,
                language="zh",
                beam_size=3,
                best_of=2,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                ),
                without_timestamps=True
            )

            texts = []
            for segment in segments:
                if segment.text.strip():
                    texts.append(segment.text.strip())

            return " ".join(texts).strip() if texts else None

        except Exception as e:
            print(f"❌ 转录错误: {e}")
            return None

    def process_audio_stream(self):
        """处理音频流"""
        audio_buffer = []
        silence_counter = 0
        last_transcription = ""

        while self.is_processing:
            try:
                # 获取音频数据
                audio_chunk = self.audio_queue.get(timeout=1.0)
                audio_buffer.extend(audio_chunk)

                # 当有足够数据时处理
                while len(audio_buffer) >= self.chunk_samples:
                    chunk_to_process = audio_buffer[:self.chunk_samples]
                    audio_buffer = audio_buffer[self.chunk_samples:]

                    # 检查是否为有效音频
                    if self.is_valid_audio(np.array(chunk_to_process)):
                        # 转录
                        transcription = self.transcribe_audio(np.array(chunk_to_process))

                        if transcription and transcription != last_transcription:
                            current_time = time.strftime("%H:%M:%S")
                            result_text = f"[{current_time}] 🎯 {transcription}"

                            # 更新GUI
                            if self.text_widget:
                                self.root.after(0, self.update_text_widget, result_text)

                            print(f"\r{result_text}" + " " * 50)

                            self.transcription_history.append({
                                'time': current_time,
                                'text': transcription
                            })
                            last_transcription = transcription

                            silence_counter = 0
                            if self.status_var:
                                self.root.after(0, lambda: self.status_var.set("状态: 检测到语音"))
                        else:
                            silence_counter += 1
                    else:
                        silence_counter += 1

                    # 静音状态更新
                    if silence_counter > 0 and silence_counter % 10 == 0 and self.status_var:
                        self.root.after(0, lambda: self.status_var.set(f"状态: 监听中... (静音{silence_counter}次)"))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                if self.status_var:
                    self.root.after(0, lambda: self.status_var.set(f"状态: 错误 - {e}"))

    def update_text_widget(self, text):
        """更新文本显示"""
        if self.text_widget:
            self.text_widget.insert(tk.END, text + "\n")
            self.text_widget.see(tk.END)

    def start_transcription(self):
        """开始转录"""
        if self.is_recording:
            self.update_status("⚠️ 转录已在运行中")
            return

        print("\n🎤 开始实时语音转录...")

        self.is_recording = True
        self.is_processing = True

        self.update_status("状态: 启动中...")

        # 启动音频捕获线程
        audio_thread = threading.Thread(target=self.audio_capture_thread)
        audio_thread.daemon = True
        audio_thread.start()

        # 启动处理线程
        process_thread = threading.Thread(target=self.process_audio_stream)
        process_thread.daemon = True
        process_thread.start()

        self.update_status("状态: 录音中...请开始说话")

    def stop_transcription(self):
        """停止转录"""
        if not self.is_recording:
            return

        print("\n🛑 停止转录系统...")
        self.is_recording = False
        self.is_processing = False

        self.update_status("状态: 已停止")

        # 显示统计
        if self.transcription_history:
            print(f"\n📊 本次会话统计:")
            print(f"   转录片段: {len(self.transcription_history)}")
            if self.transcription_history:
                print(f"   最后一条: {self.transcription_history[-1]['text']}")

    def update_status(self, message):
        """更新状态"""
        if self.status_var and self.root:
            self.root.after(0, lambda: self.status_var.set(message))

    def save_transcription(self, filename=None):
        """保存转录结果"""
        if not self.transcription_history:
            self.update_status("⚠️ 没有转录结果可保存")
            return

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("faster-whisper实时转录记录\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总记录数: {len(self.transcription_history)}\n")
                f.write("=" * 50 + "\n\n")

                for i, item in enumerate(self.transcription_history, 1):
                    f.write(f"{i:02d}. [{item['time']}] {item['text']}\n")

            print(f"✅ 转录结果已保存到: {filename}")
            self.update_status(f"✅ 已保存: {filename}")
            return filename

        except Exception as e:
            error_msg = f"❌ 保存失败: {e}"
            print(error_msg)
            self.update_status(error_msg)
            return None

    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("faster-whisper实时转录系统")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame,
                                text="🎤 faster-whisper实时语音转录",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 15))

        # 状态显示
        self.status_var = tk.StringVar(value="状态: 就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                 font=('Arial', 11), foreground='blue')
        status_label.grid(row=1, column=0, columnspan=4, pady=(0, 10))

        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=(0, 15))

        start_btn = ttk.Button(button_frame, text="🎤 开始转录",
                               command=self.start_transcription, width=15)
        start_btn.grid(row=0, column=0, padx=5)

        stop_btn = ttk.Button(button_frame, text="⏹️ 停止转录",
                              command=self.stop_transcription, width=15)
        stop_btn.grid(row=0, column=1, padx=5)

        save_btn = ttk.Button(button_frame, text="💾 保存结果",
                              command=self.save_transcription, width=15)
        save_btn.grid(row=0, column=2, padx=5)

        clear_btn = ttk.Button(button_frame, text="🗑️ 清空记录",
                               command=self.clear_text, width=15)
        clear_btn.grid(row=0, column=3, padx=5)

        # 转录结果显示
        text_frame = ttk.LabelFrame(main_frame, text="实时转录结果", padding="8")
        text_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))

        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            width=100,
            height=25,
            font=('Arial', 10),
            bg='#fafafa'
        )
        self.text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 系统信息
        info_frame = ttk.LabelFrame(main_frame, text="系统信息", padding="8")
        info_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E))

        info_text = f"""
模型: faster-whisper large-v2 | 设备: CUDA | 采样率: {self.sample_rate}Hz
特性: VAD语音检测 | 实时转录 | 中文优化 | 低延迟
提示: 说话清晰，保持适当距离，减少背景噪音
        """
        info_label = ttk.Label(info_frame, text=info_text.strip(), justify=tk.LEFT)
        info_label.grid(row=0, column=0, sticky=tk.W)

        # 配置网格
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        return self.root

    def clear_text(self):
        """清空文本"""
        if self.text_widget:
            self.text_widget.delete(1.0, tk.END)
        self.transcription_history.clear()
        self.update_status("📝 转录记录已清空")
        print("📝 转录记录已清空")

    def on_closing(self):
        """关闭处理"""
        self.stop_transcription()
        self.p.terminate()
        self.root.destroy()
        print("👋 程序已安全退出")

    def run_gui(self):
        """运行GUI"""
        if self.root is None:
            self.create_gui()

        print("🎮 启动图形界面...")
        print("💡 提示: 确保麦克风权限已开启")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 用户中断程序")
        finally:
            self.stop_transcription()
            self.p.terminate()


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 faster-whisper实时语音转录系统")
    print("   基于faster-whisper技术")
    print("   CUDA加速版本")
    print("=" * 60)

    try:
        # 初始化系统
        asr_system = FasterRealtimeASR(
            model_size="large-v2",  # 使用large-v2模型
            sample_rate=16000,
            chunk_duration=3.0
        )

        # 启动图形界面
        asr_system.run_gui()

    except Exception as e:
        print(f"\n❌ 系统启动失败: {e}")
        print("\n💡 请确保已安装: pip install faster-whisper pyaudio")


if __name__ == "__main__":
    main()