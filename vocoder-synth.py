import pygame
import numpy as np
import rtmidi
import sounddevice as sd
import time
from scipy.signal import lfilter, butter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QSlider, QGroupBox, QTabWidget, QGridLayout, QComboBox, QPushButton)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QImage


# Parametry z ederwander
Fs = 48000  # Sampling frequency
num_bands = 33  # Number of bands
f_low = 50  # Lowest frequency (Hz)
f_high = 8000  # Highest frequency (Hz)
chunk = 2048  # Buffer size

# Generate center frequencies for bands (logarithmic scale)
frequencies = np.logspace(np.log10(f_low), np.log10(f_high), num_bands)

# Generate filter coefficients as in ederwander
def generate_filter_coefficients(fs, freqs, bandwidth_factor=0.1):
    b_coeffs = []
    a_coeffs = []
    for f in freqs:
        nyquist = fs / 2
        low = (f * (1 - bandwidth_factor)) / nyquist
        high = (f * (1 + bandwidth_factor)) / nyquist
        b, a = butter(2, [low, high], btype='band', analog=False)
        k = b[0]
        b_adjusted = [k, 0, -2*k, 0, k]
        b_coeffs.append(b_adjusted)
        a_coeffs.append(a)
    return b_coeffs, a_coeffs

b, a = generate_filter_coefficients(Fs, frequencies, bandwidth_factor=0.1)

class Vocoder:
    def __init__(self, chunk_size=2048, sample_rate=48000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.r = 0.99
        self.lowpassf = [1.0, -2.0 * self.r, self.r * self.r]
        self.d = 0.41004238851988095  # From ederwander
        self.dB = 10**(40/20)  # Default gain from ederwander
        self.phase0 = self.phase1 = self.phase2 = self.phase3 = 0
        self.phase4 = self.phase5 = self.phase6 = self.phase7 = 0
        self.input_data = np.zeros(2 * chunk_size)
        self.output_data = np.zeros(2 * chunk_size)

    def set_gain(self, gain_db):
        self.dB = 10**(gain_db/20)


    def carrier(self, s, f0):
        # Adapted from ederwander, pitch controlled by f0 (MIDI note frequency)
        w = 2.0 * np.pi * f0 / self.sample_rate
        t = np.linspace(0, s, s)
        self.phase1 = 0.2 * w * t + self.phase1
        self.phase2 = 0.4 * w * t + self.phase2
        self.phase3 = 0.5 * w * t + self.phase3
        self.phase4 = 2.0 * w * t + self.phase4
        self.phase5 = np.sin(self.phase1) - np.tan(self.phase3)
        self.phase6 = np.sin(self.phase1) + np.sin(self.phase4)
        self.phase7 = np.sin(self.phase2) - np.sin(self.phase4)
        x = np.sin(self.phase5)
        y = np.sin(self.phase6)
        z = np.sin(self.phase7)
        carriersignal = 0.25 * (x + y + z + self.d)
        # Update phases for continuity
        self.phase1 = np.mod(self.phase1[-1], 2.0 * np.pi)
        self.phase2 = np.mod(self.phase2[-1], 2.0 * np.pi)
        self.phase3 = np.mod(self.phase3[-1], 2.0 * np.pi)
        self.phase4 = np.mod(self.phase4[-1], 2.0 * np.pi)
        self.phase5 = np.mod(self.phase5[-1], 2.0 * np.pi)
        self.phase6 = np.mod(self.phase6[-1], 2.0 * np.pi)
        self.phase7 = np.mod(self.phase7[-1], 2.0 * np.pi)
        return carriersignal

    def process(self, sig, f0):
        N = len(sig)
        carriersignal = self.carrier(N, f0)
        vout = 0
        for i in range(num_bands):
            bandpasscarrier = lfilter(b[i], a[i], carriersignal)
            bandpassmodulator = lfilter(b[i], a[i], sig)
            rectifiedmodulator = abs(bandpassmodulator * bandpassmodulator) / N
            envelopemodulator = np.sqrt(lfilter([1.0], self.lowpassf, rectifiedmodulator))
            vout += bandpasscarrier * envelopemodulator
        vout = np.clip(vout * self.dB, -1, 1)
        self.input_data = np.roll(self.input_data, -N)
        self.input_data[-N:] = sig
        self.output_data = np.roll(self.output_data, -N)
        self.output_data[-N:] = vout
        return vout
class MidiInputHandler:
    def __init__(self, callback=None):
        self.midi_in = rtmidi.MidiIn()
        self.callback = callback
        self.available_ports = self.midi_in.get_ports()

    def get_ports(self):
        return self.available_ports

    def open_port(self, port_index):
        if port_index < len(self.available_ports):
            self.midi_in.open_port(port_index)
            self.midi_in.set_callback(self._midi_callback)
            return True
        return False

    def close_port(self):
        self.midi_in.close_port()

    def _midi_callback(self, message, _):
        if self.callback and len(message[0]) >= 3:
            status = message[0][0] & 0xF0
            channel = message[0][0] & 0x0F
            note = message[0][1]
            velocity = message[0][2]
            if status == 0x90 and velocity > 0:
                self.callback("note_on", note, velocity)
            elif status == 0x80 or (status == 0x90 and velocity == 0):
                self.callback("note_off", note, 0)

class AudioHandler:
    def __init__(self, callback=None, sample_rate=48000, chunk_size=2048):
        self.callback = callback
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.running = False

    def get_input_devices(self):
        return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]

    def get_output_devices(self):
        return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]

    def start_stream(self, input_device_index=None, output_device_index=None):
        if self.stream:
            self.stop_stream()
        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=(input_device_index, output_device_index),
                channels=(1, 2),
                dtype='float32',
                callback=self._audio_callback
            )
            self.running = True
            self.stream.start()
            print(f"Audio stream started with {self.sample_rate} Hz")
            return True
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False

    def stop_stream(self):
        if self.stream:
            self.running = False
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        if self.running and self.callback:
            mono_out = self.callback(indata[:, 0])
            outdata[:, 0] = mono_out
            outdata[:, 1] = mono_out
        else:
            outdata.fill(0)

class PianoRollWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.active_notes = {}
        self.history = []
        self.start_time = time.time()

    def add_note(self, note, velocity):
        if velocity > 0:
            self.active_notes[note] = velocity
            self.history.append((note, velocity, time.time() - self.start_time))
        else:
            if note in self.active_notes:
                del self.active_notes[note]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        width = self.width()
        height = self.height()
        key_height = height / 88
        white_key_width = width * 0.2

        for i in range(88):
            note_number = i + 21
            note_name = note_number % 12
            if note_name not in [1, 3, 6, 8, 10]:
                y = height - (i + 1) * key_height
                color = QColor(100, 200, 255) if note_number in self.active_notes else QColor(230, 230, 230)
                painter.fillRect(0, int(y), int(white_key_width), int(key_height), color)
                painter.setPen(QPen(QColor(180, 180, 180)))
                painter.drawLine(0, int(y), int(white_key_width), int(y))

        for i in range(88):
            note_number = i + 21
            note_name = note_number % 12
            if note_name in [1, 3, 6, 8, 10]:
                y = height - (i + 1) * key_height
                color = QColor(100, 200, 255) if note_number in self.active_notes else QColor(40, 40, 40)
                painter.fillRect(0, int(y), int(white_key_width * 0.6), int(key_height), color)

        current_time = time.time() - self.start_time
        history_duration = 5.0
        self.history = [h for h in self.history if current_time - h[2] < history_duration]
        for note, velocity, note_time in self.history:
            if current_time - note_time < history_duration:
                x_pos = white_key_width + (current_time - note_time) / history_duration * (width - white_key_width)
                y_pos = height - ((note - 21) + 1) * key_height
                note_width = 10 * (velocity / 127)
                color = QColor(
                    min(int(100 + velocity * 1.2), 255),
                    min(int(100 + velocity * 0.8), 255),
                    min(int(200 + velocity * 0.4), 255),
                    max(0, min(255, int(255 - (current_time - note_time) / history_duration * 200)))
                )
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(int(x_pos), int(y_pos), int(note_width), int(key_height))

        painter.setPen(QPen(QColor(255, 255, 255)))
        for octave in range(9):
            note = 24 + octave * 12
            if 21 <= note <= 108:
                y = height - ((note - 21) + 1) * key_height
                painter.drawText(5, int(y + key_height * 0.7), f"C{octave}")

class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.surface_size = (400, 150)
        self.pygame_surface = None
        self.initialize_pygame()
        self.data = np.zeros((100, 128))
        self.cmap = self.create_colormap('viridis', 256)

    def initialize_pygame(self):
        if not pygame.get_init():
            pygame.init()
        self.pygame_surface = pygame.Surface(self.surface_size)

    def create_colormap(self, name, n_colors):
        if name == 'viridis':
            colors = []
            for i in range(n_colors):
                t = i / (n_colors - 1)
                r = int(68 + (85 - 68) * t)
                g = int(1 + (180 - 1) * t)
                b = int(84 + (164 - 84) * t)
                colors.append((r, g, b))
            return colors
        return [(i, i, i) for i in range(n_colors)]

    def update_data(self, data):
        n_fft = min(256, len(data))
        spectrum = np.abs(np.fft.rfft(data, n=n_fft))
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1, :min(128, len(spectrum))] = spectrum[:128] / (np.max(spectrum) + 1e-8)
        self.update()

    def resizeEvent(self, event):
        self.surface_size = (self.width(), self.height())
        self.pygame_surface = pygame.Surface(self.surface_size)
        super().resizeEvent(event)

    def paintEvent(self, event):
        if self.pygame_surface is None or self.pygame_surface.get_size() != (self.width(), self.height()):
            self.pygame_surface = pygame.Surface((self.width(), self.height()))
        self.pygame_surface.fill((20, 20, 20))
        max_val = np.max(self.data) if np.max(self.data) > 0 else 1
        width = self.width()
        height = self.height()
        bin_width = width / self.data.shape[0]
        bin_height = height / self.data.shape[1]

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                val = self.data[i, j] / max_val
                color_idx = min(int(val * (len(self.cmap) - 1)), len(self.cmap) - 1)
                color = self.cmap[color_idx]
                pygame.draw.rect(
                    self.pygame_surface,
                    color,
                    (i * bin_width, height - (j + 1) * bin_height, bin_width, bin_height)
                )

        font = pygame.font.SysFont('Arial', 10)
        for i in range(0, 48000 // 2000):
            freq = i * 2000
            y_pos = height - (freq / (48000/2)) * height
            text = font.render(f"{freq} Hz", True, (200, 200, 200))
            self.pygame_surface.blit(text, (5, y_pos))

        buffer = pygame.image.tostring(self.pygame_surface, "RGB")
        qimage = QImage(buffer, width, height, width * 3, QImage.Format.Format_RGB888)
        painter = QPainter(self)
        painter.drawImage(0, 0, qimage)

class VocoderSynthApp(QMainWindow):
    update_spectrogram_signal = pyqtSignal(np.ndarray)
    update_output_spectrogram_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vocoder Synth (Ederwander Style)")
        self.setMinimumSize(1000, 700)

        self.sampling_rate = Fs
        self.chunk_size = chunk
        self.vocoder = Vocoder(self.chunk_size, self.sampling_rate)
        self.active_notes = {}
        self.last_note = None
        self.mix_ratio = 0.5

        self.midi_handler = MidiInputHandler(self.handle_midi_message)
        self.audio_handler = AudioHandler(self.process_audio, self.sampling_rate, self.chunk_size)

        self.init_ui()

        self.update_spectrogram_signal.connect(self.input_spectrogram.update_data)
        self.update_output_spectrogram_signal.connect(self.output_spectrogram.update_data)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(50)

    def init_ui(self):
        # Same as original, only adjusting gain range to match ederwander
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        synth_tab = QWidget()
        vocoder_tab = QWidget()
        settings_tab = QWidget()
        tabs.addTab(synth_tab, "Synthesizer")
        tabs.addTab(vocoder_tab, "Vocoder")
        tabs.addTab(settings_tab, "Settings")

        synth_layout = QVBoxLayout(synth_tab)
        piano_group = QGroupBox("Piano Roll")
        piano_layout = QVBoxLayout(piano_group)
        self.piano_roll = PianoRollWidget()
        piano_layout.addWidget(self.piano_roll)
        synth_layout.addWidget(piano_group)

        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        spectrograms_layout = QHBoxLayout()
        input_spec_group = QGroupBox("Input")
        input_spec_layout = QVBoxLayout(input_spec_group)
        self.input_spectrogram = SpectrogramWidget()
        input_spec_layout.addWidget(self.input_spectrogram)
        spectrograms_layout.addWidget(input_spec_group)
        output_spec_group = QGroupBox("Output")
        output_spec_layout = QVBoxLayout(output_spec_group)
        self.output_spectrogram = SpectrogramWidget()
        output_spec_layout.addWidget(self.output_spectrogram)
        spectrograms_layout.addWidget(output_spec_group)
        viz_layout.addLayout(spectrograms_layout)
        synth_layout.addWidget(viz_group)

        vocoder_layout = QVBoxLayout(vocoder_tab)
        vocoder_params_group = QGroupBox("Vocoder Parameters")
        vocoder_params_layout = QGridLayout(vocoder_params_group)

        vocoder_params_layout.addWidget(QLabel("Gain:"), 0, 0)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 60)  # Adjusted to reasonable range
        self.gain_slider.setValue(40)  # Default from ederwander
        self.gain_slider.valueChanged.connect(self.update_vocoder_gain)
        vocoder_params_layout.addWidget(self.gain_slider, 0, 1)
        self.gain_label = QLabel("40 dB")
        vocoder_params_layout.addWidget(self.gain_label, 0, 2)

        vocoder_params_layout.addWidget(QLabel("Mix Ratio:"), 1, 0)
        self.mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(50)
        self.mix_slider.valueChanged.connect(self.update_mix_ratio)
        vocoder_params_layout.addWidget(self.mix_slider, 1, 1)
        self.mix_label = QLabel("50%")
        vocoder_params_layout.addWidget(self.mix_label, 1, 2)

        vocoder_layout.addWidget(vocoder_params_group)

        settings_layout = QVBoxLayout(settings_tab)
        midi_group = QGroupBox("MIDI Input")
        midi_layout = QVBoxLayout(midi_group)
        midi_ports_layout = QHBoxLayout()
        midi_ports_layout.addWidget(QLabel("MIDI Device:"))
        self.midi_combo = QComboBox()
        midi_ports = self.midi_handler.get_ports()
        if midi_ports:
            self.midi_combo.addItems(midi_ports)
        else:
            self.midi_combo.addItem("No MIDI devices found")
        midi_ports_layout.addWidget(self.midi_combo)
        self.midi_connect_button = QPushButton("Connect")
        self.midi_connect_button.clicked.connect(self.connect_midi)
        midi_ports_layout.addWidget(self.midi_connect_button)
        midi_layout.addLayout(midi_ports_layout)
        settings_layout.addWidget(midi_group)

        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout(audio_group)
        audio_input_layout = QHBoxLayout()
        audio_input_layout.addWidget(QLabel("Audio Input:"))
        self.audio_input_combo = QComboBox()
        input_devices = self.audio_handler.get_input_devices()
        if input_devices:
            for idx, name in input_devices:
                self.audio_input_combo.addItem(name, idx)
        else:
            self.audio_input_combo.addItem("No input devices found", -1)
        audio_input_layout.addWidget(self.audio_input_combo)
        audio_output_layout = QHBoxLayout()
        audio_output_layout.addWidget(QLabel("Audio Output:"))
        self.audio_output_combo = QComboBox()
        output_devices = self.audio_handler.get_output_devices()
        if output_devices:
            for idx, name in output_devices:
                self.audio_output_combo.addItem(name, idx)
        else:
            self.audio_output_combo.addItem("No output devices found", -1)
        audio_output_layout.addWidget(self.audio_output_combo)
        audio_layout.addLayout(audio_input_layout)
        audio_layout.addLayout(audio_output_layout)
        audio_control_layout = QHBoxLayout()
        self.audio_connect_button = QPushButton("Start Audio")
        self.audio_connect_button.clicked.connect(self.start_audio)
        audio_control_layout.addWidget(self.audio_connect_button)
        audio_layout.addLayout(audio_control_layout)
        settings_layout.addWidget(audio_group)

        self.setCentralWidget(central_widget)
        self.start_audio()

    def update_vocoder_gain(self, value):
        self.vocoder.set_gain(value)
        self.gain_label.setText(f"{value} dB")

    def update_mix_ratio(self, value):
        self.mix_ratio = value / 100.0
        self.mix_label.setText(f"{value}%")


    def connect_midi(self):
        port_index = self.midi_combo.currentIndex()
        if self.midi_handler.open_port(port_index):
            self.midi_connect_button.setText("Disconnect")
            self.midi_connect_button.clicked.disconnect()
            self.midi_connect_button.clicked.connect(self.disconnect_midi)
        else:
            print("Failed to open MIDI port")

    def disconnect_midi(self):
        self.midi_handler.close_port()
        self.midi_connect_button.setText("Connect")
        self.midi_connect_button.clicked.disconnect()
        self.midi_connect_button.clicked.connect(self.connect_midi)

    def start_audio(self):
        try:
            input_index = self.audio_input_combo.currentData()
            output_index = self.audio_output_combo.currentData()
            if input_index == -1: input_index = None
            if output_index == -1: output_index = None
            success = self.audio_handler.start_stream(input_index, output_index)
            if success:
                self.audio_connect_button.setText("Stop Audio")
                self.audio_connect_button.clicked.disconnect()
                self.audio_connect_button.clicked.connect(self.stop_audio)
                self.statusBar().showMessage("Audio stream started successfully")
            else:
                self.statusBar().showMessage("Failed to start audio stream")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")

    def stop_audio(self):
        self.audio_handler.stop_stream()
        self.audio_connect_button.setText("Start Audio")
        self.audio_connect_button.clicked.disconnect()
        self.audio_connect_button.clicked.connect(self.start_audio)

    def handle_midi_message(self, message_type, note, velocity):

        if message_type == "note_on":
            self.active_notes[note] = velocity
            self.last_note = note
            self.piano_roll.add_note(note, velocity)
        elif message_type == "note_off" and note in self.active_notes:
            del self.active_notes[note]
            self.piano_roll.add_note(note, 0)
            if not self.active_notes and self.last_note == note:
                self.last_note = None

    def process_audio(self, input_data):
        # Dynamic amplification of microphone input
        input_max = np.max(np.abs(input_data))
        if input_max > 0:
            input_data_amplified = input_data * (0.5 / input_max)
        else:
            input_data_amplified = np.zeros_like(input_data)

        # Calculate carrier frequency from last MIDI note
        if self.last_note is None:
            f0 = 320  # Default frequency from ederwander
        else:
            f0 = 440.0 * (2.0 ** ((self.last_note - 69) / 12.0))  # MIDI note to frequency

        vocoder_output = self.vocoder.process(input_data_amplified, f0)

        # Mixing with normalization
        mixed_output = (1 - self.mix_ratio) * input_data_amplified + self.mix_ratio * vocoder_output
        output_max = np.max(np.abs(mixed_output))
        if output_max > 0:
            mixed_output = mixed_output * (0.9 / output_max)
        mixed_output = np.clip(mixed_output, -1, 1)

        return mixed_output.astype(np.float32)

    def update_visualizations(self):
        self.update_spectrogram_signal.emit(self.vocoder.input_data)
        self.update_output_spectrogram_signal.emit(self.vocoder.output_data)

    def closeEvent(self, event):
        self.audio_handler.stop_stream()
        self.midi_handler.close_port()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = VocoderSynthApp()
    window.show()
    app.exec()
