import sounddevice as sd
from scipy.io.wavfile import write
fs = 16000
seconds = 5
print("Recording started. Speak now...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write("test_audio.wav", fs, audio)
print("Recording saved as test_audio.wav")
