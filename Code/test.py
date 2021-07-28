import torchaudio
import os

#train_dataset = torchaudio.datasets.LJSPEECH(r"C:\Users\Joker\Desktop\gitwork\tako_voice\Data")

# files
src = r"C:\Users\Joker\Desktop\gitwork\tako_voice\Data\LJSpeech-1.1\wavs"
dst = r"C:\Users\Joker\Desktop\gitwork\tako_voice\Data\LJSpeech-1.1\wavs"

def recursive_find(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".ogg")):
                file_list.append(os.path.join(root, file))
    return file_list

files = recursive_find(src)

# convert wav to mp3
for i in files:
    channel = 0
    waveform, sample_rate = torchaudio.load(i)
    new_sample_rate = 22050
    re_wave = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))
    torchaudio.save(os.path.join(dst, i.split('\\')[-1].replace('ogg', 'wav')), re_wave, new_sample_rate)