import torchaudio
from matplotlib import pyplot as plt
import soundfile
import os

directory = r'C:\Data\voice conversion\Full_voice_inna_ogg'
dest = r'C:\Data\voice conversion\Full_voice_inna_m2o'

file_list = []

def recursive_find(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ogg"):
                file_list.append(file)

recursive_find(directory)


for file in file_list:
    waveform, sample_rate = torchaudio.load(os.path.join(directory, file))
    time = len(waveform[1,:]) / sample_rate

    dest_path = os.path.join(dest, file.replace('.ogg', '.wav'))

    torchaudio.save(dest_path, waveform, sample_rate)

