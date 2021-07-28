import torchaudio
import os
import matplotlib.pyplot as plt
# files
src = r"C:\Users\Joker\Desktop\gitwork\tako_voice\Data\conversed_voice"
dst = r"C:\Users\Joker\Desktop\gitwork\tako_voice\Data\conversed_voice_graph"

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

    plt.figure()
    plt.plot(re_wave.cpu().detach().squeeze(1).t().numpy())
    plt.savefig(os.path.join(dst, i.split('\\')[-1].split('.')[0] + '.jpg'))
    plt.clf()

    # torchaudio.save(os.path.join(dst, i.split('\\')[-1].replace('ogg', 'wav')), re_wave, new_sample_rate)