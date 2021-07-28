import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import unicodedata
import re
import time
from torch.autograd import Variable
from math import sqrt
from Model import voice_model
import torch_optimizer as optim
import matplotlib.pyplot as plt

#parameters
batch_size = 1

def recursive_find(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".ogg")):
                file_list.append(os.path.join(root, file))
    return file_list

class Collect():
  def __init__(self):
    print("make collect")
  def __call__(self, batch):
    wav_list = []
    label_list = []
    # batch: N_batch * [wav, sample_rate, text, text_normalized]
    for _, data in enumerate(batch):
      wav, label = data[0], data[1]
      wav_list.append(wav[0].squeeze())
      label_list.append(label[0].squeeze())

    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(data[0][0]) for data in batch]), dim=0,
                                                      descending=True)
    wav_lengths, ids_sorted_wav = torch.sort(torch.LongTensor([wav.shape for wav in wav_list]), dim=0, descending=True)
    max_target_len = np.int(wav_lengths[0])

    wav_padded = torch.FloatTensor(len(batch), max_target_len)
    label_padded = torch.FloatTensor(len(batch), max_target_len)

    # wav_padded = torch.zeros(len(batch), max_target_len)
    # label_padded = torch.zeros(len(batch), max_target_len)
    wav_padded.zero_()
    label_padded.zero_()

    for i in range(len(ids_sorted_decreasing)):
        wav_padded[i, :wav_list[i].size(0)] = wav_list[i]
        label_padded[i, :label_list[i].size(0)] = label_list[i]

        # plt.figure()
        # plt.plot(wav_padded[i].t().numpy())
        # plt.show()
    return wav_padded, label_padded

class voiceDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, data_path, label_path):
        self.x = recursive_find(data_path)
        self.y = recursive_find(label_path)
        self.len = len(self.x)

    def __getitem__(self, index):
        x = torchaudio.load(self.x[index])
        y = torchaudio.load(self.y[index])
        return x, y

    def __len__(self):
        return self.len

total_dataset = voiceDataset(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\conversed_voice',
                             r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\LJSpeech-1.1\wavs')

collect_fn = Collect()
test_loader = torch.utils.data.DataLoader(total_dataset,batch_size = batch_size, collate_fn=(collect_fn), drop_last=True)

# model = voice_model.noise_remover(channel = 512, kernel_size = 5, stage = 3)
model = torch.load(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Model\noise_remover\noise_remover_13.655743991586519_10.689979350659286.pth')
device = torch.device('cuda')
model.to(device)
model.eval()

dest = r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\predicted_voice'
fig_dest = r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\predicted_voice_jpg'

for i, data in enumerate(test_loader):##validation batch
    with torch.no_grad():
        inputs, labels = data
        inputs = inputs.to(device).unsqueeze(1)

       # 순전파 + 역전파 + 최적화를 한 후
        outputs = model(inputs)

        for j in range(len(outputs)):
            plt.figure()
            plt.plot(outputs.cpu().squeeze(1).t().numpy())
            plt.savefig(os.path.join(fig_dest, test_loader.dataset.x[i].split('\\')[-1]+'.jpg'))
            plt.clf()
            torchaudio.save(os.path.join(dest, test_loader.dataset.x[i].split('\\')[-1].replace('.ogg', '.wav')), outputs.cpu().squeeze(1), 22050, format='wav')

print('Finished Predict')