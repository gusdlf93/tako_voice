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
batch_size = 2

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
    for data in batch:
      wav, label = data[0], data[1]
      wav_list.append(wav[0].squeeze())
      label_list.append(label[0].squeeze())

    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(data[0][0]) for data in batch]), dim=0, descending=True)
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
        self.data_path = data_path
        self.label_path = label_path

        self.data_list = recursive_find(data_path)
        self.len = len(self.data_list)

    def __getitem__(self, index):
        x = torchaudio.load(os.path.join(self.data_path, self.data_list[index]))
        y = torchaudio.load(os.path.join(self.label_path, self.data_list[index]))
        return x, y

    def __len__(self):
        return self.len

total_dataset = voiceDataset(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\conversed_voice',
                             r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\LJSpeech-1.1\wavs')

test_dataset = voiceDataset(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\parsed\ogg_22050',
                             r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\parsed\ogg_22050')

train_index = np.arange(int(len(total_dataset)* 0.8))
val_index = np.arange(int(len(total_dataset)*0.8), len(total_dataset))

collect_fn = Collect()
train_loader = torch.utils.data.DataLoader(total_dataset,batch_size = batch_size,
                                           sampler=SubsetRandomSampler(train_index), collate_fn=(collect_fn),
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(total_dataset, batch_size = batch_size,
                                         sampler=SubsetRandomSampler(val_index), collate_fn=(collect_fn),
                                         drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, collate_fn=(collect_fn),
                                         drop_last=True)

model = voice_model.noise_remover(en_channel = [1, 64, 128, 256],de_channel = [256, 128, 64, 1],
                                  kernel_size = 5, stage = 3)
device = torch.device('cuda')
model.to(device)

SGD = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

class Custom_MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.mean((input-target)**2 + 1e-12)

# Radam = optim.RAdam(
#     model.parameters(),
#     lr= 1e-6,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=0,
# )
#optimizer = optim.Lookahead(Radam, k=5, alpha=0.5)
loss_fc  = Custom_MSELoss()

epochs = 500
dest = r''
max_acc = 0
min_loss = 100
for epoch in range(epochs):  # epochs
    total = 0
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):##training batch
        #try:
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device).unsqueeze(1)

        SGD.zero_grad()  # 변화도(Gradient) 매개변수를 0으로 만들고

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model(inputs)
         #size가 안맞네. padding문제인듯, 홀수 짝수 문제인듯

        loss = loss_fc(outputs.squeeze(1).float(), labels.float())

        if torch.isinf(loss) or torch.isnan(loss):
            print("loss = Nan or Inf error")
            continue
        loss.backward()
        SGD.step()

        total += batch_size
        running_loss += loss.item()
        print(f"steps = {i} : loss = {running_loss / (i+1)}")

    running_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_total = 0
    for i, data in enumerate(val_loader):##validation batch
        with torch.no_grad():
            inputs, labels = data
            labels = labels.to(device)
            inputs = inputs.to(device).unsqueeze(1)

           # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)

            loss = loss_fc(outputs.squeeze(1).float(), labels.float())

            val_total+=batch_size
            val_loss += loss.item()

            # plt.figure()
            # plt.plot(outputs.cpu().detach().squeeze(1).t().numpy())
            # plt.savefig(os.path.join(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\val_voice', val_loader.dataset.x[j].split('\\')[-1] + '.jpg'))
            # plt.clf()

    val_loss = val_loss / len(val_loader)
    print(f"epoch = {epoch} / {epochs} : total_loss = {running_loss} ||| val_loss = {val_loss}")

    torch.save(model,
               f'C:/Users/Joker/Desktop/gitwork/tako_voice/Model/noise_remover/noise_remover_{running_loss}_{val_loss}.pth')

    test_loss = 0.0
    test_total = 0
    for i, data in enumerate(test_loader):##validation batch
        with torch.no_grad():
            inputs, labels = data
            labels = labels.to(device)
            inputs = inputs.to(device).unsqueeze(1)

           # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)

            loss = loss_fc(outputs.squeeze(1).float(), labels.float())

            test_total+=batch_size
            test_loss += loss.item()

        for j in range(batch_size):
            plt.figure()
            plt.plot(outputs[j].cpu().detach().squeeze(1).t().numpy())
            plt.savefig(os.path.join(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\predicted_voice_jpg',
                                     test_loader.dataset.data_list[i+j].split('\\')[-1] + '.jpg'))
            plt.close('all')
            torchaudio.save(
                os.path.join(r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\predicted_voice',
                             test_loader.dataset.data_list[i+j].split('\\')[-1]),
                outputs[j].detach().cpu().squeeze(1), 22050, format='wav')

    test_loss = test_loss / len(test_loader)
    print(f"epoch = {epoch} / {epochs} : ||| test_loss = {test_loss}")

print('Finished Training')