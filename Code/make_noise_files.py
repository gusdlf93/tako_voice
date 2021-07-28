import torchaudio
import random
import os
import torch
import matplotlib as plt
import numpy as np

def make_bgs(bgs):
    for _, i in enumerate(bgs):
        channel = 0
        new_sample_rate = 22050
        time = 120
        if _ == 0:
            waveform, sample_rate = torchaudio.load(i)
            re_wave = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))
            appended_waveform = re_wave[0][:new_sample_rate*time]
        else :
            waveform, sample_rate = torchaudio.load(i)
            re_wave = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))
            appended_waveform = torch.cat((appended_waveform, re_wave[0][:new_sample_rate*time]), dim=0)
    return appended_waveform

def recursive_find(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".ogg")):# or file.endswith(".ogg"):
                file_list.append(os.path.join(root, file))
    return file_list

bg_path = r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\resampled_bg'
vc_path = r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\LJSpeech-1.1\wavs'

bgs = recursive_find(bg_path)
vcs = recursive_find(vc_path)

bgs_waveform = make_bgs(bgs)

def make_vcs(vcs, bgs_waveform, dest):
    epsilon = 1e-9
    for _, i in enumerate(vcs):
        channel = 0
        new_sample_rate = 22050

        waveform, sample_rate = torchaudio.load(i)

        bgs_time = int(len(bgs_waveform) / sample_rate)
        vcs_time = int(len(waveform[channel, :]) / sample_rate) + 1

        re_wave = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))

        rtn = random.randint(0, bgs_time - vcs_time) #random time number

        bgs_median = torch.median(bgs_waveform[rtn*new_sample_rate : (rtn+vcs_time)*new_sample_rate]) + epsilon
        vcs_median = torch.median(re_wave[0][:vcs_time*new_sample_rate]) + epsilon
        max = torch.max(bgs_median, vcs_median)
        min = torch.min(bgs_median, vcs_median)

        re_wave = bgs_waveform[rtn*new_sample_rate : rtn*new_sample_rate + len(waveform[channel, :])] * (min/max) +\
                  re_wave[0][:len(waveform[channel, :])]
        if torch.max(torch.isnan(re_wave)) == True:
            print("error")
        torchaudio.save(os.path.join(dest, i.split('\\')[-1]), re_wave.reshape(1, re_wave.shape[0]) , new_sample_rate)

dest = r'C:\Users\Joker\Desktop\gitwork\tako_voice\Data\conversed_voice'

vcs_waveform = make_vcs(vcs, bgs_waveform, dest)