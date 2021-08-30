import os
import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
#from text import text_to_sequence


class PPGMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    #def __init__(self, audiopaths_and_text, hparams):
    #    self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
    #    self.text_cleaners = hparams.text_cleaners
    #    self.max_wav_value = hparams.max_wav_value
    #    self.sampling_rate = hparams.sampling_rate
    #    self.load_mel_from_disk = hparams.load_mel_from_disk
    #    self.stft = layers.TacotronSTFT(
    #        hparams.filter_length, hparams.hop_length, hparams.win_length,
    #        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    #        hparams.mel_fmax)
    #    random.seed(hparams.seed)
    #    random.shuffle(self.audiopaths_and_text)
    def __init__(self, hparams, split):
        self.ppg_dir = hparams["ppg_dir"]
        self.mel_dir = hparams["mel_dir"]
        if split == "train":
            self.filelist_path = hparams["training_file_list"]
        elif split == "val":
            self.filelist_path = hparams["validation_file_list"]
        with open(self.filelist_path, "r") as f:
            self.file_names = f.read().strip().split("\n")
        random.seed(hparams["seed"])
        random.shuffle(self.file_names)

    #def get_mel_text_pair(self, audiopath_and_text):
    #    # separate filename and text
    #    audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
    #    text = self.get_text(text)
    #    mel = self.get_mel(audiopath)
    #    return (text, mel)
    def get_ppg_mel_pair(self, index):
        file_name = self.file_names[index]
        ppg = torch.transpose(torch.load(os.path.join(self.ppg_dir, file_name)), 0, 1)
        mel = torch.load(os.path.join(self.mel_dir, file_name))
        return (ppg, mel)

    def __getitem__(self, index):
        return self.get_ppg_mel_pair(index)

    def __len__(self):
        return len(self.file_names)


class PPGMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[0] for x in batch]),
            dim=0, descending=True)
        #print("input_lengths", input_lengths)
        #print("ids_sorted_decreasing", ids_sorted_decreasing)
        max_input_len = input_lengths[0]
        #print("max_input_len:", max_input_len)

        ##text_padded = torch.LongTensor(len(batch), max_input_len)
        ##text_padded.zero_()
        ##for i in range(len(ids_sorted_decreasing)):
        ##    text = batch[ids_sorted_decreasing[i]][0]
        ##    text_padded[i, :text.size(0)] = text
        #print("batch type:", type(batch))
        #print("batch[0] type:", type(batch[0]))
        #print("batch[0][0] type:", type(batch[0][0]))
        #print("batch[0][0][0] type:", type(batch[0][0][0]))
        num_phons = batch[0][0].shape[1]
        #print("Number of phonemes:", num_phons)
        ppg_padded = torch.FloatTensor(len(batch), max_input_len, num_phons)
        #print("ppg_padded shape", ppg_padded.shape)
        ppg_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            ppg = batch[ids_sorted_decreasing[i]][0]
            #print("ppg shape", ppg.shape)
            length = ppg.shape[0]
            #print(length)
            #print("target shape", ppg_padded[i, :length, :].shape)
            ppg_padded[i, :length, :] = ppg
            del ppg, length

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            #print("mel shape", mel.shape)
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            del mel

        return ppg_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
