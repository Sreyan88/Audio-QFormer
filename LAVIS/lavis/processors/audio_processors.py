"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# import torch
# import torchaudio
import torchaudio.transforms as transforms
from moviepy.editor import VideoFileClip
from omegaconf import OmegaConf
import torchaudio.compliance.kaldi as ta_kaldi

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.models.beats.Tokenizers import TokenizersConfig, Tokenizers

import torch
import torchaudio
import torchvision
import numpy as np
from contextlib import suppress
import torch.nn.functional as F

MAX_INT = registry.get("MAX_INT")


@registry.register_processor("beats_audio")
class BeatsAudioProcessor(BaseProcessor):
    def __init__(self, model_name, sampling_rate, n_frames, frame_length, is_eval):
        """
        Adapted from https://github.com/NINAnor/rare_species_detections/blob/main/BEATs/BEATs.py
        """
        super().__init__()

        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.n_frames = n_frames
        self.frame_length = frame_length
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.is_eval = is_eval

    def _load_audio(self, aupath):
        if aupath.endswith('.mp4'):
            video = VideoFileClip(aupath)
            audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
            if len(audio_np.shape) == 2:
                audio_np = audio_np.mean(axis=1)  # Convert to mono
            waveform = torch.tensor(audio_np).float()
            sr = self.sampling_rate
        else:
            waveform, sr = torchaudio.load(aupath)
            if waveform.shape[0] == 2: 
                waveform = torch.mean(waveform, dim=0)
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)
        return waveform

    def __call__(self, aupath, start_sec=None, end_sec=None):
        """
        Args:
            aupath: path to audio file
        Returns:
            torch.tensor: audio clip after transforms.
        """
        # Helper function to return empty tensor for invalid audio
        def empty_audio_tensor():
            return torch.zeros((self.n_frames, self.frame_length, 128))
        
        try:
            # Handle MP4 files
            if aupath.endswith('.mp4'):
                video = VideoFileClip(aupath)
                if start_sec is not None and end_sec is not None:
                    video = video.subclip(start_sec, end_sec)
                audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)  # Convert to mono
                waveform = torch.tensor(audio_np).float()
                sr = self.sampling_rate
            else:
                waveform, sr = torchaudio.load(aupath)

            # Validate waveform
            if len(waveform.shape) == 0:
                return empty_audio_tensor()

            # Convert stereo to mono
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0)

            # Resample waveform if necessary
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)

        except:
            return empty_audio_tensor()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform * 2**15

        # Compute fbank features
        try:
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sampling_rate,
                frame_length=25,
                frame_shift=10,
            )
            fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        except:
            return empty_audio_tensor()

        # Handle padding and frames extraction differently for eval and training modes
        if not self.is_eval:
            fbank_pad_len = self.frame_length * self.n_frames - fbank.shape[0]
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            fbank = fbank[:self.frame_length * self.n_frames]
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(self.n_frames)]
        else:
            fbank_pad_len = fbank.shape[0] % self.frame_length
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            curr_frames = fbank.shape[0] // self.frame_length
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(curr_frames)]

        return torch.cat(frames, dim=0)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            model_name=cfg.get("model_name", 'iter3'),
            sampling_rate=cfg.get("sampling_rate", 16000),
            n_frames=cfg.get("n_frames", 2),
            frame_length=cfg.get("frame_length", 512),
            is_eval=cfg.get("is_eval", False)
        )


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def get_mel(audio_data):
    # mel shape: (n_mels, T)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=1024,
        win_length=1024,
        hop_length=480,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=50,
        f_max=14000
    ).to(audio_data.device)
    
    mel = mel_tf(audio_data)

    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)

    return mel.T

def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, require_grad=False):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    require_grad: whether to require gradient for audio data.
        This is useful when we want to apply gradient-based classifier-guidance.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data)
                # split to three parts
                chunk_frames = max_len // 480 + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    # print(sample["mel_fusion"].shape)
    # print("---------------------")
    return sample



@registry.register_processor("htsat_audio")
class HtsatAudioProcessor(BaseProcessor):
    def __init__(self, model_name, sampling_rate, n_frames, frame_length, is_eval):
        """
        Adapted from https://github.com/NINAnor/rare_species_detections/blob/main/BEATs/BEATs.py
        """
        super().__init__()

        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.n_frames = n_frames
        self.frame_length = frame_length
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.is_eval = is_eval

    def _load_audio(self, aupath):

        waveform, sr = torchaudio.load(aupath)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=48000)
        audio_data = int16_to_float32_torch(float32_to_int16_torch(waveform[0]))

        waveform = get_audio_features({}, audio_data, 480000, "fusion", "repeatpad")

        return waveform

    def __call__(self, aupath, start_sec=None, end_sec=None):
        """
        Args:
            aupath: path to audio file
        Returns:
            torch.tensor: audio clip after transforms.
        """
        # Helper function to return empty tensor for invalid audio
        waveform, sr = torchaudio.load(aupath)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=48000)
        audio_data = int16_to_float32_torch(float32_to_int16_torch(waveform[0]))

        waveform = get_audio_features({}, audio_data, 480000, "fusion", "repeatpad")

        return waveform

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            model_name=cfg.get("model_name", 'iter3'),
            sampling_rate=cfg.get("sampling_rate", 16000),
            n_frames=cfg.get("n_frames", 2),
            frame_length=cfg.get("frame_length", 512),
            is_eval=cfg.get("is_eval", False)
        )