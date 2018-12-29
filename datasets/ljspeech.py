# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import apps.vocoder.audio as audio
from apps.vocoder.hparams import hparams

from nnmnkwii import preprocessing as P


def build_from_path(in_dir, out_dir, silence_threshold, fft_size, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            # In case of test file
            if not os.path.exists(wav_path):
                continue
            text = parts[2]
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text, silence_threshold, fft_size)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, silence_threshold, fft_size):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    if hparams.input_type != "raw":
        # Mu-law quantize
        out = P.mulaw_quantize(wav)

        # Trim silences
        start, end = audio.start_and_end_indices(out, silence_threshold)
        out = out[start:end]
        wav = wav[start:end]
        constant_value = P.mulaw_quantize(0, 256)
        out_dtype = np.int16
    else:
        out = wav
        constant_value = 0.
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_value)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    wav_id = os.path.basename(wav_path).split('.')[0]   # wav_id = wav_path.split('/')[-1].split('.')[0]

    # Write the spectrograms to disk:
    audio_filename = '{}-audio.npy'.format(wav_id)
    mel_filename = '{}-mel.npy'.format(wav_id)
    np.save(os.path.join(out_dir, audio_filename),out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_filename, mel_filename, timesteps, text