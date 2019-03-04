# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os,json
from utils import audio
from text import text_to_sequence


def build_from_path(hparams, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of npz files
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    path = os.path.join(in_dir, 'son-recognition-All.json')
    
    with open(path,encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)
        for key, text in data.items():
            wav_path = key.strip().split('/')
            wav_path = os.path.join(in_dir, 'audio', '%s' % wav_path[-1])
            # In case of test file
            if not os.path.exists(wav_path):
                continue
            futures.append(executor.submit(partial(_process_utterance, out_dir, wav_path, text,hparams)))
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]
#     result = []
#     for future in tqdm(futures):
#         if future.result() is not None:
#             result.append(future.result())
#          
#     return result

def _process_utterance(out_dir, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    #rescale wav
    if hparams.rescaling:   # hparams.rescale = True
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    #M-AILABS extra silence specific
    if hparams.trim_silence:  # hparams.trim_silence = True
        wav = audio.trim_silence(wav, hparams)   # Trim leading and trailing silence

    #Mu-law quantize, default 값은 'raw'
    if hparams.input_type=='mulaw-quantize':
        #[0, quantize_channels)
        out = audio.mulaw_quantize(wav, hparams.quantize_channels)

        #Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif hparams.input_type=='mulaw':
        #[-1, 1]
        out = audio.mulaw(wav, hparams.quantize_channels)
        constant_values = audio.mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:  # raw
        #[-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:   # hparams.max_mel_frames = 1000, hparams.clip_mels_length = True
        return None

    #Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    #sanity check
    assert linear_frames == mel_frames

    if hparams.use_lws:    # hparams.use_lws = False
        #Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.fft_size if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        #Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        pad = audio.librosa_pad_lr(wav, hparams.fft_size, audio.get_hop_size(hparams))

        #Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, pad, mode='reflect')

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    wav_id = os.path.splitext(os.path.basename(wav_path))[0]
    
    # Write the spectrograms to disk:
    audio_filename = '{}-audio.npy'.format(wav_id)
    mel_filename = '{}-mel.npy'.format(wav_id)
    linear_filename = '{}-linear.npy'.format(wav_id)
    npz_filename = '{}.npz'.format(wav_id)
    npz_flag=True
    if npz_flag:
        # Tacotron 코드와 맞추기 위해, 같은 key를 사용한다.
        data = {
            'audio': out.astype(out_dtype),
            'mel': mel_spectrogram.T,  
            'linear': linear_spectrogram.T,
            'time_steps': time_steps,
            'mel_frames': mel_frames,
            'text': text,
            'tokens': text_to_sequence(text),   # eos(~)에 해당하는 "1"이 끝에 붙는다.
            'loss_coeff': 1  # For Tacotron
        }
        
        np.savez(os.path.join(out_dir,npz_filename ), **data, allow_pickle=False)
    else:
        np.save(os.path.join(out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
        np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
        np.save(os.path.join(out_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text,npz_filename)
