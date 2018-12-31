# coding: utf-8
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from tensorflow.contrib.training.python.training.hparam import HParams


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller   --> libosa type error(bug) 극복
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def trim_silence(wav, hparams):
    '''Trim leading and trailing silence

    Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    #Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:  # Tacotron에서 항상적용했다.
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_spectrogram_tensorflow(spectrogram,hparams):
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram,hparams) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power),hparams)


def inv_spectrogram(spectrogram,hparams):
    S = _db_to_amp(_denormalize(spectrogram,hparams) + hparams.ref_level_db)    # Convert back to linear.  spectrogram: (num_freq,length)
    return inv_preemphasis(_griffin_lim(S ** hparams.power,hparams),hparams.preemphasis, hparams.preemphasize)                 # Reconstruct phase



def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.fft_size, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.fft_size, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    '''compute right padding (final frame)
    '''
    return int(fsize // 2)


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    #assert hparams.fmax <= hparams.sample_rate // 2
    
    #fmin: Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    #fmax: 7600, To be increased/reduced depending on data.
    #return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, n_mels=hparams.num_mels,fmin=hparams.fmin, fmax=hparams.fmax)
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, n_mels=hparams.num_mels)  # fmin=0, fmax= sample_rate/2.0

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
             -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
 
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))
 
def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
 
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

# 김태훈 구현. 이 차이 때문에 호환이 되지 않는다.
# def _normalize(S,hparams):
#     return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)  # min_level_db = -100
# 
# def _denormalize(S,hparams):
#     return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

#From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def mulaw(x, mu=256):
    """Mu-Law companding
    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) ln (1 + mu |x|) / ln (1 + mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)
    .. math::
        f^{-1}(x) = sign(y) (1 / mu) (1 + mu)^{|y|} - 1)
    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> y = P.mulaw_quantize(x)
        >>> print(y.min(), y.max(), y.dtype)
        15 246 int64
    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    mu = mu-1
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize
    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncompressed signal ([-1, 1])
    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> x_hat = P.inv_mulaw_quantize(P.mulaw_quantize(x))
        >>> x_hat = (x_hat * 32768).astype(np.int16)
    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
    """
    # [0, m) to [-1, 1]
    mu = mu-1
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)

def _sign(x):
    #wrapper to support tensorflow tensors/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if (isnumpy or isscalar) else tf.sign(x)


def _log1p(x):
    #wrapper to support tensorflow tensors/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if (isnumpy or isscalar) else tf.log1p(x)


def _abs(x):
    #wrapper to support tensorflow tensors/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if (isnumpy or isscalar) else tf.abs(x)


def _asint(x):
    #wrapper to support tensorflow tensors/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else tf.cast(x, tf.int32)


def _asfloat(x):
    #wrapper to support tensorflow tensors/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else tf.cast(x, tf.float32)

def frames_to_hours(n_frames,hparams):
    return sum((n_frame for n_frame in n_frames)) * hparams.frame_shift_ms / (3600 * 1000)

def get_duration(audio,hparams):
    return librosa.core.get_duration(audio, sr=hparams.sample_rate)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _denormalize_tensorflow(S,hparams):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _griffin_lim_tensorflow(S,hparams):
    with tf.variable_scope('griffinlim'):
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex,hparams)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y,hparams)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles,hparams)
        return tf.squeeze(y, 0)

def _istft_tensorflow(stfts,hparams):
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_tensorflow(signals,hparams):
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)

def _stft_parameters(hparams):
    n_fft = (hparams.num_freq - 1) * 2  # hparams.num_freq = 1025
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)  # hparams.frame_shift_ms = 12.5
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)  # hparams.frame_length_ms = 50
    return n_fft, hop_length, win_length
