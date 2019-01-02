# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name = "Tacotron-Wavenet-Vocoder",
    
    # tacotron hyper parameter
    
    cleaners = 'korean_cleaners',  # 'korean_cleaners'   or 'english_cleaners'
    
    
    skip_path_filter = False, # npz파일에서 불필요한 것을 거르는 작업을 할지 말지 결정. receptive_field 보다 짧은 data를 걸러야 하기 때문에 해 줘야 한다.
    use_lws = False,
    
    # Audio
    sample_rate = 24000,  # 
    
    # shift can be specified by either hop_size(우선) or frame_shift_ms
    hop_size = 300,             # frame_shift_ms = 12.5ms
    fft_size=2048,   # n_fft. 주로 1024로 되어있는데, tacotron에서 2048사용
    win_size = 1200,   # 50ms
    num_mels=80,

    #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize = True, #whether to apply filter
    preemphasis = 0.97,
    min_level_db = -100,
    ref_level_db = 20,
    signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
    
        
    rescaling=True,
    rescaling_max=0.999, 
    
    trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_fft_size = 512, 
    trim_hop_size = 128,
    trim_top_db = 23,
    
    
    
    
    clip_mels_length = True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)   
    max_mel_frames = 1000,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
    
    

    l2_regularization_strength = 0,  # Coefficient in the L2 regularization.
    sample_size = 15000,              # Concatenate and cut audio samples to this many samples
    silence_threshold = 0,             # Volume threshold below which to trim the start and the end from the training set samples. e.g. 2

    
    filter_width = 2,
    gc_channels = 32,                  # global_condition_vector의 차원. 이것 지정함으로써, global conditioning을 모델에 반영하라는 의미가 된다.
    
    input_type="raw",    # 'mulaw-quantize', 'mulaw', 'raw',   mulaw, raw 2가지는 scalar input
    scalar_input = True,   # input_type과 맞아야 함.
    
    
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    residual_channels = 32,
    dilation_channels = 32,
    quantization_channels = 256,
    out_channels = 30,  # discretized_mix_logistic_loss를 적용하기 때문에, 3의 배수
    skip_channels = 512,
    use_biases = True,
    
    initial_filter_width = 32,
    upsample_factor=[5, 5, 12],   # np.prod(upsample_factor) must equal to hop_size



    # wavenet training hp
    wavenet_batch_size = 8,            # 16--> OOM. wavenet은 batch_size가 고정되어야 한다.
    store_metadata = False,
    num_steps = 200000,                # Number of training steps

    #Learning rate schedule
    wavenet_learning_rate = 1e-3, #wavenet initial learning rate
    wavenet_decay_rate = 0.5, #Only used with 'exponential' scheme. Defines the decay rate.
    wavenet_decay_steps = 300000, #Only used with 'exponential' scheme. Defines the decay steps.

    #Regularization parameters
    wavenet_clip_gradients = False, #Whether the clip the gradients during wavenet training.


    
    optimizer = 'adam',
    momentum = 0.9,                   # 'Specify the momentum to be used by sgd or rmsprop optimizer. Ignored by the adam optimizer.
    max_checkpoints = 3,             # 'Maximum amount of checkpoints that will be kept alive. Default: '    


    ####################################
    ####################################
    ####################################
    # TACOTRON HYPERPARAMETERS
    
    # Training
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    use_fixed_test_inputs = False,
    
    tacotron_initial_learning_rate = 1e-3,
    decay_learning_rate_mode = 0, # True in deepvoice2 paper
    initial_data_greedy = True,
    initial_phase_step = 8000,   # 여기서 지정한 step 이전에는 data_dirs의 각각의 디렉토리에 대하여 같은 수의 example을 만들고, 이후, weght 비듈에 따라 ... 즉, 아래의 'main_data_greedy_factor'의 영향을 받는다.
    main_data_greedy_factor = 0,
    main_data = [''],    # 이곳에 있는 directory 속에 있는 data는 가중치를 'main_data_greedy_factor' 만큼 더 준다.
    prioritize_loss = False,    
    

    # Model
    model_type = 'deepvoice', # [single, simple, deepvoice]
    speaker_embedding_size  = 16, 

    embedding_size = 256,    # 'ᄀ', 'ᄂ', 'ᅡ' 에 대한 embedding dim
    dropout_prob = 0.5,

    # Encoder
    enc_prenet_sizes = [256, 128],
    enc_bank_size = 16,  # cbhg에서 conv1d의 out kernel size를 1,2,..., enc_bank_size 까지 반복 적용
    enc_bank_channel_size = 128,   # cbhg에서 conv1d의 out channel size
    enc_maxpool_width = 2,   # cbhg에서 max pooling size
    enc_highway_depth = 4,
    enc_rnn_size = 128,
    enc_proj_sizes = [128, 128], # cbhg, projection layer (2번째 conv1d), channel size
    enc_proj_width = 3,                #cbhg, projection layer (2번째 conv1d), kernel size

    # Attention
    attention_type = 'bah_mon_norm', 
    attention_size = 256,
    attention_state_size = 256,

    # Decoder recurrent network
    dec_layer_num = 2,
    dec_rnn_size = 256,

    # Decoder
    dec_prenet_sizes = [256, 128],
    post_bank_size = 8,
    post_bank_channel_size = 128,
    post_maxpool_width = 2,
    post_highway_depth = 4,
    post_rnn_size = 128,
    post_proj_sizes = [256, 80], # num_mels=80
    post_proj_width = 3,

    reduction_factor = 5,


    # Eval
    min_tokens = 30,  #originally 50, 30 is good for korean,  text를 token으로 쪼갰을 때, 최소 길이 이상되어야 train에 사용
    min_iters = 30,  # min_n_frame = reduction_factor * min_iters, reduction_factor와 곱해서 min_n_frame을 설정한다.
    max_iters = 200,
    skip_inadequate = False,

    griffin_lim_iters = 60,
    power = 1.5, 


    # 사용안되는 것들인데, error방지
    recognition_loss_coeff = 0.2,   # 이게 1이 아니면, 'ignore_recognition_level' = 1,2에 걸리는 data는 무시됨.
    ignore_recognition_level = 0


)

if hparams.use_lws:
    # Does not work if fft_size is not multiple of hop_size!!
    # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
    hparams.sample_rate = 20480  # 
    
    # shift can be specified by either hop_size(우선) or frame_shift_ms
    hparams.hop_size = 256             # frame_shift_ms = 12.5ms
    hparams.frame_shift_ms=None      # hop_size=  sample_rate *  frame_shift_ms / 1000
    hparams.fft_size=2048   # 주로 1024로 되어있는데, tacotron에서 2048사용==> output size = 1025
    hparams.win_size = None # 256x4 --> 50ms
else:
    # 미리 정의되 parameter들로 부터 consistant하게 정의해 준다.
    hparams.num_freq = int(hparams.fft_size/2 + 1)
    hparams.frame_shift_ms = hparams.hop_size * 1000.0/ hparams.sample_rate      # hop_size=  sample_rate *  frame_shift_ms / 1000
    hparams.frame_length_ms = hparams.win_size * 1000.0/ hparams.sample_rate 

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
