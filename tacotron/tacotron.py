# coding: utf-8

# Code based on https://github.com/keithito/tacotron/blob/master/models/tacotron.py

"""
모델 수정
1. prenet에서 dropout 적용 오류 수정
2. AttentionWrapper 적용 순서 오류 수정: keith ito 코드는 잘 구현되어 있음
3. BahdanauMonotonicAttention에서 normalize=True적용(2018년9월11일 적용)
4. BahdanauMonotonicAttention에서 memory_sequence_length 입력
5. synhesizer.py  input_lengths 계산오류. +1 해야 함.


"""



import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, BahdanauMonotonicAttention, LuongAttention
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper

from utils.infolog import log
from text.symbols import symbols

from .modules import *
from .helpers import TacoTestHelper, TacoTrainingHelper
from .rnn_wrappers import AttentionWrapper, DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper,BahdanauMonotonicAttention_hccho,LocationSensitiveAttention,GmmAttention


class Tacotron():
    def __init__(self, hparams):
        self._hparams = hparams


    def initialize(self, inputs, input_lengths, num_speakers, speaker_id,mel_targets=None, linear_targets=None, loss_coeff=None,
                    rnn_decoder_test_mode=False, is_randomly_initialized=False,):
        
        
        is_training2 = linear_targets is not None  # test에서 이게 True로 되는데, 이게 의도한 것인가???
        is_training = not rnn_decoder_test_mode
        
        
        self.is_randomly_initialized = is_randomly_initialized

        with tf.variable_scope('inference') as scope:
            hp = self._hparams
            batch_size = tf.shape(inputs)[0]

            # Embeddings(256)
            char_embed_table = tf.get_variable('embedding', [len(symbols), hp.embedding_size], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
            
            zero_pad = True
            if zero_pad:    # transformer에 구현되어 있는 거 보고, 가져온 로직.
                # <PAD> 0 은 embedding이 0으로 고정되고, train으로 변하지 않는다. 즉, 위의 get_variable에서 잡았던 변수의 첫번째 행(<PAD>)에 대응되는 것은 사용되지 않는 것이다)
                char_embed_table = tf.concat((tf.zeros(shape=[1, hp.embedding_size]),char_embed_table[1:, :]), 0)
            
            
            # [N, T_in, embedding_size]
            char_embedded_inputs = tf.nn.embedding_lookup(char_embed_table, inputs)

            self.num_speakers = num_speakers
            if self.num_speakers > 1:
                if hp.speaker_embedding_size != 1:    # speaker_embedding_size = f(16)
                    speaker_embed_table = tf.get_variable('speaker_embedding',[self.num_speakers, hp.speaker_embedding_size], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
                    # [N, T_in, speaker_embedding_size]
                    speaker_embed = tf.nn.embedding_lookup(speaker_embed_table, speaker_id)

                if hp.model_type == 'deepvoice':
                    if hp.speaker_embedding_size == 1:
                        before_highway = get_embed(speaker_id, self.num_speakers, hp.enc_prenet_sizes[-1], "before_highway")    # 'enc_prenet_sizes': [f(256), f(128)]
                        encoder_rnn_init_state = get_embed(speaker_id, self.num_speakers, hp.enc_rnn_size * 2, "encoder_rnn_init_state")

                        attention_rnn_init_state = get_embed(speaker_id, self.num_speakers, hp.attention_state_size, "attention_rnn_init_state")
                        decoder_rnn_init_states = [get_embed(speaker_id, self.num_speakers, hp.dec_rnn_size, "decoder_rnn_init_states{}".format(idx + 1))  
                                                   for idx in range(hp.dec_layer_num)]
                    else:
                        deep_dense = lambda x, dim: tf.layers.dense(x, dim, activation=tf.nn.softsign)   # softsign: x / (abs(x) + 1)

                        before_highway = deep_dense( speaker_embed, hp.enc_prenet_sizes[-1])
                        encoder_rnn_init_state = deep_dense( speaker_embed, hp.enc_rnn_size * 2)

                        attention_rnn_init_state = deep_dense(speaker_embed, hp.attention_state_size)
                        decoder_rnn_init_states = [deep_dense(speaker_embed, hp.dec_rnn_size) for _ in range(hp.dec_layer_num)]

                    speaker_embed = None # deepvoice does not use speaker_embed directly
                elif hp.model_type == 'simple':
                    # simple model은 speaker_embed를 DecoderPrenetWrapper,ConcatOutputAndAttentionWrapper에 각각 넣어서 concat하는 방식이다.
                    before_highway = None
                    encoder_rnn_init_state = None
                    attention_rnn_init_state = None
                    decoder_rnn_init_states = None
                else:
                    raise Exception(" [!] Unkown multi-speaker model type: {}".format(hp.model_type))
            else:
                # self.num_speakers =1인 경우
                speaker_embed = None
                before_highway = None
                encoder_rnn_init_state = None   # bidirectional GRU의 init state
                attention_rnn_init_state = None
                decoder_rnn_init_states = None

            ##############
            # Encoder
            ##############

            # [N, T_in, enc_prenet_sizes[-1]]
            prenet_outputs = prenet(char_embedded_inputs, is_training, hp.enc_prenet_sizes, hp.dropout_prob, scope='prenet')  # 'enc_prenet_sizes': [f(256), f(128)],  dropout_prob = 0.5
            # ==> (N, T_in, 128)
            
            
            # enc_rnn_size = 128
            encoder_outputs = cbhg(prenet_outputs, input_lengths, is_training,hp.enc_bank_size, hp.enc_bank_channel_size,
                                    hp.enc_maxpool_width, hp.enc_highway_depth, hp.enc_rnn_size,hp.enc_proj_sizes, hp.enc_proj_width,
                                    scope="encoder_cbhg",before_highway=before_highway,encoder_rnn_init_state=encoder_rnn_init_state)

            ##############
            # Attention
            ##############

            # For manaul control of attention
            self.is_manual_attention = tf.placeholder(tf.bool, shape=(), name='is_manual_attention',)
            self.manual_alignments = tf.placeholder(tf.float32, shape=[None, None, None], name="manual_alignments",)


            # single: attention_size = 128
            if hp.attention_type == 'bah_mon':
                attention_mechanism = BahdanauMonotonicAttention(hp.attention_size, encoder_outputs,memory_sequence_length=input_lengths,normalize=False)
            elif hp.attention_type == 'bah_mon_norm':  # hccho 추가
                attention_mechanism = BahdanauMonotonicAttention(hp.attention_size, encoder_outputs,memory_sequence_length = input_lengths, normalize=True) 
            elif hp.attention_type == 'loc_sen': # Location Sensitivity Attention
                attention_mechanism = LocationSensitiveAttention(hp.attention_size, encoder_outputs,memory_sequence_length = input_lengths)
            elif hp.attention_type == 'gmm': # GMM Attention
                attention_mechanism = GmmAttention(hp.attention_size, memory=encoder_outputs,memory_sequence_length = input_lengths)
            elif hp.attention_type == 'bah_mon_norm_hccho':
                attention_mechanism = BahdanauMonotonicAttention_hccho(hp.attention_size, encoder_outputs,normalize=True)     
            elif hp.attention_type == 'bah_norm':
                attention_mechanism = BahdanauAttention(hp.attention_size, encoder_outputs,memory_sequence_length=input_lengths, normalize=True)
            elif hp.attention_type == 'luong_scaled':
                attention_mechanism = LuongAttention( hp.attention_size, encoder_outputs,memory_sequence_length=input_lengths, scale=True)
            elif hp.attention_type == 'luong':
                attention_mechanism = LuongAttention(hp.attention_size, encoder_outputs,memory_sequence_length=input_lengths)
            elif hp.attention_type == 'bah':
                attention_mechanism = BahdanauAttention(hp.attention_size, encoder_outputs,memory_sequence_length=input_lengths)
            else:
                raise Exception(" [!] Unkown attention type: {}".format(hp.attention_type))


            # DecoderPrenetWrapper, attention_mechanism을 결합하여 AttentionWrapper를 만든다.
            # carpedm20은  tensorflow 소스를코드를 가져와서 AttentionWrapper를 새로 구현했지만,  keith Ito는 tensorflow AttentionWrapper를 그냥 사용했다.
            attention_cell = AttentionWrapper(GRUCell(hp.attention_state_size),attention_mechanism, self.is_manual_attention,self.manual_alignments,
                                              initial_cell_state=attention_rnn_init_state,alignment_history=True,output_attention=False)  # output_attention=False 에 주목, attention_layer_size에 값을 넣지 않았다. 그래서 attention = contex vector가 된다.


            # attention_state_size = 256
            dec_prenet_outputs = DecoderPrenetWrapper(attention_cell , speaker_embed, is_training, hp.dec_prenet_sizes, hp.dropout_prob)  # dec_prenet_sizes =  [f(256), f(128)]



            # Concatenate attention context vector and RNN cell output into a 512D vector.
            # [N, T_in, attention_size+attention_state_size]
            
            #dec_prenet_outputs의 다음 cell에 전달하는 AttentionWrapperState의 member (attention,cell_state, ...)에서 attention과 output을 concat하여 output으로 내보낸다.
            # output이 output은 cell_state와 같기 때문에, concat [ output(=cell_state) | attention ]
            concat_cell = ConcatOutputAndAttentionWrapper(dec_prenet_outputs, embed_to_concat=speaker_embed)  # concat(output,attention,speaker_embed)해서 새로운 output을 만든다.
                        
            # Decoder (layers specified bottom to top):  dec_rnn_size= 256
            cells = [OutputProjectionWrapper(concat_cell, hp.dec_rnn_size)]   # OutputProjectionWrapper는 논문에 언급이 없는 것 같은데...
            for _ in range(hp.dec_layer_num):  # hp.dec_layer_num = 2
                cells.append(ResidualWrapper(GRUCell(hp.dec_rnn_size)))

            # [N, T_in, 256]
            decoder_cell = MultiRNNCell(cells, state_is_tuple=True)

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.reduction_factor)   # 여기에 stop token도 나올 수 있도록...수정하면 되지 않을까???   (hp.num_mels+1) * hp.reduction_factor
            decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32) # 여기서 zero_state를 부르면, 위의 AttentionWrapper에서 이미 넣은 준 값도 포함되어 있다.

            if hp.model_type == "deepvoice":
                # decoder_init_state[0] : AttentionWrapperState
                # = cell_state + attention + time + alignments + alignment_history
                # decoder_init_state[0][0] = attention_rnn_init_state (already applied: AttentionWrapper의 initial_cell_state를 이미 넣어 주었다. )
                decoder_init_state = list(decoder_init_state)

                for idx, cell in enumerate(decoder_rnn_init_states):
                    shape1 = decoder_init_state[idx + 1].get_shape().as_list()
                    shape2 = cell.get_shape().as_list()
                    if shape1 != shape2:
                        raise Exception(" [!] Shape {} and {} should be equal".format(shape1, shape2))
                    decoder_init_state[idx + 1] = cell

                decoder_init_state = tuple(decoder_init_state)

            if is_training2:
                # rnn_decoder_test_mode = True if test mode,  train mode에서는 False
                helper = TacoTrainingHelper(  inputs, mel_targets, hp.num_mels, hp.reduction_factor, rnn_decoder_test_mode)  # inputs은 batch_size 계산에만 사용됨
            else:
                helper = TacoTestHelper( batch_size, hp.num_mels, hp.reduction_factor)

            (decoder_outputs, _), final_decoder_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(BasicDecoder(output_cell, helper, decoder_init_state),maximum_iterations=hp.max_iters)  # max_iters=200
            
            # [N, T_out, M]
            mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels])

            # Add post-processing CBHG:
            # [N, T_out, 256]
            #post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training)
            post_outputs = cbhg(mel_outputs, None, is_training,hp.post_bank_size, hp.post_bank_channel_size, hp.post_maxpool_width, hp.post_highway_depth, hp.post_rnn_size,
                                hp.post_proj_sizes, hp.post_proj_width,scope='post_cbhg')

            if speaker_embed is not None and hp.model_type == 'simple':
                expanded_speaker_emb = tf.expand_dims(speaker_embed, [1])
                tiled_speaker_embedding = tf.tile(expanded_speaker_emb, [1, tf.shape(post_outputs)[1], 1])

                # [N, T_out, 256 + alpha]
                post_outputs = tf.concat([tiled_speaker_embedding, post_outputs], axis=-1)

            linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)    # [N, T_out, F(1025)]

            # Grab alignments from the final decoder state:
            # MultiRNNCell이 3단이기 때문에, final_decoder_state는 len 3 tuple이다.  ==> final_decoder_state[0]
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])  # batch_size, text length(encoder), target length(decoder)


            self.inputs = inputs
            self.speaker_id = speaker_id
            self.input_lengths = input_lengths
            self.loss_coeff = loss_coeff
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            self.final_decoder_state = final_decoder_state

            log('='*40)
            log(' model_type: %s' % hp.model_type)
            log('='*40)

            log('Initialized Tacotron model. Dimensions: ')
            log('    embedding:                %d' % char_embedded_inputs.shape[-1])
            if speaker_embed is not None:
                log('    speaker embedding:        %d' % speaker_embed.shape[-1])
            else:
                log('    speaker embedding:        None')
            log('    prenet out:               %d' % prenet_outputs.shape[-1])
            log('    encoder out:              %d' % encoder_outputs.shape[-1])
            log('    attention out:            %d' % attention_cell.output_size)
            log('    concat attn & out:        %d' % concat_cell.output_size)
            log('    decoder cell out:         %d' % decoder_cell.output_size)
            log('    decoder out (%d frames):  %d' % (hp.reduction_factor, decoder_outputs.shape[-1]))
            log('    decoder out (1 frame):    %d' % mel_outputs.shape[-1])
            log('    postnet out:              %d' % post_outputs.shape[-1])
            log('    linear out:               %d' % linear_outputs.shape[-1])


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            mel_loss = tf.abs(self.mel_targets - self.mel_outputs)

            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            expanded_loss_coeff = tf.expand_dims(tf.expand_dims(self.loss_coeff, [-1]), [-1])

            if hp.prioritize_loss:
                # Prioritize loss for frequencies.
                upper_priority_freq = int(5000 / (hp.sample_rate * 0.5) * hp.num_freq)
                lower_priority_freq = int(165 / (hp.sample_rate * 0.5) * hp.num_freq)

                l1_priority= l1[:,:,lower_priority_freq:upper_priority_freq]

                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + \
                        0.5 * tf.reduce_mean(l1 * expanded_loss_coeff) + 0.5 * tf.reduce_mean(l1_priority * expanded_loss_coeff)
                self.linear_loss = tf.reduce_mean( 0.5 * (tf.reduce_mean(l1) + tf.reduce_mean(l1_priority)))
            else:
                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + tf.reduce_mean(l1 * expanded_loss_coeff)   # optimize할 때는 self.loss를 사용하고, 출력할 때는 아래의 loss_without_coeff를 사용함
                self.linear_loss = tf.reduce_mean(l1)

            self.mel_loss = tf.reduce_mean(mel_loss)
            self.loss_without_coeff = self.mel_loss + self.linear_loss


    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams

            step = tf.cast(global_step + 1, dtype=tf.float32)

            if hp.decay_learning_rate_mode == 0:
                if self.is_randomly_initialized:
                    warmup_steps = 4000.0
                else:
                    warmup_steps = 40000.0
                self.learning_rate = hp.tacotron_initial_learning_rate * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
            elif hp.decay_learning_rate_mode == 1:
                self.learning_rate = hp.tacotron_initial_learning_rate * tf.train.exponential_decay(1., step, 3000, 0.95)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),global_step=global_step)

    def get_dummy_feed_dict(self):
        feed_dict = { self.is_manual_attention: False, self.manual_alignments: np.zeros([1, 1, 1]),}
        return feed_dict














