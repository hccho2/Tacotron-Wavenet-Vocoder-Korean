#  coding: utf-8
"""
- train data를 speaker를 분리된 디렉토리로 받아서, speaker id를 디렉토리별로 부과.
- file name에서 speaker id를 추론하는 방식이 아님.

"""

from __future__ import print_function

import argparse

import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel
from datasets import DataFeederWavenet
from hparams import hparams
from utils import validate_directories,load,save,infolog

tf.logging.set_verbosity(tf.logging.ERROR)
EPSILON = 0.001
log = infolog.log
def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    
    
    parser = argparse.ArgumentParser(description='WaveNet example network')
    
    DATA_DIRECTORY =  '.\\data\\moon,.\\data\\son'
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory containing the VCTK corpus.')


    LOGDIR = None
    #LOGDIR = './/logdir-wavenet//train//2018-12-21T22-58-10'

    parser.add_argument('--logdir', type=str, default=LOGDIR,help='Directory in which to store the logging information for TensorBoard. If the model already exists, it will restore the state and will continue training. Cannot use with --logdir_root and --restore_from.')
    
    
    parser.add_argument('--logdir_root', type=str, default=None,help='Root directory to place the logging output and generated model. These are stored under the dated subdirectory of --logdir_root. Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,help='Directory in which to restore the model from. This creates the new model under the dated directory in --logdir_root. Cannot use with --logdir.')
    
    
    CHECKPOINT_EVERY = 1000   # checkpoint 저장 주기
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    
    
    
   
    
    config = parser.parse_args()  # command 창에서 입력받을 수 있는 조건
    config.data_dir = config.data_dir.split(",")
    
    try:
        directories = validate_directories(config,hparams)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from


    log_path = os.path.join(logdir, 'train.log')
    infolog.init(log_path, logdir)


    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create coordinator.
    coord = tf.train.Coordinator()
    num_speakers = len(config.data_dir)
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = hparams.silence_threshold if hparams.silence_threshold > EPSILON else None
        gc_enable = num_speakers > 1
        
        # AudioReader에서 wav 파일을 잘라 input값을 만든다. receptive_field길이만큼을 앞부분에 pad하거나 앞조각에서 가져온다. (receptive_field+ sample_size)크기로 자른다.
        reader = DataFeederWavenet(coord,config.data_dir,batch_size=hparams.wavenet_batch_size,receptive_field=WaveNetModel.calculate_receptive_field(hparams.filter_width, hparams.dilations,hparams.scalar_input, hparams.initial_filter_width),
                                gc_enable= gc_enable)
        if gc_enable:
            audio_batch, lc_batch, gc_id_batch = reader.inputs_wav, reader.local_condition, reader.speaker_id
        else:
            audio_batch, lc_batch = reader.inputs_wav, self.local_condition

    # Create network.
    net = WaveNetModel(
        batch_size=hparams.wavenet_batch_size,
        dilations=hparams.dilations,
        filter_width=hparams.filter_width,
        residual_channels=hparams.residual_channels,
        dilation_channels=hparams.dilation_channels,
        quantization_channels=hparams.quantization_channels,
        out_channels =hparams.out_channels,
        skip_channels=hparams.skip_channels,
        use_biases=hparams.use_biases,  #  True
        scalar_input=hparams.scalar_input,
        initial_filter_width=hparams.initial_filter_width,
        global_condition_channels=hparams.gc_channels,
        global_condition_cardinality=num_speakers,
        local_condition_channels=hparams.num_mels,
        upsample_factor=hparams.upsample_factor,
        train_mode=True)

    if hparams.l2_regularization_strength == 0:
        hparams.l2_regularization_strength = None
        
       
    net.add_loss(input_batch=audio_batch,local_condition=lc_batch, global_condition_batch=gc_id_batch, l2_regularization_strength=hparams.l2_regularization_strength)
    net.add_optimizer(hparams,global_step)

    run_metadata = tf.RunMetadata()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))  # log_device_placement=False --> cpu/gpu 자동 배치.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=hparams.max_checkpoints)  # 최대 checkpoint 저장 갯수 지정
    
    try:
        start_step = load(saver, sess, restore_from)  # checkpoint load
        if is_overwritten_training or start_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            zero_step_assign = tf.assign(global_step, 0)
            sess.run(zero_step_assign)

           
    except:
        print("Something went wrong while restoring checkpoint. We will terminate training to avoid accidentally overwriting the previous model.")
        raise


    ###########

    start_step = sess.run(global_step)
    last_saved_step = start_step
    try:        
        reader.start_in_session(sess,start_step)
        while not coord.should_stop():
            
            start_time = time.time()
            if hparams.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                log('Storing metadata')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                step, loss_value, _ = sess.run([global_step, net.loss, net.optimize],options=run_options,run_metadata=run_metadata)

                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                step, loss_value, _ = sess.run([global_step,net.loss, net.optimize])

            duration = time.time() - start_time
            log('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            
            
            if step % config.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step
            
            if step >= hparams.num_steps:
                # error message가 나오지만, 여기서 멈춘 것은 맞다.
                raise Exception('End xxx~~~yyy')
            
    except Exception as e:
        print('finally')
        #if step > last_saved_step:
        #    save(saver, sess, logdir, step)        
        
        coord.request_stop(e)



if __name__ == '__main__':
    main()
    print('Done')
