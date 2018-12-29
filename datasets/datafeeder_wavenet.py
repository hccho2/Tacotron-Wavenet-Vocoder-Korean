# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import tensorflow as tf
import threading
import random
import numpy as np
import os
from utils import audio
from hparams import hparams
from glob import glob
from collections import defaultdict


def get_path_dict(data_dirs, min_length):
    path_dict = {}
    for data_dir in data_dirs:
        
        if not hparams.skip_path_filter:
        
            with open(os.path.join(data_dir,'train.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_paths = []
                for line in lines:
                    line = line.strip().split("|")
                    if int(line[3]) > min_length:
                        new_paths.append(line[6])
            
            path_dict[data_dir] = new_paths
        else:
            new_paths = glob("{}/*.npz".format(data_dir))
            
            new_paths = [os.path.basename(p) for p in new_paths]
            path_dict[data_dir] = new_paths
    return path_dict

def assert_ready_for_upsampling(x, c,hop_size):
    assert len(x) % len(c) == 0 and len(x) // len(c) == hop_size

def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


class DataFeederWavenet(threading.Thread):
    def __init__(self,coord,data_dirs,batch_size,receptive_field, gc_enable=False, queue_size=8):
        super(DataFeederWavenet, self).__init__()    
        self.data_dirs = data_dirs
        self.coord = coord
        self.batch_size = batch_size
        self.receptive_field = receptive_field
        self.hop_size = audio.get_hop_size(hparams)
        self.sample_size = ensure_divisible(hparams.sample_size,self.hop_size, True)
        self.max_frames = self.sample_size // self.hop_size  # sample_size 크기를 확보하기 위해.
        self.queue_size = queue_size
        self.gc_enable = gc_enable
        self.skip_path_filter = hparams.skip_path_filter
       
        self.rng = np.random.RandomState(123)
        self._offset = defaultdict(lambda: 2)  # key에 없는 값이 들어어면 2가 할당된다.
        
        self.data_dir_to_id = {data_dir: idx for idx, data_dir in enumerate(self.data_dirs)}  # data_dir <---> speaker_id 매핑
        self.path_dict = get_path_dict(self.data_dirs,np.max([self.sample_size,receptive_field]))# receptive_field 보다 작은 것을 버리고, 나머지만 돌려준다.
        
        self._placeholders = [
            tf.placeholder(tf.float32, shape=[None,None,1],name='input_wav'),
            tf.placeholder(tf.float32, shape=[None,None,hparams.num_mels],name='local_condition')
        ]    
        dtypes = [tf.float32, tf.float32]
    
        if self.gc_enable:
            self._placeholders.append(tf.placeholder(tf.int32, shape=[None],name='speaker_id'))
            dtypes.append(tf.int32)
 
        queue = tf.FIFOQueue(self.queue_size, dtypes, name='input_queue')
        self.enqueue = queue.enqueue(self._placeholders)
        
        if self.gc_enable:
            self.inputs_wav, self.local_condition, self.speaker_id = queue.dequeue()
        else:
            self.inputs_wav, self.local_condition = queue.dequeue()

        self.inputs_wav.set_shape(self._placeholders[0].shape)
        self.local_condition.set_shape(self._placeholders[1].shape)
        if self.gc_enable:
            self.speaker_id.set_shape(self._placeholders[2].shape)
   
            
    def run(self):
        try:
            while not self.coord.should_stop():
                self.make_batches()
        except Exception as e:
            self.coord.request_stop(e)       
    def start_in_session(self, session,start_step):
        self._step = start_step
        self.sess = session
        self.start()
              
    def make_batches(self):
        examples = []
        n = self.batch_size
        for data_dir in self.data_dirs:
            example = [self._get_next_example(data_dir) for _ in range(int(n * 32 // len(self.data_dirs)))]
            examples.extend(example)
        self.rng.shuffle(examples)
        batches = [examples[i:i+n] for i in range(0, len(examples), n)]
        
        
        for batch in batches: # batch size만큼의 data를 원하는 만큼 만든다.
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch))) 
            self.sess.run(self.enqueue, feed_dict=feed_dict)
            self._step += 1
    
    def _get_next_example(self, data_dir):
        '''npz 1개를 읽어 처리한다. Loads a single example (input_wav, local_condition,speaker_id ) from disk'''
        data_paths = self.path_dict[data_dir]
        
        while True:
            if self._offset[data_dir] >= len(data_paths):
                self._offset[data_dir] = 0
                self.rng.shuffle(data_paths)
            
            data_path = os.path.join(data_dir,data_paths[self._offset[data_dir]])  # npz파일 1개 선택
            self._offset[data_dir] += 1
            
            if os.path.exists(data_path):
                data = np.load(data_path)  # data속에는 'audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'token'
            else:
                continue       
            
            if not self.skip_path_filter:
                # 이경우는 get_path_dict함수에서 한번 걸러졌기 때문에, 여기서 다시 확인할 필요 없음.
                break
            
            # get_path_dict함수에서 걸러지지 않앗기 때문에 확인이 필요함.
            if data['time_steps'] > self.sample_size:
                break
                 

        input_wav = data['audio']
        local_condition = data['mel']
        input_wav = input_wav.reshape(-1, 1)
        assert_ready_for_upsampling(input_wav, local_condition,self.hop_size)
        
        
        
        s = np.random.randint(0, len(local_condition) - self.max_frames+1)  # hccho
        ts = s * self.hop_size
        input_wav = input_wav[ts:ts + self.hop_size * self.max_frames, :]
        local_condition = local_condition[s:s + self.max_frames, :]        
        if self.gc_enable:
            return (input_wav,local_condition, self.data_dir_to_id[data_dir])
        else: return (input_wav,local_condition)
def _prepare_batch(batch):
    input_wavs = [x[0] for x in batch]
    local_conditions = [x[1] for x in batch]
    if len(batch[0])==3:
        speaker_ids = [x[2] for x in batch]
        return (input_wavs,local_conditions,speaker_ids)
    else:
        return (input_wavs,local_conditions)
        
        
if __name__ == '__main__':
    coord = tf.train.Coordinator()
    data_dirs=['D:\\hccho\\Tacotron-Wavenet-Vocoder-hccho\\data\\moon','D:\\hccho\\Tacotron-Wavenet-Vocoder-hccho\\data\\son']
    mydatafeed =  DataFeederWavenet(coord,data_dirs,batch_size=5,receptive_field=1200, gc_enable=True, queue_size=8)
    
    
    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())
            step = 0
            mydatafeed.start_in_session(sess,step) 
            
            while not coord.should_stop():
                a,b,c=sess.run([mydatafeed.inputs_wav, mydatafeed.local_condition, mydatafeed.speaker_id])
                
                print(a.shape,b.shape,c.shape)
                print(step, c)
                
                a,b,c=sess.run([mydatafeed.inputs_wav, mydatafeed.local_condition, mydatafeed.speaker_id])
                
                print(a.shape,b.shape,c.shape)
                print(step, c)               
                
                
                step =  step +1
                
        
        except Exception as e:
            print('finally')
            coord.request_stop(e)
    