# coding: utf-8
import os
import time
import pprint
import random
import threading
import traceback
import numpy as np
from glob import glob
import tensorflow as tf
from collections import defaultdict

import text
from utils.infolog import log
from utils import parallel_run, remove_file
from utils.audio import frames_to_hours



_pad = 0

def get_frame(path):
    data = np.load(path)
    n_frame = data["linear"].shape[0]
    n_token = len(data["tokens"])
    return (path, n_frame, n_token)

def get_path_dict(data_dirs, hparams, config,data_type, n_test=None,rng=np.random.RandomState(123)):

    # Load metadata:
    path_dict = {}
    for data_dir in data_dirs:  # ['datasets/moon\\data']
        paths = glob("{}/*.npz".format(data_dir)) # ['datasets/moon\\data\\001.0000.npz', 'datasets/moon\\data\\001.0001.npz', 'datasets/moon\\data\\001.0002.npz', ...]

        if data_type == 'train':
            rng.shuffle(paths)  # ['datasets/moon\\data\\012.0287.npz', 'datasets/moon\\data\\004.0215.npz', 'datasets/moon\\data\\003.0149.npz', ...]

        if not config.skip_path_filter:
            items = parallel_run( get_frame, paths, desc="filter_by_min_max_frame_batch", parallel=True)  # [('datasets/moon\\data\\012.0287.npz', 130, 21), ('datasets/moon\\data\\003.0149.npz', 209, 37), ...]

            min_n_frame = hparams.reduction_factor * hparams.min_iters   # 5*30
            max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor  # 5*200 - 5
            
            # 다음 단계에서 data가 많이 떨어져 나감. 글자수가 짧은 것들이 탈락됨.
            new_items = [(path, n) for path, n, n_tokens in items if min_n_frame <= n <= max_n_frame and n_tokens >= hparams.min_tokens] # [('datasets/moon\\data\\004.0383.npz', 297), ('datasets/moon\\data\\003.0533.npz', 394),...]

            if any(check in data_dir for check in ["son", "yuinna"]):
                blacklists = [".0000.", ".0001.", "NB11479580.0001"]
                new_items = [item for item in new_items if any(check not in item[0] for check in blacklists)]

            new_paths = [path for path, n in new_items]
            new_n_frames = [n for path, n in new_items]

            hours = frames_to_hours(new_n_frames,hparams)

            log(' [{}] Loaded metadata for {} examples ({:.2f} hours)'.format(data_dir, len(new_n_frames), hours))
            log(' [{}] Max length: {}'.format(data_dir, max(new_n_frames)))
            log(' [{}] Min length: {}'.format(data_dir, min(new_n_frames)))
        else:
            new_paths = paths

        if data_type == 'train':
            new_paths = new_paths[:-n_test]
        elif data_type == 'test':
            new_paths = new_paths[-n_test:]
        else:
            raise Exception(" [!] Unkown data_type: {}".format(data_type))

        path_dict[data_dir] = new_paths  # ['datasets/moon\\data\\001.0621.npz', 'datasets/moon\\data\\003.0229.npz', ...]

    return path_dict


# run -> _enqueue_next_group -> _get_next_example
class DataFeederTacotron(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, data_dirs,hparams, config, batches_per_group, data_type, batch_size):  #batches_per_group = 32 or 8,  data_type: 'train' or 'test'
        super(DataFeederTacotron, self).__init__()

        self._coord = coordinator
        self._hp = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._step = 0
        self._offset = defaultdict(lambda: 2)
        self._batches_per_group = batches_per_group

        self.rng = np.random.RandomState(config.random_seed)   # random number generator
        self.data_type = data_type
        self.batch_size = batch_size

        self.min_tokens = hparams.min_tokens  # 30
        self.min_n_frame = hparams.reduction_factor * hparams.min_iters   # 5*30
        self.max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor  # 5*200 - 5
        self.skip_path_filter = config.skip_path_filter

        # Load metadata:
        self.path_dict = get_path_dict(data_dirs, self._hp, config, self.data_type,n_test=self.batch_size, rng=self.rng) # data_dirs: ['datasets/moon\\data']

        self.data_dirs = list(self.path_dict.keys()) # ['datasets/moon\\data']
        self.data_dir_to_id = {data_dir: idx for idx, data_dir in enumerate(self.data_dirs)}  # {'datasets/moon\\data': 0}

        data_weight = {data_dir: 1. for data_dir in self.data_dirs} # {'datasets/moon\\data': 1.0}

        if self._hp.main_data_greedy_factor > 0 and any(main_data in data_dir for data_dir in self.data_dirs for main_data in self._hp.main_data):   # 'main_data': ['']
            for main_data in self._hp.main_data:
                for data_dir in self.data_dirs:
                    if main_data in data_dir:
                        data_weight[data_dir] += self._hp.main_data_greedy_factor

        weight_Z = sum(data_weight.values())  # 1
        self.data_ratio = { data_dir: weight / weight_Z for data_dir, weight in data_weight.items()}  # 각 data들의 weight sum이 1이 되도록...

        log("="*40)
        log(pprint.pformat(self.data_ratio, indent=4))
        log("="*40)

        #audio_paths = [path.replace("/data/", "/audio/").replace(".npz", ".wav") for path in self.data_paths]
        #duration = get_durations(audio_paths, print_detail=False)

        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.

        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, [None], 'loss_coeff'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
        ]

        # Create queue for buffering data:
        dtypes = [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32]

        self.is_multi_speaker = len(self.data_dirs) > 1

        if self.is_multi_speaker:
            self._placeholders.append( tf.placeholder(tf.int32, [None], 'speaker_id'),)     # speaker_id 추가  'inputs'  --> 'speaker_id'로 바꿔야 하지 않나??
            dtypes.append(tf.int32)

        num_worker = 8 if self.data_type == 'train' else 1
        queue = tf.FIFOQueue(num_worker, dtypes, name='input_queue')

        self._enqueue_op = queue.enqueue(self._placeholders)

        if self.is_multi_speaker:
            self.inputs, self.input_lengths, self.loss_coeff, self.mel_targets, self.linear_targets, self.speaker_id = queue.dequeue()
        else:
            self.inputs, self.input_lengths, self.loss_coeff, self.mel_targets, self.linear_targets = queue.dequeue()

        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.loss_coeff.set_shape(self._placeholders[2].shape)
        self.mel_targets.set_shape(self._placeholders[3].shape)
        self.linear_targets.set_shape(self._placeholders[4].shape)

        if self.is_multi_speaker:
            self.speaker_id.set_shape(self._placeholders[5].shape)
        else:
            self.speaker_id = None

        if self.data_type == 'test':
            examples = []
            while True:
                for data_dir in self.data_dirs:
                    examples.append(self._get_next_example(data_dir))
                    #print(data_dir, text.sequence_to_text(examples[-1][0], False, True))
                    if len(examples) >= self.batch_size:
                        break
                if len(examples) >= self.batch_size:
                    break
            
            # test 할 때는 같은 examples로 계속 반복
            self.static_batches = [examples for _ in range(self._batches_per_group)]  # [examples, examples,...,examples] <--- 각 example은 2개의 data를 가지고 있다.

        else:
            self.static_batches = None

    def start_in_session(self, session, start_step):
        self._step = start_step
        self._session = session
        self.start()


    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)


    def _enqueue_next_group(self):
        start = time.time()

        # Read a group of examples:
        n = self.batch_size   # 32
        r = self._hp.reduction_factor  #  4 or 5  min_n_frame,max_n_frame 계산에 사용되었던...

        if self.static_batches is not None:  # 'test'에서는 static_batches를 사용한다. static_batches는 init에서 이미 만들어 놓았다.
            batches = self.static_batches
        else: # 'train'
            examples = []
            for data_dir in self.data_dirs:
                if self._hp.initial_data_greedy:
                    if self._step < self._hp.initial_phase_step and any("krbook" in data_dir for data_dir in self.data_dirs):
                        data_dir = [data_dir for data_dir in self.data_dirs if "krbook" in data_dir][0]

                if self._step < self._hp.initial_phase_step:  # 'initial_phase_step': 8000
                    example = [self._get_next_example(data_dir) for _ in range(int(n * self._batches_per_group // len(self.data_dirs)))]  # _batches_per_group 8,또는 32 만큼의 batch data를 만. 각각의 batch size는 2, 또는 32
                else:
                    example = [self._get_next_example(data_dir) for _ in range(int(n * self._batches_per_group * self.data_ratio[data_dir]))]
                examples.extend(example)
            examples.sort(key=lambda x: x[-1])  # 제일 마지막 기준이니까,  len(linear_target) 기준으로 정렬

            batches = [examples[i:i+n] for i in range(0, len(examples), n)]
            self.rng.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:  # batches는 batch의 묶음이다.
            # test 또는 train mode에 맞게 만든 batches의  batch data를 placeholder에 넘겨준다.
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r, self.rng, self.data_type)))   # _prepare_batch에서 batch data의 길이를 맞춘다. return 순서 = placeholder순서
            self._session.run(self._enqueue_op, feed_dict=feed_dict)
            self._step += 1


    def _get_next_example(self, data_dir):
        '''npz 1개를 읽어 처리한다. Loads a single example (input, mel_target, linear_target, cost) from disk'''
        data_paths = self.path_dict[data_dir]

        while True:
            if self._offset[data_dir] >= len(data_paths):
                self._offset[data_dir] = 0

                if self.data_type == 'train':
                    self.rng.shuffle(data_paths)

            data_path = data_paths[self._offset[data_dir]]  # npz파일 1개 선택
            self._offset[data_dir] += 1

            try:
                if os.path.exists(data_path):
                    data = np.load(data_path)  # data속에는 "linear","mel","tokens","loss_coeff"
                else:
                    continue
            except:
                remove_file(data_path)
                continue

            if not self.skip_path_filter:
                break

            if self.min_n_frame <= data["linear"].shape[0] <= self.max_n_frame and  len(data["tokens"]) > self.min_tokens:
                break

        input_data = data['tokens']   # 1-dim
        mel_target = data['mel']

        if 'loss_coeff' in data:
            loss_coeff = data['loss_coeff']
        else:
            loss_coeff = 1
        linear_target = data['linear']

        return (input_data, loss_coeff, mel_target, linear_target, self.data_dir_to_id[data_dir], len(linear_target))


def _prepare_batch(batch, reduction_factor, rng, data_type=None):
    if data_type == 'train':
        rng.shuffle(batch)

    # batch data: (input_data, loss_coeff, mel_target, linear_target, self.data_dir_to_id[data_dir], len(linear_target))
    inputs = _prepare_inputs([x[0] for x in batch])  # batch에 있는 data들 중, 가장 긴 data의 길이에 맞게 padding한다.
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)  # batch_size, [37, 37, 32, 32, 38,..., 39, 36, 30]
    loss_coeff = np.asarray([x[1] for x in batch], dtype=np.float32)   # batch_size, [1,1,1,,..., 1,1,1]

    mel_targets = _prepare_targets([x[2] for x in batch], reduction_factor)  # ---> (32, 175, 80) max length는 reduction_factor의  배수가 되도록
    linear_targets = _prepare_targets([x[3] for x in batch], reduction_factor)  # ---> (32, 175, 1025)  max length는 reduction_factor의  배수가 되도록

    if len(batch[0]) == 6:  # is_multi_speaker = True인 경우
        speaker_id = np.asarray([x[4] for x in batch], dtype=np.int32)   # speaker_id로 list 만들기
        return (inputs, input_lengths, loss_coeff,mel_targets, linear_targets, speaker_id)
    else:
        return (inputs, input_lengths, loss_coeff, mel_targets, linear_targets)  # ('inputs' 'input_lengths' 'loss_coeff' 'mel_targets' 'linear_targets')


def _prepare_inputs(inputs):  # inputs: batch 길이 만큼의 list
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])  # (batch_size, max_len)
    """
    batch_size = 2 일 떼,
    [[13, 26, 13, 41, 13, 21, 13, 41, 13, 21, 13, 41,  9, 41, 13, 40,79, 14, 34, 13, 33, 79, 20, 32, 13, 35, 45,  2, 34, 42, 13, 39,7, 29, 11, 25,  1],
    [ 6, 29, 79, 14, 26, 14, 34,  5, 29, 79,  2, 30, 45,  2, 28, 14,21, 79, 13, 27,  7, 25,  9, 34, 45, 13, 40, 79,  4, 29,  2, 29,13, 26,  1,  0,  0]]    
    """

def _prepare_targets(targets, alignment):
    # targets: shape of list [ (162,80) , (172, 80), ...] 
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    # t: 2 dim array. ( xx, num_mels) ==> (length,num_mels)
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)  # (169, 80) ==> (length, 80)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
