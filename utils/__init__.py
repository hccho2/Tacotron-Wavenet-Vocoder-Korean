# -*- coding: utf-8 -*-
import re,json,sys,os
import tensorflow as tf
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file


PARAMS_NAME = "params.json"
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR_ROOT_Wavenet = './logdir-wavenet'


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
def prepare_dirs(config, hparams):
    if hasattr(config, "data_paths"):
        config.datasets = [os.path.basename(data_path) for data_path in config.data_paths]
        dataset_desc = "+".join(config.datasets)

    if config.load_path:
        config.model_dir = config.load_path
    else:
        config.model_name = "{}_{}".format(dataset_desc, get_time())
        config.model_dir = os.path.join(config.log_dir, config.model_name)

        for path in [config.log_dir, config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    if config.load_path:
        load_hparams(hparams, config.model_dir)
    else:
        setattr(hparams, "num_speakers", len(config.datasets))

        save_hparams(config.model_dir, hparams)
        copy_file("hparams.py", os.path.join(config.model_dir, "hparams.py"))

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    #ckpt = get_most_recent_checkpoint(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    if not os.path.exists(logdir):
        os.makedirs(logdir)    
    return logdir


def validate_directories(args,hparams):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT_Wavenet
        
        
    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))
        save_hparams(logdir, hparams)
        copy_file("hparams.py", os.path.join(logdir, "hparams.py"))
    else:
        load_hparams(hparams, logdir)
        
    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }
def save_hparams(model_dir, hparams):
    param_path = os.path.join(model_dir, PARAMS_NAME)

    info = eval(hparams.to_json(),{'false': False, 'true': True, 'null': None})
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))
    
def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)

def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  #json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx',]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)
def load_json(path, as_class=False, encoding='euc-kr'):
    with open(path,encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)

    return data    
def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def add_prefix(path, prefix):
    dir_path, filename = os.path.dirname(path), os.path.basename(path)
    return "{}/{}.{}".format(dir_path, prefix, filename)

def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)

def remove_postfix(path):
    items = path.rsplit('.', 2)
    return items[0] + "." + items[2]

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool()) as pool:
            for out in tqdm(pool.imap_unordered(fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results
def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)
        
def str2bool(v):
    return v.lower() in ('true', '1')

def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
def get_git_diff():
    return subprocess.check_output(['git', 'diff']).decode("utf-8")

def warning(msg):
    print("="*40)
    print(" [!] {}".format(msg))
    print("="*40)
    print()		
		