# coding: utf-8
"""
python preprocess.py --num_workers 10 --name son --in_dir .\datasets\son --out_dir .\data\son
python preprocess.py --num_workers 10 --name moon --in_dir .\datasets\moon --out_dir .\data\moon
 ==> out_dir에  'audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'tokens', 'loss_coeff'를 묶은 npz파일이 생성된다.
 
 
 
"""
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams, hparams_debug_string
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(mod, in_dir, out_root,num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(hparams, in_dir, out_dir,num_workers=num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--in_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=str, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    args = parser.parse_args()

    if args.hparams is not None:
        hparams.parse(args.hparams)
    print(hparams_debug_string())

    name = args.name
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    num_workers = cpu_count() if num_workers is None else int(num_workers)  # cpu_count() = process 갯수

    print("Sampling frequency: {}".format(hparams.sample_rate))

    assert name in ["cmu_arctic", "ljspeech", "son", "moon"]
    mod = importlib.import_module('datasets.{}'.format(name))
    preprocess(mod, in_dir, out_dir, num_workers)
