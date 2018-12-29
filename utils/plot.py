# coding: utf-8
import os 
import matplotlib
import matplotlib.font_manager as font_manager
from jamo import h2j, j2hcj

matplotlib.use('Agg')

# font 문제 해결
#matplotlib.rc('font', family="NanumBarunGothic")

#font_manager._rebuild()  <---- 1번만 해주면 됨

font_fname = './/utils//NanumBarunGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
matplotlib.rc('font', family="NanumBarunGothic")


import matplotlib.pyplot as plt

from text import PAD, EOS
from utils import add_postfix
from text.korean import normalize

def plot(alignment, info, text, isKorean=True):
    char_len, audio_len = alignment.shape # 145, 200

    fig, ax = plt.subplots(figsize=(char_len/5, 5))
    im = ax.imshow(
            alignment.T,
            aspect='auto',
            origin='lower',
            interpolation='none')

    xlabel = 'Encoder timestep'
    ylabel = 'Decoder timestep'

    if info is not None:
        xlabel += '\n{}'.format(info)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if text:
        if isKorean:
            jamo_text = j2hcj(h2j(normalize(text)))
        else:
            jamo_text=text
        pad = [PAD] * (char_len - len(jamo_text) - 1)
        A = [tok for tok in jamo_text] + [EOS] + pad
        A = [x if x != ' ' else '' for x in A]   # 공백이 있으면 그 뒤가 출력되지 않는 문제...
        plt.xticks(range(char_len), A)

    if text is not None:
        while True:
            if text[-1] in [EOS, PAD]:
                text = text[:-1]
            else:
                break
        plt.title(text)

    plt.tight_layout()

def plot_alignment(
        alignment, path, info=None, text=None, isKorean=True):

    if text:  # text = '대체 투입되었던 구급대원이'
        tmp_alignment = alignment[:len(h2j(text)) + 2]  # '대체 투입되었던 구급대원이' 푼 후, 길이 측정  <--- padding제거 효과

        plot(tmp_alignment, info, text, isKorean)
        plt.savefig(path, format='png')
    else:
        plot(alignment, info, text, isKorean)
        plt.savefig(path, format='png')

    print(" [*] Plot saved: {}".format(path))
    

