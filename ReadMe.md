# Tocotron + Wavenet Vocoder + Korean TTS
Tacotron모델과 Wavenet Vocoder를 결합하여  한국어 TTS구현하는 project입니다.

Based on 
- https://github.com/keithito/tacotron
- https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
- https://github.com/ibab/tensorflow-wavenet
- https://github.com/r9y9/wavenet_vocoder
- https://github.com/Rayhane-mamah/Tacotron-2



## Tacotron History
- [keithito](https://github.com/keithito/tacotron)가 Tocotron을 처음 구현하여 공개하였고, 이를 기반으로 한국어를 적용한 [carpedm20](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)의 구현이 있다.
- carpedm20의 구현은 deep voice2에서 제안하고 있는 multi-speaker도 같이 구현했다.
- Tacotron모델에서는 vocoder로 Griffin Lim 알고리즘을 사용하고 있다.

## Wavenet History
- Wavenet 구현은 [ibab](https://github.com/ibab/tensorflow-wavenet)의 구현이 대표적이다.
- ibab은 local condition을 구현하지 않았다. 그래서 train 후, 소리를 생성하면 알아들을 수 있는 말이 아니고, '옹알거리는 소리'만 들을 수 있다. 의미 있는 소리를 들을 수 있기 위해서는 local condition이 적용해서 구현해야 한다.
- local condition을 구현한 wavenet-vocoder 구현은 [r9y9](https://github.com/r9y9/wavenet_vocoder)의 구현이 대표적이다.
- local condition으로 mel spectrogram을 넣어주는데, mel spectrogram은 raw audio 길이보다 짧아지기 때문에 upsampling 과정이 필요하다. upsampling은 conv2d_transpose를 이용한다.

## Tacotron 2
- [Tacotron2](https://arxiv.org/abs/1712.05884)에서는 모델 구조도 바뀌었고, Location Sensitive Attention, Stop Token, Vocoder로 Wavenet을 제안하고 있다.
- Tacotron2의 구현은 [Rayhane-mamah](https://github.com/Rayhane-mamah/Tacotron-2)의 것이 있는데, 이 역시, keithito, r9y9의 코드를 기반으로 발전된 것이다.

## This Project
* Tacotron 모델에 Wavenet Vocoder를 적용하는 것이 1차 목표이다.
* Tacotron과 Wavenet Vocoder를 같이 구현하기 위해서는 mel spectrogram을 만들때 부터, 두 모델 모두에 적용할 수 있도록 만들어 주어야 한다(audio의 길이가 hop_size의 배수가 될 수 있도록). 이렇게 해야, wavenet training할 때, upsampling이 원할하다.
* Tacotron2의 stop token이나 Location Sensitive Attention은 그렇게 효과적이지 못했다(제 경험상).
* carpedm20의 구현과 다른 점
    * Tensorflow 1.3에서만 실행되는 carpedm20의 구현을 tensorflow 1.8이상에서도 작동할 수 있게 수정.
    * dropout bug 수정 
	* DecoderPrenetWrapper, AttentionWrapper 순서를 바로 잡음. 이렇게 해야 keithito의 구현과 같아지고 논문에서의 취지와도 일치함(사실 keithito의 초기 구현에서도 잘못되어 있었는데, 나중에 수정되었음 ).
	* mel spectrogram 생성 방식을 keithito의 구현 방법으로 환원(이것도 keithito가 추후에 수정한 것임). 이렇게 mel spectrogram 생성방식을 바꾸면 train 속도가 많이 향상됨. 20k step 이상 train해야 소리가 들리기 시작하는데, 이렇게 하면 8k step부터 소리가 들린다.
	* padding이 된 곳에 Attention이 가지 않도록 보완.
	* Attention 모델 추가: 
* ibab의 wavenet구현은 [fast generation](https://github.com/tomlepaine/fast-wavenet)을 위해서 tf.Variable을 이용해서 구현했다. 이 project에서는 Tensorflow middle level api tf.conv1d를 이용하여, 코드를 이해하기 쉽게 만들었다.


	
## Tacotron에서 좋은 결과를 얻기 위해서는 
- BahdanauMonotonicAttention에 normalize=True로 적용하면 Attention이 잘 학습된다.
- Location Sensitive Attention, GMM Attention등은 제 경험으로는 성능이 잘 나지 않음.


## 단계별 실행

### 실행 순서
- data 만들기
- tacotron training 후, synthesize.py로 test.
- wavenet training 후, generate.py로 test(tactron이 만들지 않은 mel spectrogram으로 test)
- 2개 모델 모두 train 후, tacotron에서 생성한 mel spectrogram을 wavent에 local condition으로 넣어 test하면 된다.

### Data 만들기
- audio data(e.g. wave 파일)을 다운받고,  1~3초(최대 12초)길이로 잘라주는 작업을 해야 한다. 그리고  잘라진 audio와 text(script)의 sync를 맞추는 것은 고단한 작업이다.
- 특별히 data를 확보할 방법이 없으면, [carpedm20](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)에서 설명하고 있는대로 하면 된다. 여기서는 data를 다운받은 후, 침묵(silence)구간을 기준으로 자른 후, Google Speech API를 이용하여 text와 sync를 맞추고 있다.
- 한글 data는 [KSS Dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)도 있다.
- 영어 data는 [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) 등이 있다.

### Tacotron Training
```
> python train_tacotron.py
```

### Wavenet Vocoder Training

