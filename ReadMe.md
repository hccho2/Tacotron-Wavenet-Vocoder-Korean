# Tocotron + Wavenet Vocoder + Korean TTS


Based on 
- https://github.com/keithito/tacotron
- https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
- https://github.com/ibab/tensorflow-wavenet
- https://github.com/r9y9/wavenet_vocoder
- https://github.com/Rayhane-mamah/Tacotron-2



## Tocotron History
- [kiithito](https://github.com/keithito/tacotron)이 Tocotron을 처음 구현하여 공개하였고, 이를 기반으로 한국어를 적용한 [carpedm20](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)의 구현이 있다.
- carpedm20의 구현은 deep voice2에서 제안하고 있는 multi-speaker도 같이 구현했다.
- Tacotron모델에서는 vocoder로 Griffin Lim 알고리즘을 사용하고 있다.

## Wavenet History
- Wavenet 구현은 [ibab](https://github.com/ibab/tensorflow-wavenet)의 구현이 대표적이다.
- ibab의 구현은 local condition을 구현하지 않았다. 그래서 train 후, 소리를 생성하면 알아들을 수 있는 말이 아니고, '옹알거리는 소리'만 들을 수 있다.
- local condition을 구현한 wavenet-vocoder 구현은 [r9y9](https://github.com/r9y9/wavenet_vocoder)의 구현이 대표적이다.

## Tacotron 2
- [Tacotron2](https://arxiv.org/abs/1712.05884)에서는 모델 구조도 바뀌었고, Location Sensitive Attention, Stop Token, Vocoder로 Wavenet을 제안하고 있다.

## This Project
- Tacotron 모델에 Wavenet Vocoder를 적용하는 것이 1차 목표이다.
- Tacotron2의 stop token이나 Location Sensitive Attention은 그렇게 효과적이지 못했다(제 경험상).
- carpedm20의 구현과 다른 점
 - Tensorflow 1.3에서만 실행되는 carpedm20의 구현을 tensorflow 1.8이상에서도 작동할 수 있게 수정.
 - dropout bug 수정 
