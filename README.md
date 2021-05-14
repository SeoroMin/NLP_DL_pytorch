# KD-pytorch

* Knowledge Distillation (KD) - pytorch
* PyTorch implementation of [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
* This repository is forked from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
* Dataset: CIFAR10
* Teacher Network: VGG16
* Student Network: CNN with 3 convolutional blocks

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- See `docker/` folder.

## Pretrain Teacher Networks
* Result: 91.90%
* SGD, no weight decay.
* Learning rate adjustment
  * `0.1` for epoch `[1,150]`
  * `0.01` for epoch `[151,250]`
  * `0.001` for epoch `[251,300]`
```
python -m pretrainer --optimizer=sgd --lr=0.1   --start_epoch=1   --n_epoch=150 --model_name=ckpt
python -m pretrainer --optimizer=sgd --lr=0.01  --start_epoch=151 --n_epoch=100 --model_name=ckpt --resume
python -m pretrainer --optimizer=sgd --lr=0.001 --start_epoch=251 --n_epoch=50  --model_name=ckpt --resume
```
Teahcer Network를 통해 large model을 학습시키는 명령어로, 단순히 lr가 높다고 해서 학습이 빠르거나 하지 않음. lr=0.01에서 가장 학습 속도가 느렸고 0.001에서 가장 빨랐음.

## Student Networks
* We use Adam optimizer for fair comparison.
  * max epoch: `300`
  * learning rate: `0.0001`
  * no weight decay for fair comparison.

### EXP0. Baseline (without Knowledge Distillation)
* Result: 85.01%
```
python -m pretrainer --optimizer=adam --lr=0.0001 --start_epoch=1 --n_epoch=300 --model_name=student-scratch --network=studentnet
```
Distillation을 적용하지 않은 large model이다. 성능은 85.01%가 나왔다.

### EXP1. Effect of loss function
* Similar performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 84.99%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 84.85%
```
loss function을 바꿔주면서 student network를 통해 small model을 생성해내는 명령어다. cse(크로스 엔트로피 오차)가 mse(평균 제곱 오차)보다 성능이 좋았다.

### EXP2. Effect of Alpha
* alpha = 0.5 may show better performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 84.99%
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=cse # 85.38%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 84.85%
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=mse # 84.92%
```
soft loss의 가중치의 값을 변경하여 학습을 진행하였다.

### EXP3. Effect of Temperature Scaling
* Higher the temperature, better the performance. Consistent results with the paper.
```
python -m trainer --T=1.0  --alpha=0.5 --kd_mode=cse # 85.38%
python -m trainer --T=2.0  --alpha=0.5 --kd_mode=cse # 85.27%
python -m trainer --T=4.0  --alpha=0.5 --kd_mode=cse # 86.46%
python -m trainer --T=8.0  --alpha=0.5 --kd_mode=cse # 86.33%
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse # 86.58%
```
soft label의 T를 조정하여 스케일링 후 학습시키는 경우이다. 논문에서는 2~3이 좋다고 나왔었는데, 본 코드에서는 4~16이 성능이 좋았다.

### EXP4. More Alpha Tuning
* alpha=0.5 seems to be local optimal.
```
python -m trainer --T=16.0 --alpha=0.1 --kd_mode=cse # 85.69%
python -m trainer --T=16.0 --alpha=0.3 --kd_mode=cse # 86.48%
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse # 86.58%
python -m trainer --T=16.0 --alpha=0.7 --kd_mode=cse # 86.16%
python -m trainer --T=16.0 --alpha=0.9 --kd_mode=cse # 86.08%
```
가장 성능이 좋았던 T=16을 기준으로 soft loss의 파라미터 alpha를 좀 더 세밀하게 변경하여 학습을 진행하였다.

### EXP5. SGD Testing
```
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse --optimizer=sgd-cifar10 # 87.04%
python -m pretrainer --model_name=student-scratch-sgd-cifar10 --network=studentnet --optimizer=sgd-cifar10 # 86.34%
```

## TODO
* [ ] fix seed.
* [ ] multi gpu handling.
* [ ] split validation set.
* [ ] experiments with 5 random seed.
* [ ] remove code redundancy.
* [ ] check the optimal T is equal to calibrated T.
* [ ] Progressbar code fix in `trainer.py`.
