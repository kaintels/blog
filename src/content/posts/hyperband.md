---
title: Hyperband 리뷰
published: 2024-09-11
description: "Hyperband: A novel bandit-based approach to hyperparameter optimization"
tags: [Research]
category: Review
draft: false
---

## 원본 논문

[Hyperband: A novel bandit-based approach to hyperparameter optimization]:
https://scholar.google.com/scholar_url?url=https://www.jmlr.org/papers/volume18/16-558/16-558.pdf&hl=ko&sa=T&oi=gsb-gga&ct=res&cd=0&d=13404637959739378207&ei=BrLgZolH9ILqtA_Tk_6YCg&scisig=AFWwaeZWglwIHNU7vkIkHMm0SM1j
[Hyperband: A novel bandit-based approach to hyperparameter optimization]

Journal of Machine Learning Research, 2018

## 요약
>
> 이 논문은 **하이퍼밴드(Hyperband)** 라는 새로운 알고리즘을 소개하며, 이를 통해 하이퍼파라미터 최적화(hyperparameter optimization) 문제를 해결하는 방법을 제시함. 하이퍼파라미터 최적화는 머신 러닝 모델의 성능을 크게 좌우하는 중요한 요소임. 하이퍼밴드는 하이퍼파라미터 최적화 문제를 **무한 팔 밴딧 문제(infinite-armed bandit problem)** 로 정의하며, 각 하이퍼파라미터 구성을 실험할 때 컴퓨팅 자원을 어떻게 효율적으로 분배할지를 결정함. 하이퍼밴드는 다양한 하이퍼파라미터 설정을 빠르게 평가할 수 있도록 하며, 성능이 낮은 설정은 조기 종료하여 더 유망한 설정에 자원을 집중하도록 함.
이 논문은 하이퍼밴드를 소개하면서, 다양한 딥러닝 및 커널 기반 학습 문제에서 기존 베이지안 최적화 방법들과의 성능을 비교한 결과, 하이퍼밴드가 속도 면에서 월등히 빠르다는 것을 보여줌. 특히, 하이퍼밴드는 베이지안 최적화보다 5배에서 30배까지 빠른 속도를 보이며, 여러 딥러닝 및 커널 학습 문제에서 우수한 성능을 나타냄.

## 1. 도입부

최근 몇 년 동안 머신러닝 모델의 복잡성과 표현력은 폭발적으로 증가했지만, 그 대가로 높은 계산 비용이 발생. 그리고 고려할만한 튜닝 파라미터의 수가 증가하면서 표준 최적화 기법으로는 설정하기 매우 어려움. 이러한 '하이퍼파라미터(hyperparameter)'는 머신러닝 알고리즘에 입력되어 알고리즘의 성능이 보이지 않는 새로운 데이터에 일반화되는 방식을 결정하는 입력값으로, 예시로는 모델 아키텍처, 정규화의 양 등이 있음. 예측 모델의 성능은 하이퍼파라미터 구성에 따라 크게 좌우되지만, 이러한 하이퍼파라미터가 서로 어떻게 상호작용하여 결과 모델에 영향을 미치는지에 대해서는 잘 알려져 있지 않음

따라서 실무자들은 랜덤 서치(Random search)나 그리드 서치(Grid search) 같은 무차별 대입 방법을 기본으로 사용하는 경우가 많음. 또한 보다 효율적인 검색 방법을 개발하기 위해 최근 하이퍼파라미터 최적화 문제는 하이퍼파라미터 구성 선택 최적화에 초점을 맞춘 베이지안 최적화(Bayesian Opitmization) 방법이 주로 사용됨.  이러한 방법은 적응형 방식으로 구성을 선택함으로써 랜덤 서치와 같은 기준보다 더 빠르게 좋은 구성을 식별하는 것을 목표로 수행되어 옴. 이 방법은 무작위 검색보다 성능이 우수하다고 알려짐. **그러나 이러한 방법은 고차원의 함수를 동시에 적합하고 최적화하는 근본적으로 까다로운 문제를 해결해야 하고, 평가가 노이즈가 있을 수 있어 해결방안이 필요함.**

본 논문은 하이퍼파라미터 최적화를 순수 탐색 적응형 리소스 할당 문제로 풀어 무작위로 샘플링된 하이퍼파라미터 구성 중에서 리소스를 할당하는 방법을 다루는 새로운 구성 평가 접근법을 개발

## 2. 상세 설명

### Bandit Problems

Bandit Problems(밴딧 문제)은 주어진 환경에서 최적 해로부터의 거리로 정의되는 단순 후회(simple regret)를 가능한 한 빨리 최소화하는 것을 목표로 함

> 후회라는 것은 우리가 선택한 옵션과 최적의 옵션 사이의 성능 차이
> 단순 후회는 후회 중에서도 최적의 선택을 빠르게 찾아내는 것에 중점을 둔 지표
> 즉, 주어진 시간 내에 여러 옵션을 실험했을 때, **최적의 옵션과 현재까지 탐색한 옵션들 중 가장 좋은 옵션 간의 차이(성능의 차이)**
> 예를 들어, 탐색 결과 중 가장 좋은 구성의 검증 오차가 0.2이고, 최적의 구성의 검증 오차가 0.15라면, 이 두 구성 간의 성능 차이가 simple regret!

#### 밴딧 문제의 주요 요소

팔(Arm): 각각의 팔은 하이퍼파라미터 구성 하나를 의미. 논문에서는 이 팔이 무한히 많을 수 있다고 가정하며, 이를 통해 매우 큰 하이퍼파라미터 공간에서의 최적화 문제를 다루게 됨

보상(Reward) 또는 손실(Loss): 각 팔을 당길 때마다 얻는 결과로, 하이퍼파라미터 구성의 성능(예를 들어, 검증 오차(validation error)같은 것)임. 초기에는 이 성능을 알 수 없으므로 당연히 자원 할당을 통해 평가를 시작해야 함

자원 할당(Resource Allocation): 자원은 하이퍼파라미터 구성을 평가하는 데 사용되는 컴퓨팅 자원(예: 훈련 데이터 양, 반복 횟수 등)을 의미함. 하이퍼밴드는 성능이 나쁜 구성에 자원을 적게 할당하고, 유망한 구성에는 더 많은 자원을 할당하여 최적의 구성을 찾는 방식으로 작동함.

> 논문에서는 Pure-exploration Non-stochastic Infinite-armed Bandit을 따로 언급함

#### 논문에서 정의한 Pure-exploration Non-stochastic Infinite-armed Bandit 문제의 개념

순수 탐색(Pure Exploration): 이 문제는 **여러 옵션(팔) 중에서 최적의 옵션을 찾는 것이 목표임.** 따라서 논문의 목적은 최종적으로 최고의 하이퍼파라미터 구성을 찾는 것, 이를 위해 자원을 효율적으로 배분하여 각 구성을 탐색

비확률적(Non-stochastic): 이 문제는 비확률적이라고 설명되는데, 이유는 각 팔의 결과가 확률 분포에 따라 결정되는 것이 아니라, 정해진 성능(손실 함수, 정확도 등)이 있다고 가정하기 때문

### Hyperband

알고리즘은 아래 그림과 같음 (출처 : 원 논문)
![](https://velog.velcdn.com/images/kaintels/post/52f65a5e-2e92-4941-8162-8f5fdc12dd8d/image.png)

$$R$$은 설정하고자하는 전체 리소스, $$η$$는(eta, 에타라고 읽음) 각 라운드마다 남길 하이퍼파라미터 설정의 비율을 결정하는 값으로, Successive Halving에서 사용함.
예를 들어, $$η$$ = 3이면, 각 라운드마다 남겨지는 설정의 수는 1/3이 됨
$$η$$ 값이 클수록 더 많은 설정이 한 번에 제거되며, 이는 더 빠른 탐색을 가능하게 하지만 일부 최적의 설정이 초기 단계에서 제거될 위험도 있으므로 적절히 사용하는 것이 중요함

$$s_{max}$$는 Hyperband가 사용할 수 있는 최대 브래킷($$B$$)의 수를 결정하는데, 이는 Hyperband에서 최적의 하이퍼파라미터를 찾기 위해 여러 번의 Successive Halving을 수행할 수 있도록 함
여기서 $$s_{max}$$는 $$⌊log_η(R)⌋$$ 으로, 이 값이 클수록 더 많은 브래킷이 생성되어 탐색 공간이 넓어지고, 다양한 하이퍼파라미터 설정이 평가될 수 있음

이하 알고리즘 내용 중 Successive Halving에 대해 설명

#### Successive Halving

Successive Halving(연속 절반 감소) 알고리즘은 하이퍼파라미터 최적화나 모델 선택 문제에서 사용되는 효율적인 자원 할당 방법
이 알고리즘은 성능이 나쁜 후보들을 빠르게 제거하고, 성능이 좋은 후보에게 더 많은 자원을 할당하여 최적의 하이퍼파라미터나 모델을 찾는 방법임.

- 간단히 말해서, 조기 종료 기법을 사용하여 탐색 공간을 효율적으로 줄여 나가는 방식

#### 구현 방법

1. 초기 설정 및 자원 할당:

> n개의 하이퍼파라미터 설정 또는 모델 구성들을 초기 후보군으로 설정함
> 각 설정마다 일정한 양의 자원(예: 훈련 데이터 양, 반복(iteration) 횟수 등)을 할당하여 평가. 여기서는 이 자원을 **예산(Budget)** 이라고 표현

1. 성능 평가:

> 주어진 자원을 사용하여 각 후보군을 일정 시간 동안 학습하거나 평가함.
> 평가 결과를 바탕으로 성능을 측정하고, 각 설정에 대한 손실(loss) 등 계산

3. 절반 제거:

> 평가가 끝난 후, 성능이 낮은 절반의 후보들을 제거함
> 즉, 상위 50%의 성능을 보인 후보들만 다음 단계로 넘어가게 됨

4. 자원 재분배:

남은 상위 후보들에게 더 많은 자원을 할당함. 각 단계에서 생존한 후보들은 점점 더 많은 자원을 받게 되며, 최종적으로 하나의 최적의 구성만 남게 됨

5. 반복:

이 과정을 반복하면서, 각 단계마다 후보의 수는 절반으로 줄어들고, 남은 후보에게 할당되는 자원은 점점 더 늘어남

#### 예시

(100개의 파라미터 조합을 1번만 학습 -> 성능이 좋은 50개만 살려냄 -> 50개의 파라미터 조합을 2번만 학습 -> 성능이 좋은 25개만 살려냄 -> 25개의 파라미터 조합을 3번 학습 -> 13개만 살려냄 -> 13개의 파라미터를 4번 학습 -> (중략...) -> 최종 하나의 파라미터만 남음)

## 3. 실험 결과

하이퍼밴드와 성능 비교하는 모델은 3개의 Bayesian Optimization 방법을 사용함 (SMAC, TPE, Spearmint)
Baseline 모델: random search, random search 2X. (budget을 2배만큼 사용)

Budget 조건을 바꿔가며 3가지 다른 실험을 진행함
(학습 iterations, 학습 dataset 크기, feature subsample)

### 학습 iterations인 경우

CNN 모델 최적화

![](https://velog.velcdn.com/images/kaintels/post/039f87a5-951d-410a-bfea-1b1a82ffcfd1/image.png)

Random search보다 20배 빠름.

### 학습 dataset 크기인 경우

kernel-based(SVM) 모델 최적화

Bayesian Optimization보다 30배 빠르고, Random search보다 70배 빠름 (아래 그림 왼쪽)

### feature subsample

무작위로 특징을 생성하여 RBF 커널을 근사화하고, 그 후 무작위 특징을 ridge 회귀 분류기의 입력으로 사용한 경우를 최적화

베이지안 방법 및 무작위 검색보다 약 6배 빠름 (아래 그림 오른쪽)

![](https://velog.velcdn.com/images/kaintels/post/607fa93b-390c-455c-b4b6-ebe8845e985e/image.png)

## 4. 결론

논문의 결론에서는 하이퍼밴드(Hyperband) 알고리즘의 강점을 요약하고, 향후 확장 가능성을 제시함. 결론에서 논의된 주요 내용은 다음과 같음

1. **병렬화 가능성**: 하이퍼밴드는 각 후보군(Arm)이 독립적이며 무작위로 샘플링되기 때문에 병렬화가 가능하다는 점을 강조함. 특히, Successive Halving의 예산을 여러 머신에 분배하여 비동기적으로 실행할 수 있음

2. **예산 배분 조정 중요성**: 하이퍼밴드는 구성에 따라 자원을 할당하여 공정한 비교가 이루어질 수 있도록 예산 배분을 조정하는 것이 중요하다고 언급함

3. **비랜덤 샘플링 방법 통합**: 하이퍼밴드는 랜덤 탐색 외에도 quasi-random 방법(예: Sobol or latin hypercube)과 메타러닝 기반의 지능형 사전학습을 결합함으로써 성능을 향상시킬 수 있는 가능성을 언급함.

## 사용 방법

[optuna](https://optuna.org/) 라이브러리에 잘 설명되어 있음

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import optuna

X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
classes = np.unique(y)
n_train_iter = 100


def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    clf = SGDClassifier(alpha=alpha)

    for step in range(n_train_iter):
        clf.partial_fit(X_train, y_train, classes=classes)

        intermediate_value = clf.score(X_valid, y_valid)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return clf.score(X_valid, y_valid)


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=n_train_iter, reduction_factor=3
    ), # 하이퍼밴드, eta=3
)
study.optimize(objective, n_trials=20)
```
