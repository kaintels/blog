---
title: QBSO-FS 리뷰
published: 2024-09-21
description: "QBSO-FS: A reinforcement learning based bee swarm optimization metaheuristic for feature selection"
tags: [Research]
category: Review
draft: false
---

## 원본 논문

[QBSO-FS: A Reinforcement Learning Based Bee Swarm Optimization Metaheuristic for Feature Selection](https://link.springer.com/chapter/10.1007/978-3-030-20518-8_65)

Advances in Computational Intelligence: 15th International Work-Conference on Artificial Neural Networks, IWANN 2019

논문을 결제해야 볼 수 있기 때문에, 모든 내용을 담으면 불법이므로 초록을 기준으로 핵심만 설명

## 요약

> 본 논문은 강화 학습인 큐러닝(Q-learning)과 꿀벌 군집 최적화(Bee Swarm Optimization, BSO)을 통합하여 특징 선택 문제를 해결하는 알고리즘 제안

## 1. 상세 설명

### 특징 선택

> 머신러닝 문제(분류, 회귀 등)를 수행할 때 모델에 입력으로 들어가는 특징(feature)에 대한 차원의 저주(curse of dimensionality) 문제가 존재함

#### 차원의 저주?

> 입력된 특징이 많아지면서 이에 따라 그 차원도 증가되면서, 결국 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상

> 따라서 적절한 특징을 선택하여 적은 수의 특징으로도 성능이 어느정도 유지, 또는 향상되는 특징 선택 기술 연구가 수행되어왔음.

### 큐러닝

Q-러닝은 강화 학습(Reinforcement Learning) 알고리즘의 일종임

![](https://velog.velcdn.com/images/kaintels/post/27769825-725f-4fd2-bd8b-72e250821aaf/image.png)
(Q 테이블, 출처 : <https://en.wikipedia.org/wiki/Q-learning>)

에이전트(학습하는 주체)가 환경과 상호작용하며 학습을 진행하고, 이 과정에서 주어진 상태에 따라 행동을 선택하며 보상(또는 벌칙)을 받음. 이를 통해 에이전트는 최적의 정책(policy), 즉 주어진 상태에서 최적의 행동을 선택하는 방법을 학습하게 됨

Q-러닝의 핵심은 각 상태에서 특정 행동을 수행할 때 얻을 수 있는 기대 보상값을 나타내는 Q-테이블(table)의 값(Q-value)을 계산하는 것임. 이를 통해 에이전트는 과거에 수행했던 행동의 결과를 바탕으로 미래의 행동을 결정하게 됨. 처음에는 Q테이블의 값은 0으로 초기화되어 있지만, 학습이 진행됨에 따라 갱신됨, Q-값은 다음과 같은 수식으로 갱신할 수 있음

$Q(s,a)=(1−α)Q(s,a)+α(r(s,a)+γmax{_a′}Q(s′,a′))$

- $Q(s,a)$: 상태 $s$에서 행동 $a$를 취했을 때의 기대 보상
- $α$: 학습률 (learning rate)
- $γ$: 할인율 (discount factor), 장기적인 보상을 고려하는 정도 $r(s,a)$
- $r(s,a)$: 현재 상태 $s$에서 행동 $a$를 했을 때 얻은 즉각적인 보상
- $s′$ : 다음 상태

이 알고리즘은 경험을 통해 점진적으로 최적의 Q-값을 학습하며, 에이전트는 이를 이용해 장기적인 보상을 극대화하는 방식으로 행동함

### 꿀벌 군집 최적화

BSO는 자연에서 벌의 먹이 찾기 행동을 모방한 군집 지능(Swarm Intelligence) 기반 메타휴리스틱 알고리즘. 이 알고리즘은 여러 인공 벌들이 협력하여 최적화 문제를 해결하는 방식으로 동작함

![](https://velog.velcdn.com/images/kaintels/post/ca80618c-6a54-4f57-a2a0-529f870157e1/image.png)

일꾼들이 자원을 채집해 보관장소에 가져다 놓듯이, 벌집의 벌의 행동을 모방한 알고리즘이라고 생각하면 편함

BSO는 다음의 주요 단계로 구성됨

1. 탐색 영역 결정
2. 벌들의 탐색 과정: 각 벌은 탐색 영역(여기서는 특징 선택할 범위)에서 지역 탐색을 수행하여 최적의 해를 탐색
3. 참조 솔루션 선택: 벌들이 찾은 최적의 해를 바탕으로 새로운 참조 솔루션을 선택하고, **이를 다른 벌들에게 공유**, 이를 통해 전역 최적화 방향으로 탐색

BSO는 탐색을 통해 솔루션 공간을 광범위하게 검색하는 **(탐색, exploration)**과 현재까지 얻은 최적의 해를 더욱 세밀하게 개선하는 **(탐험, exploitation)** 사이의 균형을 맞추어 활동함.

이 알고리즘은 탐색 영역의 크기를 제어하는 매개변수인 $flip$과 탐색 반복 횟수를 제어하는 $MaxChances$와 같은 변수들을 통해 탐색 과정에서 지역 최적해에 갇히지 않도록 함. 이를 통해 전역 최적해를 찾아가는 효율적인 탐색 방법을 구현함.

이 두 이론은 논문에서 제시된 QBSO-FS 알고리즘에서 결합되어, BSO의 탐색 과정을 Q-러닝을 통해 더욱 효율적으로 수행할 수 있음. Q-러닝을 활용해 각 벌들이 탐색 과정에서 얻은 경험을 바탕으로 보다 나은 솔루션을 선택할 수 있게 됨.

## 2. 실험 결과

저자들이 업로드한 논문의 정확도 결과는 아래와 같음

![](https://velog.velcdn.com/images/kaintels/post/9c9db505-e4cc-44e6-884b-7d3dcefded55/image.png)

PSO는 파티클 군집 최적화, HBBEPSO는 Hybrid Binary Bat Enhanced Particle Swarm Optimization. GA는 일반적인 유전 알고리즘

거의 대다수의 정확도 결과가 제안하는 방법이 효과적인것을 알 수 있음

## 사용방법

논문의 깃허브 링크를 토대로 main.py를 실행하면 됨. [Github](https://github.com/amineremache/qbso-fs)

```python
from fs_data import FSData

if __name__=="__main__":

    # RL 

    alhpa = 0.1
    gamma = 0.99
    epsilon = 0.01

    # BSO

    flip = 5 # 검색 영역을 정의하는 솔루션 집합을 결정하는 값
    max_chance = 3 # 다른 검색할 영역으로 이동하기 전에 검색 영역에 부여된 기회의 수
    bees_number = 10 # 꿀벌 수
    maxIterations = 10 # 로컬 검색에 대한 반복 횟수
    locIterations = 10 # 최대 반복횟수

    # Test type

    typeOfAlgo = 1
    nbr_exec = 1
    dataset = "Iris"
    data_loc_path = "./datasets/"
    location = data_loc_path + dataset + ".csv"
    method = "qbso_simple"
    test_param = "rl"
    param = "gamma"
    val = str(locals()[param])
    classifier = "knn"

    instance = FSData(typeOfAlgo,location,nbr_exec,method,test_param,param,val,classifier,alhpa,gamma,epsilon)
    instance.run(flip,max_chance,bees_number,maxIterations,locIterations)
```

### 후기?

강화학습 방법은 이후 DQN, PPO 등 여러 방법이 있지만, 어째서 예전 방식인 Q러닝을 사용했는지에 대한 의문점이 있음. **아마도 특징 선택을 진행 할때 그 특징 차원이 그렇게 많은 것은 아니기 때문에** Q-네트워크로 근사화할 필요 없이 즉각 Q테이블에 대입하여 적용한 것이라고 생각함.
