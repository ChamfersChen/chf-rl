# 强化学习

## Tianshou框架


### 1 概述

在强化学习中，智能体(Agent)通过与环境进行交互(interact)来提升\训练自己。

![RL Loop](https://tianshou.readthedocs.io/en/master/_images/rl-loop.jpg)

在强化训练的流程中，有3个主要的数据流：
1. 从智能体到环境(Agent to environment): `action`会由智能体产生，并送入环境中；
2. 从环境到智能体(Environment to agent): `env.step`获得动作`action`，并返回结果元组`(observation, reward, done, info)`；
3. 智能体和环境进行交互来训练智能体: 由交互生成的数据会被存储起来，并送入智能体的训练器(Trainer)

![Pip Line](https://tianshou.readthedocs.io/en/master/_images/pipeline.png)

设置向量化的环境(vectorized environments)
Policy (with neural network) 
Collector (with buffer)
Trainer 【可以通过自定义，不能不需要Tianshou的框架】

### 2 创建一个环境

首先通过`gym`创建一个简单的环境`gym.make(environment_name)`
```python
import gym
import tianshou as ts

env = gym.make('CartPole-v0')
```
`CartPole-v0`环境介绍：
- state: 车的位置；车的速度；木棒的角度；木棒顶部的速度
- action: [0,1,2]中的一个值，分别对应车左移；车不移动；车右移
- reward: 车每持续一个时间步，`reward` a+1
- done: CartPole 出界或者超时
- info: 额外的信息
一个好的策略能够在这个环境中获得高分

### 3 设置向量化环境
Tianshou支持对任何算法进行向量化
- `DummyVectorEnv`: 使用单线程的for循环的序列化版本(the sequential version, using a single-thread for-loop);

- `SubprocVectorEnv`: 使用python多进程模块船舰并行环境 (use python multiprocessing and pipe for concurrent execution);

- `ShmemVectorEnv`: use share memory instead of pipe based on SubprocVectorEnv;

- `RayVectorEnv`: use Ray for concurrent activities and is currently the only choice for parallel simulation in a cluster with multiple machines. It can be used as follows: (more explanation can be found at Parallel Sampling)

```python 
train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])
```

### 4 建立网络
Tianshou支持任何自定义的PyTorch网络和优化器。但是，输入和输出需要匹配Tianshou的API
```python
import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
```
自定义网络的输入和输出的规则：
- 输入(Input): opservation `ops` (可以是`np.ndarry`, `torch.Tensor`, `dict`, `self-defined class`); hidden state `state` (for RNN usage), and other information `info` provided by the environment.
- 输出(Output): some `logits`, the next hidden state `state`. The logits could be a tuple instead of a torch.Tensor, or some other useful variables or results during the policy forwarding procedure. It depends on how the policy class process the network output. 

### 5 设置Policy
我们将上述的`net`和`optim`作为policy的超参数，来定义Policy。
例如，定义一个`DQNPolicy`:
```python
policy = ts.policy.DQNPolicy(net, optim,discount_factor=0.9, estimation_step=3, target_update_freq=320)

```
### 6 设置Collector
Collector是Tianshou的一个核心概念。它能够让policy很方便的在不同类型的环境中进行交互。在每个步骤中，收集器将让Policy执行（至少）指定数量的步骤(steps)或回合(episodes)，并将数据存储在回放缓冲区(Buffer)中。

## Tianshou基础概念
![Tianshou控制流程](https://tianshou.readthedocs.io/en/master/_images/concepts_arch2.png)

### 1 Batch
Tianshou 提供`Batch`作为内部数据结构来向其他方法传递任何类型的数据。
通过字典的形式来存储不同类型的数据。可以用键值对(key-value pair)来定义一个`Batch`
```python
>>> import torch, numpy as np
>>> from tianshou.data import Batch
>>> data = Batch(a=4, b=[5, 5], c='2312312', d=('a', -2, -3))
>>> # the list will automatically be converted to numpy array
>>> data.b
array([5, 5])
>>> data.b = np.array([3, 4, 5])
>>> print(data)
Batch(
    a: 4,
    b: array([3, 4, 5]),
    c: '2312312',
    d: array(['a', '-2', '-3'], dtype=object),
)
>>> data = Batch(obs={'index': np.zeros((2, 3))}, act=torch.zeros((2, 2)))
>>> data[:, 1] += 6
>>> print(data[-1])
Batch(
    obs: Batch(
             index: array([0., 6., 0.]),
         ),
    act: tensor([0., 6.]),
)
```


### 2 Buffer
`ReplayBuffer` stores data generated from interaction between the policy and environment. 
`ReplayBuffer` 可以被认为是Batch的一种特殊的形式或组织。它通过循环队列来存储Batch中的数据。

The current implementation of Tianshou typically use 7 reserved keys in Batch:
- `obs` the observation of step t ;
- `act` the action of step t ;
- `rew` the reward of step t ;
- `done` the done flag of step t ;
- `obs_next` the observation of step t+1 ;
- `info` the info of step t (in gym.Env, the env.step(） function returns 4 arguments, and the last one is info);
- `policy` the data computed by policy in step t;

其他类型的`ReplayBuffer`:
- `PrioritizedReplayBuffer` (based on Segment Tree and numpy.ndarray) 
- **`VectorReplayBuffer`**: 加入不同的回合数据，但是不会丢失时间顺序(add different episodes’ data but without losing chronological order)

### 3 Policy
Tianshou的目标是模块化强化学习算法。Tianshou包含多个Policy类。所有的Policy类都必须继承`BasePolicy`.

一个典型的Policy类有以下部分：
- `__init__()`: 初始化Policy，包括网络的初始化等 (initialize the policy, including copying the target network and so on);

- `forward()`: 根据给定的观测计算动作(compute action with given observation);

- `process_fn()`: 预处理来自buffer的数据(pre-process data from the replay buffer);

- `learn()`: 根据数据来更新Policy (update policy with a given batch of data).

- `post_process_fn()`: update the buffer with a given batch of data.

- `update()`: the main interface for training. This function **samples data from buffer**, **pre-process data (such as computing n-step return)**, **learn with the data**, and finally **post-process the data** (such as updating prioritized replay buffer); in short, *process_fn -> learn -> post_process_fn*.

### 4 Collector
`collect()` is the main method of Collector: it let the policy perform a specified number of step `n_step` or episode `n_episode` and store the data in the replay buffer, then return the statistics of the collected data such as episode’s total reward.