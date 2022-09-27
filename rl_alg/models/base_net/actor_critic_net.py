import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal, Categorical

"""
Actor-CriticN网络
"""
class ActorCriticNet(nn.Module):
    def __init__(self, obs_shape, action_shape,  action_std):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(in_features=obs_shape,out_features=64), nn.ReLU(),
            nn.Linear(in_features=64,out_features=64),nn.ReLU(),
            nn.Linear(in_features=64,out_features=action_shape),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features=obs_shape,out_features=64), nn.ReLU(),
            nn.Linear(in_features=64,out_features=64),nn.ReLU(),
            nn.Linear(in_features=64,out_features=1),
        )
        # 方差
        self.action_var = torch.full((action_shape,), action_std * action_std)
    
    def forward():
        raise NotImplementedError
    
    def act(self, state ):
        # 通过Actor网络获得动作
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)

        # 创建多元正态分布 dist
        dist = MultivariateNormal(action_mean, cov_mat)
        # 通过在dist中采样获得下一步动作分布 action
        action = dist.sample()
        # 对动作分布取对数
        action_logprob = dist.log_prob(action)

        # # 存储到memory中
        # memory.obs.append(state)
        # memory.actions.append(action)
        # memory.logprobs.append(action_logprob)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        # torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
        # Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input
        cov_mat = torch.diag_embed(action_var)
        # 生成一个多元高斯分布矩阵
        dist = MultivariateNormal(action_mean, cov_mat)


            # action_probs = self.actor(state)
            # dist = Categorical(action_probs)
        # 我们的目的是要用这个随机的去逼近真正的选择动作action的高斯分布
        action_logprobs = dist.log_prob(action)
        # log_prob 是action在前面那个正太分布的概率的log ，我们相信action是对的 ，
        # 那么我们要求的正态分布曲线中点应该在action这里，所以最大化正太分布的概率的log， 改变mu,sigma得出一条中心点更加在a的正太分布。
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
