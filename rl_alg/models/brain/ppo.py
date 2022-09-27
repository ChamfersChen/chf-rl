import torch.nn as nn
import torch
from rl_alg.models.brain.policy import Policy
from rl_alg.models.base_net.actor_critic_net import ActorCriticNet
from rl_alg.utils.rollout_buffer import RolloutBuffer

class PPOPolicy(Policy):
    def __init__(self, 
                state_space,
                action_space, 
                config
                ):
        self.lr = config["lr"]
        self.betas = config["betas"]
        self.gamma = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.K_epochs = config["K_epochs"]
        self.action_std = config["action_std"]
        self.buffer_size = config["buffer_size"]
        self.device = config["device"]

        action_dim = action_space.shape[0] or action_space.n
        state_dim = state_space.shape[0]
        

        self.buffer = RolloutBuffer(self.buffer_size,state_dim,action_dim,device=self.device)
        self.policy = ActorCriticNet(state_dim, action_dim, self.action_std)
        self.policy_old = ActorCriticNet(state_dim, action_dim, self.action_std)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.policy_old.act(state)

    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = [None]*self.buffer_size
        discounted_reward = 0
        for i in range(self.buffer_size):
            reward = self.buffer.reward_pool[-(i+1)]
            is_terminal = self.buffer.done_pool[-(i+1)]
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards[-(i+1)]= discounted_reward

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.squeeze(self.buffer.state_pool).detach()
        old_actions = torch.squeeze(self.buffer.action_pool).detach()
        old_logprobs = torch.squeeze(self.buffer.logprob_pool).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            loss = loss.mean()
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.item()
