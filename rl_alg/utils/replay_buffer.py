import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device) -> None:
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.index = 0

        self.state_pool = torch.zeros(self.buffer_size, state_dim).float().to(self.device)
        self.action_pool = torch.zeros(self.buffer_size, self.action_dim).float().to(self.device)
        self.reward_pool = torch.zeros(self.buffer_size, 1).float().to(self.device)
        self.next_state_pool= torch.zeros(self.buffer_size, state_dim).float().to(self.device)
        self.done_pool = torch.zeros(self.buffer_size, 1).float().to(self.device)

    def push(self,state, action, reward, next_state, done):
        state = torch.Tensor(state).float().to(self.device)
        action = torch.Tensor(action).float().to(self.device)
        reward = torch.Tensor(reward).float().to(self.device)
        next_state = torch.Tensor(next_state).float().to(self.device)
        done = torch.Tensor(done).float().to(self.device)

        for pool, item in zip(
            [self.state_pool,self.action_pool,self.reward_pool,self.next_state_pool,self.done_pool],
            [state,action, reward, next_state,done]
        ):
            self.index = self.index%self.buffer_size
            pool[self.index] = item
        
        self.index += 1

    def sample(self, batch_size):
        if self.index < self.buffer_size:
            index_list = np.random.choice(self.index, size=batch_size, replace=False)
        else:
            index_list = np.random.choice(self.buffer_size, size=batch_size, replace=False)
        
        bs_state, bs_action, bs_reward, bs_next_state, bs_done = \
            self.state_pool[index_list],self.action_pool[index_list], self.reward_pool[index_list],self.next_state_pool[index_list], self.done_pool[index_list]
        
        return bs_state, bs_action, bs_reward, bs_next_state, bs_done