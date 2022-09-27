from operator import index
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device) -> None:
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.index = 0

        self.build_pool()

    def build_pool(self):
        self.state_pool = torch.zeros(self.buffer_size, self.state_dim).float().to(self.device)
        self.action_pool = torch.zeros(self.buffer_size, self.action_dim).float().to(self.device)
        self.logprob_pool = torch.zeros(self.buffer_size, 1).float().to(self.device)
        self.reward_pool = torch.zeros(self.buffer_size, 1).float().to(self.device)
        self.done_pool = torch.zeros(self.buffer_size, 1).float().to(self.device)


    def push(self,state, action, logprob, reward, done):
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).float().to(self.device)
        logprob = torch.tensor(logprob).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)

        for pool, item in zip(
            [self.state_pool,self.action_pool,self.logprob_pool,self.reward_pool,self.done_pool],
            [state,action, logprob,reward,done]
        ):
            self.index = self.index%self.buffer_size
            pool[self.index] = item
        
        self.index += 1