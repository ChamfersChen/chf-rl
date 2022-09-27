import torch
import torch.nn as nn
import numpy as np
from rl_alg.models.brain.policy import Policy
from rl_alg.models.base_net.qnet import QNet
from rl_alg.utils.replay_buffer import ReplayBuffer

class DQNPolicy(Policy):
    def __init__(self, 
                state_space, 
                action_space, 
                config) -> None:
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = config["lr"]
        self.buffer_size = config["buffer_size"]
        self.device = config["device"]
        self.epsilon = config["epsilon"]
        self.update_steps = config["update_steps"]
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        # self.is_continuous_action_space = config["is_continuous_action_space"]

        self.state_dim = self.state_space.shape[0] or self.state_space.n
        self.action_dim = self.action_space.shape[0] or self.action_space.n

        self.eval_net = QNet(self.state_dim,self.action_dim)
        self.target_net = QNet(self.state_dim,self.action_dim)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.learn_step_counter = 0
        self.replay_buffer = ReplayBuffer(self.buffer_size,self.state_dim,1,self.device)
   

    def select_action(self, state):
        if np.random.random()<self.epsilon:
            action = self.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device)
            action_value = self.eval_net(state)
            action = torch.max(action_value,0)[1].item()
        return action

    def update(self):
        for _ in range(self.update_steps):
            self.learn_step_counter += 1
            batch_state, batch_action, batch_reward, batch_next_state, _ =\
                self.replay_buffer.sample(self.batch_size) 

            q_eval=self.eval_net(batch_state)
            q_eval = torch.gather(q_eval,1,batch_action.long())

            q_next = self.target_net(batch_next_state)
            q_target = batch_reward+self.gamma*q_next.max(1)[0].view(self.batch_size,1)
            
            loss = self.loss_fn(q_eval,q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()