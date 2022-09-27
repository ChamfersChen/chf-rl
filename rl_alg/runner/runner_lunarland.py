# from .memory import Memory
from rl_alg.runner.base_runner import BaseRunner
from rl_alg.utils.log_writer import LogWriter
import torch


class LunarLandRunner(BaseRunner):
    def __init__(self, config, agent, env) -> None:
        super().__init__()
        self.max_episodes = config["max_episodes"]
        self.max_timesteps = config["max_timesteps"]
        self.update_timestep = config["update_timestep"]
        self.render = config["render"]
        self.env_name = config["env_name"]
        self.log_interval = config["log_interval"]

        self.env = env
        self.agent = agent


    def running(self, log_writer:LogWriter):

        clearn_buffer = True if self.agent.__class__.__name__=="PPOPolicy" else False
        running_reward = 0
        avg_length = 0
        time_step = 0
        
        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            mean_loss = 0.
            for t in range(self.max_timesteps):
                time_step += 1
                next_state,reward,done,info = self.collect(state)
                state = next_state

                if time_step % self.update_timestep == 0:
                    mean_loss = self.agent.update()
                    if clearn_buffer:
                        self.agent.buffer.build_pool()
                    time_step = 0
                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break
            avg_length += t+1

            if i_episode % 500 == 0:
                torch.save(self.agent.policy.state_dict(), './PPO_continuous_{}.pth'.format(self.env_name))

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                log_writer.episode_write([i_episode,running_reward,avg_length,mean_loss])
                running_reward = 0
                avg_length = 0

    def collect(self, state):

        action, action_logprob = self.agent.select_action(state)
        next_state, reward, done, _ = self.env.step(action.cpu().data.numpy().flatten())
        self.agent.buffer.push(state,action,action_logprob,reward,done)

        return next_state,reward,done,_