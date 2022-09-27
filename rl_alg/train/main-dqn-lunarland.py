import gym
from rl_alg.utils.log_writer import LogWriter
from rl_alg.runner.runner_lunarland import LunarLandRunner
from rl_alg.models.brain.dqn import DQNPolicy
import toml


def main():
    # parameters
    config = toml.load("./config/dqn-config.toml")["config"]
    
    print(config)
    env_name = config["env_name"]
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_space = env.observation_space # or env.observation_space.n
    action_space = env.action_space # or env.action_space.n
    dqn = DQNPolicy(state_space, action_space,config)

    # define logger
    writer = LogWriter(log_path="./",env_name=env_name,alg_name="ppo")
    writer.open_writer()
    # train 
    runner = LunarLandRunner(config,dqn,env)
    runner.running(writer)
    
    writer.close_writer()
if __name__ == '__main__':
    print("Hello LunarLand continuous")
    main()