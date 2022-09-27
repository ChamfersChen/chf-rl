import gym
from rl_alg.utils.log_writer import LogWriter
from rl_alg.runner.runner_lunarland import LunarLandRunner
from rl_alg.models.brain.ppo import PPOPolicy
import toml


def main():
    # get parameters from config
    config = toml.load(".\config\ppo-config.toml")["config"]

    # creating environment
    env_name = config["env_name"]
    env = gym.make(env_name)
    
    # get space for build policy
    state_space = env.observation_space
    action_space = env.action_space  
    # create policy
    ppo = PPOPolicy(state_space, action_space,config)

    # define logger
    writer = LogWriter(log_path="./ppo_log/",env_name=env_name,alg_name="ppo")
    writer.open_writer()
    
    # define trainer 
    runner = LunarLandRunner(config,ppo,env)
    # do run
    runner.running(writer)
    
    writer.close_writer()
if __name__ == '__main__':
    print("Hello LunarLand continuous")
    main()