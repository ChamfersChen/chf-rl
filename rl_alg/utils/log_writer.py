import json
import os

class LogWriter:
    def __init__(self,log_path, env_name, alg_name) -> None:
        self.env_name = env_name
        self.alg_name = alg_name
        self.episode_record_key = ["env_name", "alg_name", "episode","avg_episode_reward","avg_episode_length","loss"]
        
        self.episode_record_file = os.path.join(log_path, "episode_record.txt")

    def open_writer(self):
        self.episode_writer = open(self.episode_record_file,"a")
    
    def episode_write(self, data):
        epicode_record = dict(zip(self.episode_record_key,[self.env_name,self.alg_name]+data))
        self.episode_writer.write("%s\n"%json.dumps(epicode_record))
        self.episode_writer.flush()
    def close_writer(self):
        self.episode_writer.close()