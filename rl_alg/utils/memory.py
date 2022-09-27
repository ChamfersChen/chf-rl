# define memmory buffer
class Memory:
    def __init__(self) -> None:
        self.actions     = []
        self.obs        = []
        self.logprobs   = []
        self.rewards     = []
        self.is_done    = []
    def clean_memory(self):
        del self.actions[:]
        del self.obs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_done[:]

