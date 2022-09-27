from abc import  abstractmethod


class Policy:
    def __init__(self) -> None:
        pass
    @abstractmethod
    def select_action(self, state, memory):
        pass
    @abstractmethod
    def update(self):
        pass