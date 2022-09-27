

from abc import abstractmethod


class BaseRunner(object):
    """
    设置整个训练的流程
    """
    def __init__(self) -> None:
       pass


    
    @abstractmethod
    def running(self):
        """主要训练流程: 与环境交互,获取训练数据存入buffer,执行更新policy,"""
        raise NotImplementedError 
    
    @abstractmethod
    def collect(self):
        """收集训练数据"""
        raise NotImplementedError