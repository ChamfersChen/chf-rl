
from collections import OrderedDict
import torch.nn as nn
import torch
ACTIVATION_DICT={
    "ReLU": nn.ReLU()
}
class QNet(nn.Module):
    def __init__(self, 
                state_dim, 
                action_dim, 
                hiden_layer_list=[64,64],
                hiden_activatie_fn = "ReLU",
                output_activate_fn=None):
        
        super(QNet,self).__init__()

        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.hiden_layer_list   = hiden_layer_list
        self.hiden_activate_fn  = hiden_activatie_fn
        self.output_activate_fn = output_activate_fn

        layer_size_list = [self.state_dim]+self.hiden_layer_list
        layer_list = []

        for i in range(1,len(layer_size_list)):
            layer_list.append(
                ("linear-{}".format(i),nn.Linear(layer_size_list[i-1],layer_size_list[i]))
                )
            layer_list[-1][1].weight.data.normal_(0,0.01)
            layer_list.append(
                ("{}-{}".format(self.hiden_activate_fn,i),ACTIVATION_DICT[self.hiden_activate_fn])
                )
        
        layer_list.append(
            ("linear-{}".format(i+1),nn.Linear(layer_size_list[i],self.action_dim))
            )
        layer_list[-1][1].weight.data.normal_(0,0.01)
        if self.output_activate_fn:
            layer_list.append(
                ("{}-{}".format(self.output_activate_fn,i+1),ACTIVATION_DICT[self.output_activate_fn])
                )

        self.net = nn.Sequential(OrderedDict(layer_list))
        
    def forward(self,state):
        return self.net(state)

if __name__ == '__main__':

    net = QNet(state_dim=8,action_dim=4)
