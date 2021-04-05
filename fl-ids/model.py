import torch
import torch.nn as nn

class MLP(nn.Module):	
    # Defining the DNN model
    hidden_layers = [128]
    output_size = 15

    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_inputs, hidden_layers[0])
        self.activ1 = nn.ReLU()
        
#         self.layer2 = nn.Linear(hidden_layers[0], hidden_layers[1])
#         self.activ2 = nn.ReLU()
        
#         self.layer3 = nn.Linear(hidden_layers[1], hidden_layers[2])
#         self.activ3 = nn.ReLU()
        
        self.layer2 = nn.Linear(hidden_layers[0], output_size)
        
    # forward propagate input
    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        
        x = self.layer2(x)
#         x = self.activ2(x)
        
#         x = self.layer3(x)
#         x = self.activ3(x)
        
#         x = self.layer4(x)
        return x
