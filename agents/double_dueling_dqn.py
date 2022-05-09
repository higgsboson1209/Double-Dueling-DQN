#IMPORT PACKAGES
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

class dueling_DQN_Network(nn.Module):
  def __init__(self,input_shape,output_shape):

    super().__init__()
    #STATE SPACE DIMENSION
    self.input_shape=input_shape
    #ACTION SPACE DIMENSION
    self.action_space=output_shape

    self.features=nn.Sequential(
        nn.Linear(self.input_shape,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU()
    )

    self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
    )

    self.advantage_stream = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, self.action_space)
    )

  def forward(self,state):
    inp_feature=self.features(state)
    values=self.value_stream(inp_feature)
    advantages=self.advantage_stream(inp_feature)
    return values+(advantages-advantages.mean())
