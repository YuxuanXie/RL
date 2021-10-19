from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, observation_shape, action_shape, model_config=None):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        hidden_size = model_config["hidden_size"] if model_config else 128
        self.hidden_layers = [hidden_size, int(hidden_size/2)]
        
        self.f1 = nn.Linear(self.observation_shape, self.hidden_layers[0])
        self.f2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        # Action head
        # Use ModuleList rather not list !!!! the later one doesnot appear in model.parameters()
        self.logits = nn.ModuleList()
        for output_size in self.action_shape:
            self.logits.append(nn.Linear(self.hidden_layers[1], output_size))

        # Value head
        self.value = nn.Sequential(
            nn.Linear(self.observation_shape, self.hidden_layers[1]),
            nn.Tanh(),
            nn.Linear(self.hidden_layers[1], 1)
        )


    def forward(self, obs):
        x = F.relu(self.f1(obs))
        x = F.relu(self.f2(x))
        logits = [head(x) for head in self.logits]
        probs = [F.softmax(each, dim=-1) for each in logits]
        value = self.value(obs)
        return probs, value
