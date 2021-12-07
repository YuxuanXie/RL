import tensorflow
import math
import torch
from torch import nn
import torch.nn.functional as F

class GBModel(nn.Module):
    def __init__(self, obs, entity_shape, action_shape):
        super().__init__()
        self.obs_shape = obs
        self.entity_shape = entity_shape
        # bs * obs
        self.obs_encoder = nn.Linear(obs, 128)

        # bs * N * entity_shape
        self.entity_encoder_q, self.entity_encoder_k, self.entity_encoder_v = [nn.Linear(entity_shape, 128) for _ in range(3)]

        self.all_encoder = nn.Linear(256, 16)

    
    def forward(self, input):
        obs = input["obs"]
        bs = obs.shape[0]
        entities = input["entities"]
        d_model = entities.shape[1]
        entities = entities.reshape(-1, self.entity_shape)

        obs_embedding = self.obs_encoder(obs)

        q, k, v = [self.entity_encoder_q(entities), self.entity_encoder_k(entities), self.entity_encoder_v(entities)]

        q = q.reshape(bs, -1, 128)
        k = k.reshape(bs, -1, 128)
        v = v.reshape(bs, -1, 128)


        entity_embedding = F.softmax((torch.matmul(q, k.permute(0,2,1)))/math.sqrt(d_model), dim=-1)
        entity_embedding = torch.matmul(entity_embedding, v)
        entity_embedding = torch.mean(entity_embedding, dim=1)

        all_embedding = torch.cat((obs_embedding, entity_embedding), dim=-1)
        output = self.all_encoder(all_embedding)
        return output


if __name__ == "__main__":
    bs = 10
    N = 5
    obs = 256
    entity_shape = 128
    action_shape = 16

    input = {
        "obs" : torch.rand((bs, obs)),
        "entities" : torch.rand((bs, N, entity_shape))
    }

    model = GBModel(obs, entity_shape, action_shape)
    output = model(input)
    print(output)

    input = {
        "obs" : torch.rand((bs, obs)),
        "entities" : torch.rand((bs, 2*N, entity_shape))
    }

    output = model(input)
    print(output)



