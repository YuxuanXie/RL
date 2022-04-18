import tensorflow
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ScaleDotProductionAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        d_model = q.shape[-1]
        entity_embedding = F.softmax((torch.matmul(q, k.permute(0,2,1)))/math.sqrt(d_model), dim=-1)
        entity_embedding = torch.matmul(entity_embedding, v)
        entity_embedding = torch.mean(entity_embedding, dim=1)
        return entity_embedding

class GBModel(nn.Module):
    def __init__(self, obs, entity_shape, action_shape):
        super().__init__()
        self.obs_shape = obs
        self.action_shape = action_shape
        self.entity_shape = entity_shape
        self.obs_embedding_size, self.entity_embedding_size = 128, 128
        self.all_embedding_size = 128
        self.rnn_size, self.sequence_length = 128, 1


        # bs * obs
        self.obs_encoder = nn.Linear(obs, self.obs_embedding_size)

        # bs * N * entity_shape
        self.entity_encoder_q, self.entity_encoder_k, self.entity_encoder_v = [nn.Linear(entity_shape, self.entity_embedding_size) for _ in range(3)]

        self.attention = ScaleDotProductionAttention()

        self.all_encoder = nn.Linear(self.obs_embedding_size + self.entity_embedding_size, self.all_embedding_size)

        self.rnn = nn.LSTM(input_size=self.all_embedding_size, hidden_size=self.rnn_size, num_layers=self.sequence_length)

        self.logits = nn.Linear(entity_shape, self.action_shape)

    def init_hidden(self, batch_size):
	    return (autograd.Variable(torch.randn(self.sequence_length, batch_size, self.rnn_size)), autograd.Variable(torch.randn(self.sequence_length, batch_size, self.rnn_size)))

    
    def forward(self, input, hc):
        obs = input["obs"]
        entities = input["entities"]
        bs = obs.shape[0]
        entities = entities.reshape(-1, self.entity_shape)

        obs_embedding = F.relu(self.obs_encoder(obs))

        q, k, v = [self.entity_encoder_q(entities), self.entity_encoder_k(entities), self.entity_encoder_v(entities)]
        q = q.reshape(bs, -1, self.entity_embedding_size)
        k = k.reshape(bs, -1, self.entity_embedding_size)
        v = v.reshape(bs, -1, self.entity_embedding_size)
        entity_embedding = self.attention(q, k, v)

        all_embedding = torch.cat((obs_embedding, entity_embedding), dim=-1)
        core = F.relu(self.all_encoder(all_embedding))

        # packed_input = pack_padded_sequence(output, self.sequence_length)
        core = core.unsqueeze(0)
        core, hct = self.rnn(core, hc)
        core = core.squeeze(0)
        logits = self.logits(core)
        

        return logits, hct


if __name__ == "__main__":
    bs = 10
    N = 5
    obs = 256
    entity_shape = 128
    action_shape = 16

    model = GBModel(obs, entity_shape, action_shape)
    input = {
        "obs" : torch.randn((bs, obs)),
        "entities" : torch.randn((bs, N, entity_shape)),
    }

    hc = model.init_hidden(bs)
    output = model(input, hc)
    print(output)



