import tensorflow
import torch
from ppo import PPO


config = {
    "model" :  {"hidden_size" : 128},
}

alg = PPO(93, (3,3,2), config=config)
alg.load_model("./300023.pth", cuda=False)
# model = torch.jit.script(alg.model)
dummy_input = torch.randn(1, 93, requires_grad=False)
torch.onnx.export(alg.model, dummy_input, "leanring.onnx", export_params=True, opset_version=9, do_constant_folding=True, input_names = ['vector_observation'], output_names = ['action0', 'action1', 'action2', 'value'],)
