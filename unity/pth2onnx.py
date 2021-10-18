import torch
from ppo import PPO

alg = PPO(93, (3,3,2))
# alg.load_model("results/model/2021-10-16-18-04-22/900000.pth")
# model = torch.jit.script(alg.model)
dummy_input = torch.randn(1, 93, requires_grad=False)
torch.onnx.export(alg.model, dummy_input, "leanring.onnx", export_params=True, input_names = ['modelInput'], output_names = ['modelOutput'],)
