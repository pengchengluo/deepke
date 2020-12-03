from collections import OrderedDict
from typing import Dict, Iterable, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F



# basic format hook function
def module_hook(module: nn.Module, input: Tensor, output: Tensor):
    # For nn.Module objects only.
    pass

def tensor_hook(grad: Tensor):
    # For Tensor objects only.
    # Only executed during the *backward* pass!
    pass





class VerboseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128, 50)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(50, 10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(10, 3)),
        ]))

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x):
        return self.model(x)

# demo1: 模型执行详情
model = VerboseModule()
dummy_input = torch.randn(32, 128)
print(model(dummy_input).shape)






class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])['model.'+layer_id]
            layer.register_forward_hook(self._save_outputs_hook(layer_id))

    def _save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features

# 特征提取
model_feature = FeatureExtractor(model, layers=['relu1', 'fc2'])
features = model_feature(dummy_input)
print({name: output.shape for name, output in features.items()})







# 梯度裁剪
def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))
    return model

clipped_model = gradient_clipper(model, 0.01)
pred = clipped_model(dummy_input)
loss = nn.CrossEntropyLoss()
target = torch.empty(32, dtype=torch.long).random_(3)
output = loss(pred, target)
output.backward()
print(clipped_model.model.fc1.bias.grad.data)