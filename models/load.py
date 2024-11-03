import torch 
from collections import OrderedDict

def load_model(model_class, model_args, weights_path=None, ignore_weights=[],device='cpu'):
    model = model_class(**model_args)
    if weights_path:
        print('Loading weights from {}.'.format(weights_path))
        weights = torch.load(weights_path,weights_only=True,map_location=device)
        weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
        for w in ignore_weights:
            weights = {k: v for k, v in weights.items() if w not in k}
        model.load_state_dict(weights, strict=False)
        model.to(device)
    return model