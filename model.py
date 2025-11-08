import torch
import torch.nn as nn

def get_model(model_name='vgg16_bn', pretrained=True, num_classes=100):
    valid_models = ['vgg11_bn', 'vgg16_bn']
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        f'cifar100_{model_name}',
        pretrained=pretrained
    )
    return model
