import torch

from args import Args
from models import build_model
from util.misc import nested_tensor_from_tensor_list 

def load_model(args, device):

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


def run_inference(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        nested_tensor = nested_tensor_from_tensor_list([image_tensor])
        outputs = model(nested_tensor)
    return outputs