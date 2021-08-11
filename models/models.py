from .birealnet import birealnet18
from .irnet import get_irnet 

def get_model(model, num_classes , insize):
    if model=="r18_bireal":
        return birealnet18(num_classes)
    else:
        return get_irnet(num_classes)

