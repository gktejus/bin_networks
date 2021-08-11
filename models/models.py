from .birealnet import *
from .irnet import * 

def get_model(model, num_classes , insize):
    if model=="r18_bireal":
        return birealnet18(num_classes)
    else:
        return get_irnet(num_classes)
    


