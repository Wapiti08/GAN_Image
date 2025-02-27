from models.mobile_net import MobileNet_wrapper
from models.resnet import ResNet
from models.vgg16 import Vgg16_wrapper
from models.xception import Xception_wrapper


def get_model_from_name(model_name):
    print("Using model {}".format(model_name))
    if model_name.lower() == "vgg16":
        return Vgg16_wrapper()
    elif model_name.lower() == "xception":
        return Xception_wrapper()
    elif model_name.lower() == "mobilenet":
        return MobileNet_wrapper()
    elif model_name.lower() == "resnet":
        return ResNet()
    else:
        raise NotImplementedError
