import torchvision.transforms.functional as TF
from PIL import Image


def open_image(path):
    image = Image.open(path)
    image = TF.to_tensor(image).unsqueeze_(0)
    return image