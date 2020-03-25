from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from main.ImageUtils import  draw_rect

# Image transformations
__transforms__ = [
    transforms.Resize((128,128), Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataSet(Dataset):
    def __init__(self, transform=__transforms__, file_name="./dataset/train.txt"):
        file = open(file_name, "r")
        s = file.read().strip()
        file.close()
        self.labels = s.split("\n")
        self.transforms = self.transform = transforms.Compose(transform)

    def __getitem__(self, item):
        s = self.labels[item]
        s = s.split(",")
        image = Image.open(s[0])
        if image.mode != "RGB":
            image = to_rgb(image)
        image = self.transforms(image)
        image_type = int(s[1])
        return image, image_type

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = ImageDataSet()
    _image, _image_type = dataset[100]
    print(_image_type)
    print(_image.shape)
    draw_rect(_image.numpy())
