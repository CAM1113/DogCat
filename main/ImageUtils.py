import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# 输入image：C * H * W,numpy数据
def draw_rect(image, top=0, left=0, width=10, height=10, is_save=False, save_path="./temp.jpg"):
    # np.transpose( xxx,  (2, 0, 1))   # 将 C x H x W 转化为 H x W x C
    image = np.transpose(image, (1, 2, 0))
    fig, ax = plt.subplots(1)
    rect = patches.Rectangle((top, left), width, height, linewidth=1, edgecolor='r', fill=False)
    ax.imshow(image)
    ax.add_patch(rect)
    plt.axis('off')
    if is_save:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    pass
