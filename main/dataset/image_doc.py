import os
import pandas as pd


def generate_doc(mode="train"):
    list_folder = "./train_dataset/train"
    target_txt = "/train.txt"

    file_pre = os.getcwd() + "/train_dataset/train/"
    if mode != 'train':
        list_folder = "./test_dataset/test"
        target_txt = "/test.txt"
        file_pre = os.getcwd() + "/test_dataset/test/"

    list = os.listdir(list_folder)
    f = open(os.getcwd() + target_txt, mode="w")

    for name in list:
        if not name.endswith(".jpg"):
            continue
        file_name = file_pre + name
        file_label = 0
        if name.startswith("cat."):
            file_label = 0
        else:
            file_label = 1
        line = file_name + ",{}\n".format(file_label)
        f.write(line)
    f.close()


if __name__ == '__main__':
    generate_doc()
