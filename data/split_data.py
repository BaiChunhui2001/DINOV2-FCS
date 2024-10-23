import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中20%的数据划分到验证集中
    split_rate = 0.3

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = cwd
    origin_image_path = os.path.join(data_root, "image/grape2")
    origin_label_path = os.path.join(data_root, "label/grape2")
    assert os.path.exists(origin_image_path), "path '{}' does not exist.".format(origin_image_path)
    assert os.path.exists(origin_label_path), "path '{}' does not exist.".format(origin_label_path)

    image_files = sorted(os.listdir(origin_image_path))
    label_files = sorted(os.listdir(origin_label_path))

    assert len(image_files) == len(label_files), "Number of images and labels do not match."

    num_samples = len(image_files)
    eval_indices = random.sample(range(num_samples), k=int(num_samples * split_rate))

    image_train_root = os.path.join("/root/autodl-tmp/fruitorigin/image", "train/grape2")
    label_train_root = os.path.join("/root/autodl-tmp/fruitorigin/label", "train/grape2")
    image_val_root = os.path.join("/root/autodl-tmp/fruitorigin/image", "val/grape2")
    label_val_root = os.path.join("/root/autodl-tmp/fruitorigin/label", "val/grape2")

    mk_file(image_train_root)
    mk_file(label_train_root)
    mk_file(image_val_root)
    mk_file(label_val_root)

    for index, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        if index in eval_indices:
            copy(os.path.join(origin_image_path, image_file), os.path.join(image_val_root, image_file))
            copy(os.path.join(origin_label_path, label_file), os.path.join(label_val_root , label_file))
        else:
            copy(os.path.join(origin_image_path, image_file), os.path.join(image_train_root, image_file))
            copy(os.path.join(origin_label_path, label_file), os.path.join(label_train_root, label_file))

        print("\rProcessing [{}/{}]".format(index + 1, num_samples), end="")  # processing bar

    print("\nProcessing done!")


if __name__ == '__main__':
    main()
