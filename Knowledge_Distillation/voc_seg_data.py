import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import cv2
from osgeo import gdal

class VOC_SEG_Dataset(Dataset):
    def __init__(self, root, width, height, train=True, transforms=None):
        # 图像统一剪切尺寸（width, height）
        self.width = width
        self.height = height
        self.fnum = 0
        if transforms is None:
            # normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                # normalize
            ])

        self.train = train
        if train:
            txt_fname = root + "/ImageSets/Segmentation/train.txt"
        else:
            txt_fname = root + "/ImageSets/Segmentation/val.txt"
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        imgs = [os.path.join(root, "Images/myimg" + item[7:] + ".tif") for item in images]
        labels = [os.path.join(root, "SegmentationClass", item + ".tif") for item in images]
        self.imgs = self._filter(imgs, "imgs")
        self.labels = self._filter(labels, "labels")
        if train:
            print("Train：loaded " + str(len(self.imgs)) + " images and labels" + ",filtered " + str(self.fnum) + " images")
        else:
            print("Val：loaded " + str(len(self.imgs)) + " images and labels" + ", filtered " + str(self.fnum) + " images")

    def _crop(self, data, label):
        """
        切割函数，默认都是从图片的左上角开始切割。切割后的图片宽是width,高是height
        data和label都是Image对象
        """
        box = (0, 0, self.width, self.height)
        data = data.crop(box)
        label = label.crop(box)
        return data, label

    def _image_transforms(self, data, label):
        data, label = self._crop(data, label)
        data = self.transforms(data)
        # label = self._image2label(label)
        # label = self.transforms(label)
        label = np.array(label, dtype="int64")
        label = torch.from_numpy(label)
        return data, label

    def _filter(self, imgs, description):
        img = []
        i = 1
        for im in imgs:
            if self.train:
                if description == "imgs":
                    print("Train: %d images have been loaded" % i)
                else:
                    print("Train: %d labels have been loaded" % i)
            else:
                if description == "imgs":
                    print("Val: %d images have been loaded" % i)
                else:
                    print("Val: %d labels have been loaded" % i)
            i = i + 1
            if (cv2.imread(im).shape[1] >= self.height and
                    cv2.imread(im).shape[0] >= self.width):
                img.append(im)
            else:
                self.fnum = self.fnum + 1
        return img

    def __getitem__(self, index: int):
        img_path = self.imgs[index]
        label_path = self.labels[index]
        # img = cv2.imread(img_path)

        img = gdal.Open(img_path)
        img_width = img.RasterXSize  # 栅格矩阵的列数
        img_height = img.RasterYSize  # 栅格矩阵的行数
        img = img.ReadAsArray(0, 0, img_width, img_height)
        img = np.transpose(np.array(img), (1, 2, 0))
        img = Image.fromarray(np.uint8(img))

        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label = gdal.Open(label_path)
        label_width = label.RasterXSize  # 栅格矩阵的列数
        label_height = label.RasterYSize  # 栅格矩阵的行数
        label = label.ReadAsArray(0, 0, label_width, label_height)
        label = Image.fromarray(np.uint8(label))

        img, label = self._image_transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = "data"
    width = 512
    height = 512
    voc_train = VOC_SEG_Dataset(root, width, height, train=True)
    voc_val = VOC_SEG_Dataset(root, width, height, train=False)

    for data, label in voc_train:
        print(data.shape)
        print(label.shape)
        break


