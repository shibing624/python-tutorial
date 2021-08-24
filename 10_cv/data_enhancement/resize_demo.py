# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import cv2
import os
from PIL import Image


def display_cv(image_path):
    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    print(height, width)
    # 缩小图像
    size = (200, 200)
    print(size)
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # 放大图像
    fx = 1.6
    fy = 1.2
    enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    # 显示
    cv2.imshow("src", img)
    cv2.imshow("shrink", shrink)
    cv2.imshow("enlarge", enlarge)

    cv2.waitKey(0)


def display_pil(image_path):
    img = Image.open(image_path)
    # 缩小图像
    size = (200, 200)
    print(size)
    new_img = img.resize((200, 200), Image.BILINEAR)
    new_img.show()
    new_img.save('data/resize_a.png')


if __name__ == '__main__':
    # display_cv('flower.png')
    display_pil('data/flower.png')
