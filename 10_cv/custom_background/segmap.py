# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models


def decode_segmap(image, nc=21):
    """
    函数：将 2D 分割图像转换为 RGB 图像，其中每一个标签被映射到对应的颜色.
    :param image:
    :param nc:
    :return:
    """
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (255, 255, 255), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (255, 255, 255),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path, show_orig=True, device='cpu'):
    """
    图像预处理
    :param net:
    :param path:
    :param show_orig:
    :param device:
    :return:
    """
    img = Image.open(path)
    if show_orig:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(device)
    out = net.to(device)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    return rgb


def change_background_image(foreground_img_file, background_img_file, rgb):
    # 背景融合(Alpha blending)实现
    # alpha blending to customize the background of the image

    # Read the images
    foreground = cv2.imread(foreground_img_file)
    background = cv2.imread(background_img_file, cv2.IMREAD_COLOR)
    background = cv2.resize(background, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    alpha = rgb  # 2.3

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    out = cv2.add(foreground, background)

    # Save/download image
    cv2.imwrite('org_plus_cust_bkg_img.png', out)
    return out


def whiten_background(foreground_img_file, rgb):
    img = cv2.imread(foreground_img_file)
    # whiten the background of the image
    mask_out = cv2.subtract(rgb, img)
    mask_out = cv2.subtract(rgb, mask_out)
    mask_out[rgb == 0] = 255

    # Display the result
    numpy_horizontal_concat = np.concatenate((img, mask_out), axis=1)
    # Save/download the resulting image
    cv2.imwrite('org_plus_white_bkg_image.jpeg', numpy_horizontal_concat)
    return mask_out


def remove_background(foreground_img_file, rgb):
    img = cv2.imread(foreground_img_file)
    # whiten the background of the image
    mask_out = cv2.subtract(rgb, img)
    mask_out = cv2.subtract(rgb, mask_out)
    mask_out[rgb == 0] = 255

    b_channel, g_channel, r_channel = cv2.split(mask_out)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # 最小值为0, alpha=0表示透明，不可见，仅png图片支持显示
    alpha_channel[np.where(b_channel == 255)] = 0
    out = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    # Save/download the resulting image
    cv2.imwrite('rebg.png', out)
    return mask_out


def blur_background(foreground_img_file, rgb):
    # Read the images
    foreground = cv2.imread(foreground_img_file)

    # Create a Gaussian blur of kernel size 7 for the background image
    blurred_image = cv2.GaussianBlur(foreground, (7, 7), 0)
    # Convert uint8 to float
    foreground = foreground.astype(float)
    blurred_image = blurred_image.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, blurred_image)
    # Add the masked foreground and background
    out = cv2.add(foreground, background)

    # Save/download the resulting image
    cv2.imwrite('res_blur.png', out)
    return out


def grayscale_background(foreground_img_file, rgb):
    # Load the foreground input image
    foreground = cv2.imread(foreground_img_file)

    # Resize image to match shape of R-band in RGB output map
    foreground = cv2.resize(foreground, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    # Create a background image by copying foreground and converting into grayscale
    background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # convert single channel grayscale image to 3-channel grayscale image
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)
    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background
    out = cv2.add(foreground, background)
    # Save image
    cv2.imwrite('res_gray.png', out)
    return out


if __name__ == '__main__':
    # 加载 deeplabv3_resnet101 模型
    dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    foreground_img_file = './data/bicycle-1.jpg'
    background_img_file = './data/field-1.jpg'
    rgb = segment(dlab, foreground_img_file, show_orig=False, device='cpu')
    ## If there are multiple labeled objects in the image, use the below code to have only the target as the foreground
    rgb[rgb != 255] = 0

    remove_background(foreground_img_file, rgb)

    # replace background image
    change_background_image(foreground_img_file, background_img_file, rgb)

    # whiten background image
    whiten_background(foreground_img_file, rgb)

    blur_background(foreground_img_file, rgb)
    grayscale_background(foreground_img_file, rgb)
