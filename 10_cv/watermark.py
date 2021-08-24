# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from PIL import Image, ImageDraw


# 指定要使用的字体和大小；黑体,24号
# font = ImageFont.truetype('heiti.ttf', 24)


# image: 图片  text：要添加的文本 font：字体
def add_text_watermark(image_path, text, font=None):
    image = Image.open(image_path)
    rgba_image = image.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    # (255,255,255,0):第四个是图片的透明度,值越大,越浅
    image_draw = ImageDraw.Draw(text_overlay)
    print(rgba_image)
    # 设置文本文字位置
    text_xy = (rgba_image.size[0] - 80, rgba_image.size[1] - 80)
    # 设置文本颜色和透明度
    image_draw.text(text_xy, text, fill=(87, 250, 255, 360), font=None)
    # (87,250,255,360)第四个是文字的透明度,值越大,越深
    out = Image.alpha_composite(rgba_image, text_overlay)

    return out


def add_logo_watermark(src_path, mask_path, rate=0.1):
    im = Image.open(src_path)
    mask = Image.open(mask_path)

    h, w = im.size[0], im.size[1]
    if w > h:
        limit = int(w * rate)
    else:
        limit = int(h * rate)
    mask = mask.resize((limit, int(limit * mask.size[1] / mask.size[0])))

    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer.paste(mask, (im.size[0] - 80, im.size[1] - 60))
    out = Image.composite(layer, im, layer)

    return out


if __name__ == '__main__':
    image_path = './search/data/images/car-1.jpg'
    mask_path = './search/data/images/car-2.jpg'
    im_after = add_text_watermark(image_path, 'world')
    im_after.show()
    im_after_logo = add_logo_watermark(image_path, mask_path)
    im_after_logo.show()
